# SPDX-License-Identifier: Apache-2.0
"""Async engine-driven data transfer context for multiprocess worker adapters."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContext,
    gather_paged_kv_to_cpu,
)
from lmcache.v1.multiprocess.transfer_context.pickle import EngineDrivenContextPickle
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
    IPCEvent,
    _single_group_block_ids,
)

logger = init_logger(__name__)

# Number of background threads used to run the deferred store pipeline for the
# async engine-driven path, and therefore how many stores may hold staging
# buffers at once.
#
# One, because overlapping stores buy nothing here and cost host memory: the
# gathers all serialize on the single copy stream and the commits on
# _commit_lock, while each in-flight store holds its own pinned staging.
# Pinned host memory is never returned to the OS (torch's caching host
# allocator keeps freed blocks), so N workers raise the *permanent* host
# high-water mark N-fold -- for a hybrid-attention model storing 440 KiB per
# token per rank, one 8192-token store already stages 3.4 GiB.
DEFAULT_ENGINE_DRIVEN_COMMIT_WORKERS = 1


class _StorePlan(NamedTuple):
    """Where one deferred store gathers to, and what its commit sends.

    Group-major throughout: single-group contexts use one entry, multi-group
    contexts one per registered LMCache KV group, so the gather loop is the
    same code in both cases.

    Attributes:
        targets: Per-group gather destinations — server-reserved SHM views in
            SHM mode, pooled pinned staging buffers in pickle mode. An empty
            per-group list means that group has nothing to write.
        chunk_indices: Per-group chunk positions to gather (SHM mode writes
            only the chunks the server reserved), or ``None`` for that group's
            full chunk sequence.
        commit_arg: The ``chunks`` argument for ``commit_store``: a flat chunk
            list for single-group, the group-major lists for multi-group
            pickle, and ``[]`` in SHM mode where the data is already in the
            server's slots.
    """

    targets: list[list[torch.Tensor]]
    chunk_indices: list[list[int] | None]
    commit_arg: "list[torch.Tensor] | list[list[torch.Tensor]]"


# TODO: async retrieve path TBD, but benefit might be very limited
class AsyncEngineDrivenTransferContext(EngineDrivenTransferContext):
    """Fully async engine-driven data transfer context (store-only async).

    "Store-only async" means ``submit_store`` returns an *unresolved* future
    that resolves only after the deferred gather (GPU->CPU copy) and commit
    (CPU->server) both complete off the forward thread, while
    ``submit_retrieve`` stays synchronous and returns an already-resolved
    future exactly as on the base context.

    Inherits :class:`EngineDrivenTransferContext` and reuses its
    ``register()`` (layout / SHM registration, no stream dependency) and
    ``submit_retrieve()`` (this path does not change retrieve). Only the store
    is made async.

    Store is three-phase, all executed entirely in a background thread:

    1. prepare: call prepare_store() (or prepare_store_grouped() for a
       multi-group registration) to negotiate buffers with the server — the
       costliest step in pickle mode due to the synchronous RPC round-trip.
    2. gather: wait for the forward event on the copy stream, then enqueue
       GPU->CPU copies. When SHM buffers are available, gather writes directly
       into SHM views (matching the synchronous path). Otherwise, gather
       targets pinned staging buffers.
    3. commit: wait for gather completion (via a recorded CUDA event), then
       perform commit_store() and resolve the returned future.

    Multi-group (hybrid KV cache) registrations run the same three phases: each
    group is gathered from its own layers with its own ``blocks_in_chunk`` and
    sliding-window coverage, all enqueued in order on the single copy stream, so
    one recorded event covers every group. Commit is all-or-nothing — a key is
    never committed with only some of its groups gathered.

    ``submit_store`` performs only O(1) work on the forward thread (registration
    check and block-id flattening) before submitting all three phases to the
    background ``commit_executor``, so the forward thread is never blocked by
    the RPC round-trip or gather kernel launch latency.

    This class is only instantiated by the factory when the device is
    async-capable, so the constructor creates async resources unconditionally;
    there is no ``self._async_capable`` flag.
    """

    def __init__(
        self,
        commit_workers: int = DEFAULT_ENGINE_DRIVEN_COMMIT_WORKERS,
    ) -> None:
        """Initialize the async context and create its async resources.

        Args:
            commit_workers: Number of background threads running the deferred
                store pipeline; see :data:`DEFAULT_ENGINE_DRIVEN_COMMIT_WORKERS`
                for why raising it costs host memory.
        """
        super().__init__()
        self._commit_workers = max(1, int(commit_workers))
        self._copy_stream: Any = torch_dev.Stream()
        self._commit_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=self._commit_workers,
            thread_name_prefix="lmcache_engine_driven_commit",
        )
        self._inflight_lock = threading.Lock()
        self._inflight_gather_events: set[Any] = set()
        # Tracks gather tasks that have been submitted to _commit_executor but
        # have not yet recorded their CUDA event. flush_inflight_stores waits
        # on all of these before synchronizing _inflight_gather_events, closing
        # the window where preemption could overwrite paged KV blocks before an
        # in-flight gather has had a chance to record its CUDA event.
        self._pending_stores: set[threading.Event] = set()
        # Serializes commit_store calls across worker threads, since the
        # underlying ZMQ socket is not thread-safe. A no-op at the default
        # commit_workers=1, but callers may raise that.
        self._commit_lock = threading.Lock()
        self._staging_pool: dict[
            tuple[tuple[int, ...], torch.dtype], list[torch.Tensor]
        ] = {}
        self._is_closing = False

    def _alloc_pinned_staging(
        self, shape: torch.Size, dtype: torch.dtype, count: int
    ) -> list[torch.Tensor]:
        """Allocate pinned (page-locked) staging tensors for GPU->CPU copies.

        Tensors are reused from the pool when available to avoid repeated
        allocations on the hot path.

        Args:
            shape: Tensor shape to allocate.
            dtype: Tensor dtype to allocate.
            count: Number of tensors needed.

        Returns:
            List of ``count`` pinned CPU tensors.
        """
        key = (tuple(shape), dtype)
        with self._inflight_lock:
            pooled = self._staging_pool.setdefault(key, [])
            staged = [pooled.pop() for _ in range(min(len(pooled), count))]
        if len(staged) == count:
            return staged

        missing = count - len(staged)
        for _ in range(missing):
            try:
                staged.append(
                    torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
                )
            except RuntimeError:
                # Graceful fallback for CPU-only / pin-memory-disabled setups.
                logger.warning(
                    "Falling back to non-pinned CPU staging buffer "
                    "(shape=%s, dtype=%s)",
                    tuple(shape),
                    dtype,
                )
                staged.append(torch.empty(shape, dtype=dtype, device="cpu"))
        return staged

    def _release_staging(self, chunks: list[torch.Tensor]) -> None:
        """Return staging tensors to the pool for reuse.

        Args:
            chunks: Tensors previously obtained from :meth:`_alloc_pinned_staging`.
        """
        if not chunks:
            return
        key = (tuple(chunks[0].shape), chunks[0].dtype)
        with self._inflight_lock:
            self._staging_pool.setdefault(key, []).extend(chunks)

    def _prepare_store_target(
        self,
        ctx: EngineDrivenContext,
        key: Any,
        instance_id: int,
        group_block_ids: list[list[int]],
        blocks_in_chunk: int,
        staged: list[list[torch.Tensor]],
    ) -> "_StorePlan | None":
        """Phase 1: negotiate server buffers and pick the gather destinations.

        Runs on a background thread. In pickle mode the prepare RPC reserves
        nothing and the gather targets are pinned staging buffers sized from the
        registered layout; in SHM mode the server hands back the exact slots to
        write, possibly only for the chunks it does not already hold.

        Args:
            ctx: The registered engine-driven context (captured by the caller so
                a concurrent ``close()`` cannot swap it mid-store).
            key: LMCache key object for the store range.
            instance_id: Worker process instance identifier.
            group_block_ids: vLLM block IDs per LMCache KV group.
            blocks_in_chunk: Paged blocks per LMCache chunk (single-group only;
                multi-group uses each group's own ``blocks_in_chunk``).
            staged: Output list, extended with one entry per group taken from the
                pinned staging pool. The caller owns returning them, and is
                handed each group as it is allocated so a failure partway through
                a multi-group allocation still releases the earlier groups.

        Returns:
            The :class:`_StorePlan` for this store, or ``None`` when the server
            already holds every chunk and there is nothing to gather or commit.

        Raises:
            RuntimeError: If the registered layout carries no shape/dtype, or
                the server returned a malformed grouped prepare response.
        """
        if self._group_states:
            return self._prepare_store_target_grouped(
                ctx, key, instance_id, group_block_ids, staged
            )

        result = ctx.prepare_store(key, instance_id)
        out_buffers, chunk_indices = result if result is not None else (None, None)
        if chunk_indices is not None and len(chunk_indices) == 0:
            return None
        if out_buffers is not None:
            return _StorePlan(
                targets=[out_buffers],
                chunk_indices=[chunk_indices],
                commit_arg=out_buffers,
            )

        layout_desc = ctx.layout_desc
        if not layout_desc.shapes:
            raise RuntimeError("engine-driven layout_desc.shapes is empty")
        if not layout_desc.dtypes:
            raise RuntimeError("engine-driven layout_desc.dtypes is empty")
        num_chunks = (
            len(chunk_indices)
            if chunk_indices is not None
            else len(group_block_ids[0]) // blocks_in_chunk
        )
        staged.append(
            self._alloc_pinned_staging(
                layout_desc.shapes[0], layout_desc.dtypes[0], num_chunks
            )
        )
        return _StorePlan(
            targets=list(staged),
            chunk_indices=[chunk_indices],
            commit_arg=staged[0],
        )

    def _prepare_store_target_grouped(
        self,
        ctx: EngineDrivenContext,
        key: Any,
        instance_id: int,
        group_block_ids: list[list[int]],
        staged: list[list[torch.Tensor]],
    ) -> "_StorePlan | None":
        """Phase 1 for a multi-group registration, one destination per group.

        Args:
            ctx: The registered engine-driven context.
            key: LMCache key object for the store range.
            instance_id: Worker process instance identifier.
            group_block_ids: vLLM block IDs per LMCache KV group.
            staged: Output list, see :meth:`_prepare_store_target`.

        Returns:
            The :class:`_StorePlan` for this store, or ``None`` when the server
            already holds every chunk of every group.

        Raises:
            RuntimeError: If the server returned a malformed grouped prepare
                response.
        """
        if isinstance(ctx, EngineDrivenContextPickle):
            # Handshake only: the pickle strategy reserves nothing at prepare,
            # so each group gathers into its own pinned staging (the groups have
            # different chunk shapes under mixed KV geometry) and the whole
            # group-major list is serialized by commit_store.
            ctx.prepare_store(key, instance_id)
            for gid, state in enumerate(self._group_states):
                staged.append(
                    self._alloc_pinned_staging(
                        state.layout_desc.shapes[0],
                        state.layout_desc.dtypes[0],
                        len(group_block_ids[gid]) // state.blocks_in_chunk,
                    )
                )
            return _StorePlan(
                targets=list(staged),
                chunk_indices=[None] * len(staged),
                commit_arg=list(staged),
            )

        result = ctx.prepare_store_grouped(key, instance_id)
        if result is None:
            raise RuntimeError(
                "PREPARE_STORE returned a malformed grouped response "
                f"for instance_id={instance_id}"
            )
        tensors, chunk_indices, group_ids = result
        if not tensors:
            return None
        targets: list[list[torch.Tensor]] = []
        indices: list[list[int] | None] = []
        for gid in range(len(self._group_states)):
            out_g, chunks_g = self._group_slots(tensors, group_ids, gid, chunk_indices)
            targets.append(out_g)
            indices.append(chunks_g)
        # The data lands in the server's own slots, so commit sends no chunks.
        return _StorePlan(targets=targets, chunk_indices=indices, commit_arg=[])

    def _gather_into(
        self,
        plan: _StorePlan,
        kv_caches: dict[str, torch.Tensor],
        group_block_ids: list[list[int]],
        blocks_in_chunk: int,
    ) -> None:
        """Phase 2: enqueue the GPU->CPU copies for every group.

        Must be called inside ``torch_dev.stream(self._copy_stream)`` and after
        the forward event has been waited on that stream. All groups enqueue on
        that one stream in order, so a single event recorded afterwards covers
        the whole store.

        Args:
            plan: The plan returned by :meth:`_prepare_store_target`.
            kv_caches: Worker KV cache tensors keyed by layer name.
            group_block_ids: vLLM block IDs per LMCache KV group.
            blocks_in_chunk: Paged blocks per LMCache chunk (single-group only).
        """
        if self._group_states:
            self._gather_groups(
                kv_caches, group_block_ids, plan.targets, plan.chunk_indices
            )
            return

        gather_paged_kv_to_cpu(
            kv_caches,
            group_block_ids[0],
            blocks_in_chunk,
            layout_hints=self._layout_hints,
            engine_kv_format=self._engine_kv_format,
            out=plan.targets[0],
            chunk_indices=plan.chunk_indices[0],
        )

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Three-phase async store (prepare, gather and commit all in background).

        Performs only O(1) work on the forward thread (registration check and
        block-id shape validation), then submits all three phases —
        prepare_store, gather (GPU->CPU), and commit — to the background
        ``commit_executor``. Returns an unresolved future that resolves only
        after all three phases complete. Multi-group registrations take the same
        path, one gather per group on the shared copy stream.

        Args:
            _request_id: External request identifier (used for logging).
            key: LMCache key object for the store range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to store, indexed by LMCache KV group id.
            _event: Synchronization event; ``wait()`` is called in background.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.

        Returns:
            An unresolved :class:`MessagingFuture` that resolves to ``True``
            on success, ``False`` on failure.

        Raises:
            RuntimeError: If register() was not called first, or ``block_ids``
                does not carry exactly one list per registered KV group.
        """
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )
        # Normalize to one block-id list per LMCache KV group so the phases
        # below are group-count agnostic. Both shape checks stay on the forward
        # thread: a caller passing the wrong number of lists is a bug that must
        # raise, not resolve to a logged False in a worker thread.
        if self._group_states:
            if len(block_ids) != len(self._group_states):
                raise RuntimeError(
                    f"got {len(block_ids)} block-id lists for "
                    f"{len(self._group_states)} registered groups"
                )
            group_block_ids = block_ids
        else:
            group_block_ids = [_single_group_block_ids(block_ids)]
        completion: MessagingFuture[bool] = MessagingFuture()
        engine_driven_context = self._engine_driven_context
        commit_executor = self._commit_executor

        # Signals when this task has recorded its CUDA event (or exited early),
        # allowing flush_inflight_stores to safely proceed.
        gather_launched = threading.Event()
        try:
            with self._inflight_lock:
                if self._is_closing:
                    completion.set_result(False)
                    return completion
                self._pending_stores.add(gather_launched)

            def _prepare_gather_and_commit() -> None:
                gather_done: Any | None = None
                ok = False
                # Pinned staging taken from the pool, per group; stays empty when
                # the gather wrote straight into the server's SHM slots. Owned
                # here rather than returned by phase 1, so the release below also
                # covers a multi-group allocation that failed partway through.
                staged: list[list[torch.Tensor]] = []
                try:
                    # --- Phase 1: prepare_store ---
                    # In pickle mode this is the costliest step (sync RPC
                    # round-trip).  Running it here keeps the forward thread free.
                    plan = self._prepare_store_target(
                        engine_driven_context,
                        key,
                        instance_id,
                        group_block_ids,
                        blocks_in_chunk,
                        staged,
                    )
                    if plan is None:
                        # All chunks are already in cache: no gather, no commit.
                        ok = True
                        return

                    # --- Phase 2: gather (GPU->CPU copy on copy stream) ---
                    with torch.inference_mode(), torch_dev.stream(self._copy_stream):
                        _event.wait(stream=self._copy_stream)

                        self._gather_into(
                            plan, kv_caches, group_block_ids, blocks_in_chunk
                        )

                        gather_done = torch_dev.Event()
                        gather_done.record(self._copy_stream)

                    with self._inflight_lock:
                        if gather_done is not None:
                            self._inflight_gather_events.add(gather_done)
                        self._pending_stores.discard(gather_launched)
                    gather_launched.set()

                    if gather_done is not None:
                        gather_done.synchronize()

                    # --- Phase 3: commit ---
                    with self._commit_lock:
                        ok = engine_driven_context.commit_store(
                            key, instance_id, plan.commit_arg
                        )

                    if not ok:
                        logger.error(
                            "Async engine-driven commit_store failed for request_id=%s",
                            _request_id,
                        )
                except Exception:
                    logger.exception(
                        "Async engine-driven store failed for request_id=%s",
                        _request_id,
                    )
                    ok = False
                finally:
                    for group_chunks in staged:
                        self._release_staging(group_chunks)
                    with self._inflight_lock:
                        if gather_done is not None:
                            self._inflight_gather_events.discard(gather_done)
                        self._pending_stores.discard(gather_launched)
                    gather_launched.set()
                    completion.set_result(ok)

            # Submitting the task is the ownership-transfer point: once it
            # succeeds, the closure is solely responsible for releasing staging
            # buffers and resolving the future. The except below therefore only
            # handles failures that occur *before* this submit.
            commit_executor.submit(_prepare_gather_and_commit)
        except Exception:
            logger.exception("Failed to submit async engine-driven store")
            with self._inflight_lock:
                self._pending_stores.discard(gather_launched)
            gather_launched.set()
            completion.set_result(False)
            return completion

        return completion

    def flush_inflight_stores(self) -> None:
        """Synchronize all in-flight gather (GPU->CPU) events.

        Called at preemption/eviction time so that vLLM cannot overwrite
        paged KV blocks before a deferred gather has finished reading them.

        Waits for all submitted-but-not-yet-launched stores to record their
        CUDA events before synchronizing those events, preventing a race where
        ``flush_inflight_stores`` returns before a background gather has
        started.
        """
        with self._inflight_lock:
            pending = list(self._pending_stores)
        for ev in pending:
            ev.wait()
        self._sync_gather_events(suppress_errors=False)

    def close(self) -> None:
        """Drain in-flight gather/commit work before closing the base context."""
        with self._inflight_lock:
            self._is_closing = True
            pending = list(self._pending_stores)
        for ev in pending:
            ev.wait()
        self._sync_gather_events(suppress_errors=True)
        self._commit_executor.shutdown(wait=True, cancel_futures=False)
        super().close()

    def _sync_gather_events(self, suppress_errors: bool = False) -> None:
        """Synchronize all in-flight gather (GPU->CPU) events.

        Args:
            suppress_errors: If True, log exceptions instead of propagating.
        """
        with self._inflight_lock:
            gather_events = list(self._inflight_gather_events)
        for event in gather_events:
            try:
                event.synchronize()
            except Exception:
                if not suppress_errors:
                    raise
                logger.exception("Failed while draining gather events")

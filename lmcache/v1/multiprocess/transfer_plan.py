# SPDX-License-Identifier: Apache-2.0
"""Driven-path-agnostic KV transfer planning.

This module owns the "what to copy" step of a multiprocess KV transfer: given
one object group's geometry and the set of memory objects to move, it produces
an ordered list of :class:`TransferPlanStep` describing every batched copy and
per-kernel-group kernel launch. The plan contains only indices, block
positions, and counts -- no tensors, pointers, streams, or IPC state -- so any
transfer path (LMCache-driven or engine-driven) can build a plan the same way
and materialize it with its own source/destination resolution and copy engine.
"""

# Standard
from dataclasses import dataclass
from itertools import islice
from typing import Callable, Generator, Sequence

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

__all__ = [
    "KernelGroupGeometry",
    "ObjectGroupGeometry",
    "KernelGroupLaunch",
    "TransferPlanStep",
    "batched_iteration_with_skip",
    "build_object_group_transfer_plan",
]


@dataclass(frozen=True)
class KernelGroupGeometry:
    """Per-kernel-group block geometry needed to plan a transfer.

    Args:
        kernel_group_id: Index of the kernel group in the layer-groups manager.
        blocks_per_chunk: Number of paged blocks holding one full LMCache
            chunk for this group.
        blocks_per_window: Number of paged blocks holding one chunk's worth of
            retained tokens for this group. Equal to ``blocks_per_chunk`` for
            full-attention groups; smaller when the group's sliding window is
            narrower than the chunk.
        tokens_to_blocks: Maps a token count to this group's block count.
            Supplied by the owner of the block-geometry math (e.g.
            ``BaseCacheContext.calculate_num_blocks``) so the planner never
            duplicates per-group slot/compression rules.
    """

    kernel_group_id: int
    blocks_per_chunk: int
    blocks_per_window: int
    tokens_to_blocks: Callable[[int], int]


@dataclass(frozen=True)
class ObjectGroupGeometry:
    """Geometry of one object group, the planning unit of a transfer.

    Args:
        object_group_id: Index of the object group in the layer-groups manager.
        kernel_groups: Geometry of every kernel group backing this object
            group, in the group's declared layout order.
        tokens_per_chunk: LMCache chunk size in tokens.
        num_chunks_in_sw: Number of trailing prefix chunks a sliding-window
            group must retrieve; negative for full-attention groups (retrieve
            everything).
    """

    object_group_id: int
    kernel_groups: tuple[KernelGroupGeometry, ...]
    tokens_per_chunk: int
    num_chunks_in_sw: int


@dataclass(frozen=True)
class KernelGroupLaunch:
    """One kernel group's slice of a batched copy.

    Args:
        kernel_group_id: Index of the kernel group to launch for.
        start_block_pos: Offset of the batch's first block in the group's
            (window-downsampled) block-id list.
        num_blocks: Number of block ids the launch consumes from
            ``start_block_pos``.
        skip_blocks: Leading blocks within the batch whose writes must be
            skipped (protects APC-shared GPU blocks under a retrieve).
    """

    kernel_group_id: int
    start_block_pos: int
    num_blocks: int
    skip_blocks: int


@dataclass(frozen=True)
class TransferPlanStep:
    """One batched copy: which objects move and how each kernel group runs.

    Args:
        start_object_idx: Index of the batch's first memory object in the
            original (pre-skip) object sequence.
        object_indices: Indices of the memory objects in this batch, in copy
            order.
        launches: One :class:`KernelGroupLaunch` per kernel group in the
            object group, in the group's declared layout order.
    """

    start_object_idx: int
    object_indices: tuple[int, ...]
    launches: tuple[KernelGroupLaunch, ...]


def batched_iteration_with_skip(
    lst: Sequence,
    batch_size: int,
    skip_count: int,
) -> Generator[tuple[int, tuple], None, None]:
    """Utility function to iterate over a list in batches with an initial skip.

    Args:
        lst: The list to iterate over.
        batch_size: The size of each batch.
        skip_count: The number of items to skip at the start of the list.

    Yields:
        Tuples of (batch_start_idx, batch) where batch is a tuple of items
        from the list, and batch_start_idx is the "original" index of the first
        item in the batch.

    Raises:
        ValueError: If batch_size is less than 1 or skip_count is negative.

    Note:
        Batch_idx is the index of the batch in the original list, accounting
        for the skipped items. For example, if skip_count is 10 and batch_size
        is 5, the first yielded batch will have batch_start_idx=10.
    """
    if batch_size < 1:
        raise ValueError("batch size must be at least one")
    if skip_count < 0:
        raise ValueError("skip_count must be non-negative")

    it = iter(lst)
    # Skip the initial items
    for _ in range(skip_count):
        next(it, None)
    batch_start_idx = skip_count
    while batch := tuple(islice(it, batch_size)):
        yield batch_start_idx, batch
        batch_start_idx += len(batch)


def _recalculate_blocks_to_skip(
    blocks_per_chunk: int,
    blocks_per_window: int,
    blocks_to_skip: int,
) -> int:
    """Re-calculate the number of blocks to skip for a batch of chunks based
    on the blocks per chunk and blocks per sliding window WHEN the window
    size is smaller than the lmcache chunk size.

    Args:
        blocks_per_chunk: The total number of blocks in one chunk for the
            current group.
        blocks_per_window: The number of blocks in the sliding window
            for the current group. Should be less than or equal to
            blocks_per_chunk.
        blocks_to_skip: The number of blocks to skip.

    Returns:
        The re-calculated number of blocks to skip for the current batch of
        chunks.
    """
    if blocks_per_chunk == blocks_per_window:
        return blocks_to_skip

    full_windows_to_skip = blocks_to_skip // blocks_per_chunk
    tail_blocks = blocks_to_skip % blocks_per_chunk
    tail_blocks_to_skip = tail_blocks - (blocks_per_chunk - blocks_per_window)
    return full_windows_to_skip * blocks_per_window + max(0, tail_blocks_to_skip)


def build_object_group_transfer_plan(
    geometry: ObjectGroupGeometry,
    present: Sequence[bool],
    batch_size: int,
    skip_first_n_tokens: int,
    is_h2d: bool,
) -> list[TransferPlanStep]:
    """Plan one object group's transfer as a list of batched copy steps.

    Sliding-window groups retrieve only their trailing ``num_chunks_in_sw``
    chunks, so on H2D the leading objects beyond the window are skipped
    entirely. Batches that fall wholly before ``skip_first_n_tokens`` are
    dropped; the batch straddling it carries per-kernel-group ``skip_blocks``
    so the copy engine leaves the already-populated leading blocks untouched.

    Args:
        geometry: The object group's geometry.
        present: Availability of each memory object, in object order; the
            plan covers ``len(present)`` objects. On D2H a batch containing
            an absent object is skipped (the storage layer declined those
            keys); on H2D an absent object is an error.
        batch_size: Number of memory objects per batched copy.
        skip_first_n_tokens: Tokens to skip writing at the start of the
            transfer range (H2D APC protection; pass 0 for stores).
        is_h2d: True for retrieve (H2D), False for store (D2H).

    Returns:
        The plan steps in copy order; empty when nothing needs to move.

    Raises:
        ValueError: If ``is_h2d`` and any planned batch contains an absent
            object, or if ``batch_size``/``skip_first_n_tokens`` is invalid.
    """
    if skip_first_n_tokens < 0:
        raise ValueError("skip_first_n_tokens must be non-negative")

    num_objects = len(present)
    num_objects_to_skip = 0
    if geometry.num_chunks_in_sw >= 0 and is_h2d:
        num_objects_to_skip = max(0, num_objects - geometry.num_chunks_in_sw)
        if num_objects_to_skip > 0:
            logger.debug(
                "Sliding window: skipping the first %d of %d leading objects "
                "(H2D, window covers %d trailing chunks)",
                num_objects_to_skip,
                num_objects,
                geometry.num_chunks_in_sw,
            )

    tokens_per_chunk = geometry.tokens_per_chunk
    steps: list[TransferPlanStep] = []
    for start_object_idx, index_batch in batched_iteration_with_skip(
        range(num_objects), batch_size, skip_count=num_objects_to_skip
    ):
        if not all(present[i] for i in index_batch):
            if is_h2d:
                raise ValueError(
                    "MemoryObj is None for some objects in the batch, cannot "
                    f"perform H2D copy. object indices: {index_batch}"
                )
            continue

        batch_len = len(index_batch)
        batch_start_token = start_object_idx * tokens_per_chunk
        batch_end_token = batch_start_token + batch_len * tokens_per_chunk

        effective_start = max(batch_start_token, skip_first_n_tokens)
        if effective_start >= batch_end_token:
            continue

        skip_tokens_in_chunk = effective_start - batch_start_token

        launches = tuple(
            KernelGroupLaunch(
                kernel_group_id=kg.kernel_group_id,
                start_block_pos=start_object_idx * kg.blocks_per_window,
                num_blocks=batch_len * kg.blocks_per_window,
                skip_blocks=_recalculate_blocks_to_skip(
                    kg.blocks_per_chunk,
                    kg.blocks_per_window,
                    kg.tokens_to_blocks(skip_tokens_in_chunk),
                ),
            )
            for kg in geometry.kernel_groups
        )
        steps.append(
            TransferPlanStep(
                start_object_idx=start_object_idx,
                object_indices=index_batch,
                launches=launches,
            )
        )
    return steps

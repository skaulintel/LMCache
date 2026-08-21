# SPDX-License-Identifier: Apache-2.0
"""Engine-driven (non-CUDA) transfer with several KV geometries per model.

Intended repo path: ``tests/v1/multiprocess/test_engine_driven_kv_groups.py``.
Runs standalone (``python test_engine_driven_kv_groups.py``) or under pytest;
CPU only, no server / message queue needed.
"""

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
    _chunk_shape,
    _empty_chunk,
    _plan_chunk_groups,
)

HINTS = {"kv_layout": "NHD"}
NUM_BLOCKS, BLOCK_SIZE, BLOCKS_IN_CHUNK = 8, 16, 2
# Two KV geometries in one model (gemma-4 shape: sliding layers with more,
# narrower heads interleaved with full-attention layers), plus one layer in no
# group at all (cross-layer KV sharing, which aliases another layer's blocks).
GEOMETRIES = [(4, 32), (2, 32), (4, 32), (2, 32), (4, 32), (2, 32), (4, 32)]
GROUPS = [
    EngineGroupInfo(engine_group_id=0, layer_indices=(0, 2, 4)),
    EngineGroupInfo(engine_group_id=1, layer_indices=(1, 3, 5)),
]


def make_kv_caches() -> dict[str, torch.Tensor]:
    """Return vLLM-style paged KV tensors, one per layer, all on CPU."""
    return {
        f"layer.{i}": torch.randn(2, NUM_BLOCKS, BLOCK_SIZE, num_heads, head_dim)
        for i, (num_heads, head_dim) in enumerate(GEOMETRIES)
    }


def test_plan_packs_groups_back_to_back() -> None:
    kv_caches = make_kv_caches()
    plans, block_size, dtype_str, num_layers, hidden, kv_size = _plan_chunk_groups(
        kv_caches, BLOCKS_IN_CHUNK, HINTS, GROUPS
    )
    chunk_tokens = BLOCKS_IN_CHUNK * block_size
    assert (block_size, dtype_str) == (BLOCK_SIZE, "float32")
    assert [plan.layer_names for plan in plans] == [
        ("layer.0", "layer.2", "layer.4"),
        ("layer.1", "layer.3", "layer.5"),
    ], "the ungrouped KV-sharing layer must not be packed into the chunk"
    assert [tuple(plan.shape) for plan in plans] == [
        (2, 3, chunk_tokens, 4 * 32),
        (2, 3, chunk_tokens, 2 * 32),
    ]
    # Slices tile the chunk exactly: no gap, no overlap, nothing left over.
    assert plans[0].begin == 0
    assert plans[0].end == plans[1].begin
    assert [plan.end - plan.begin for plan in plans] == [
        plan.shape.numel() for plan in plans
    ]
    registered = _chunk_shape(num_layers, chunk_tokens, hidden, kv_size)
    assert registered.numel() == plans[-1].end
    # Planning is a pure function of the registration: a second worker (or a
    # later retrieve) must carve the identical slices.
    assert _plan_chunk_groups(kv_caches, BLOCKS_IN_CHUNK, HINTS, GROUPS)[0] == plans


def test_single_group_layout_is_unchanged() -> None:
    """One group keeps its natural chunk shape, not the flat packing."""
    kv_caches = {"layer.0": torch.randn(2, NUM_BLOCKS, BLOCK_SIZE, 4, 32)}
    plans, block_size, _, num_layers, hidden, kv_size = _plan_chunk_groups(
        kv_caches, BLOCKS_IN_CHUNK, HINTS, ()
    )
    assert len(plans) == 1
    assert _chunk_shape(
        num_layers, BLOCKS_IN_CHUNK * block_size, hidden, kv_size
    ) == tuple(plans[0].shape)


def _registered_context() -> tuple[
    EngineDrivenTransferContext, torch.Size, torch.dtype
]:
    """Build a context with only the group plan filled in (no MQ, no server)."""
    ctx = EngineDrivenTransferContext()
    plans, block_size, dtype_str, num_layers, hidden, kv_size = _plan_chunk_groups(
        make_kv_caches(), BLOCKS_IN_CHUNK, HINTS, GROUPS
    )
    ctx._group_plans = plans
    ctx._layout_hints = HINTS
    shape = _chunk_shape(num_layers, BLOCKS_IN_CHUNK * block_size, hidden, kv_size)
    return ctx, shape, getattr(torch, dtype_str)


def test_roundtrip_two_geometries() -> None:
    """Gather both groups into one chunk buffer and scatter them back intact."""
    ctx, chunk_shape, dtype = _registered_context()
    kv_caches = make_kv_caches()
    reference = {name: kv.clone() for name, kv in kv_caches.items()}
    # Distinct block IDs per group: each group must read its own list, and the
    # two lists are unrelated for a genuinely hybrid (separate address space)
    # model. Two chunks of BLOCKS_IN_CHUNK blocks each.
    block_ids = [[3, 1, 6, 0], [5, 2, 7, 4]]

    chunks = [_empty_chunk(chunk_shape, dtype) for _ in range(2)]
    gathered = ctx._gather_groups(kv_caches, block_ids, BLOCKS_IN_CHUNK, chunks, None)
    assert gathered is chunks

    for kv in kv_caches.values():
        kv.zero_()
    ctx._scatter_groups(kv_caches, block_ids, BLOCKS_IN_CHUNK, chunks, 0)

    for group_idx, plan in enumerate(ctx._group_plans):
        for name in plan.layer_names:
            for block_id in block_ids[group_idx]:
                assert torch.equal(
                    kv_caches[name][:, block_id], reference[name][:, block_id]
                ), f"{name} block {block_id} did not survive the round trip"
            untouched = set(range(NUM_BLOCKS)) - set(block_ids[group_idx])
            for block_id in untouched:
                assert not kv_caches[name][:, block_id].any(), (
                    f"{name} block {block_id} was written but is not in this "
                    "group's block IDs"
                )
    assert not kv_caches["layer.6"].any(), "ungrouped layer must not be touched"


def test_block_id_group_count_mismatch_names_the_groups() -> None:
    ctx, _, _ = _registered_context()
    try:
        ctx._require_group_plans([[3, 1]])
    except RuntimeError as exc:
        assert "1 block-id list(s) for 2 registered KV group(s)" in str(exc)
        assert "chunk_shape=(2, 3, 32, 128)" in str(exc)
    else:
        raise AssertionError("a wrong per-group block-id count must raise")

    try:
        EngineDrivenTransferContext()._require_group_plans([[3, 1]])
    except RuntimeError as exc:
        assert "not registered" in str(exc)
    else:
        raise AssertionError("using the context before register() must raise")


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_"):
            case()
            print(f"{name}: ok")

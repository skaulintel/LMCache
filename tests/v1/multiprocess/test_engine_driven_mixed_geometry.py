# SPDX-License-Identifier: Apache-2.0
"""Engine-driven multi-group registration when the groups have *different*
KV geometries (e.g. gemma-4: sliding layers 16 x 256, global layers 4 x 512).

``test_engine_driven_multigroup.py`` covers multi-group transfers, but every
case there builds its groups from identically shaped tensors, so nothing
exercises the per-group *shape*. Measured on XPU (2x B70, TP2, gemma-4-31B,
hybrid KV cache manager on), that gap is where the data went missing: the tier
stored 440.00 KiB/token where the model's own geometry says 880.00 -- exactly
half -- and the retrieved KV decoded to garbage. A uniform-geometry hybrid model
(gpt-oss-20b) on the same branch and box is exact.

The cause is the last test below. vLLM equalises page sizes by giving each group
its own block size (gemma-4: 32 tokens for the five sliding groups, 64 for the
global one) and schedules in blocks of the LCM, which is the unit
``blocks_in_chunk`` counts. Registration multiplied it by the block size detected
from the first layer instead, so a 256-token chunk registered as 128 tokens.
"""

# Standard
from typing import Any
from unittest.mock import MagicMock
import math

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context.base import compute_kv_layout
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
)

# Shapes are (kv planes, blocks, block size, heads, head size). Two "sliding"
# layers of one geometry plus one "global" layer of another, with the 2:1 hidden
# ratio gemma-4 has. _WIDE_SHAPE keeps the same block size so the byte-
# conservation test's arithmetic stays layout-independent; _WIDE_PAGED_SHAPE is
# gemma-4's real situation -- half the hidden size, twice the block size, so the
# two groups have equal page sizes and *unequal* block sizes.
_NARROW_SHAPE = (2, 4, 4, 2, 8)
_WIDE_SHAPE = (2, 4, 4, 2, 16)
_WIDE_PAGED_SHAPE = (2, 4, 8, 2, 4)
_BLOCKS_IN_CHUNK = 2


def _hidden_of(shape: tuple[int, ...]) -> int:
    """Per-token object width the registration path derives for one layer.

    Uses the production helper rather than re-deriving it from ``shape`` so the
    test cannot disagree with the code it is checking about which dims are
    heads and which are blocks.
    """
    _, _, hidden_dim_size, _, _, _ = compute_kv_layout({"layer": torch.zeros(shape)})
    return hidden_dim_size


def _block_size_of(shape: tuple[int, ...]) -> int:
    """Tokens one paged block of ``shape`` physically holds."""
    block_size, *_ = compute_kv_layout({"layer": torch.zeros(shape)})
    return block_size


def _register_mixed_geometry(
    wide_shape: tuple[int, ...] = _WIDE_SHAPE,
    blocks_in_chunk: int = _BLOCKS_IN_CHUNK,
    tokens_per_block: tuple[int, int] = (0, 0),
) -> tuple[RegisterEngineDrivenContextPayload, dict[str, torch.Tensor]]:
    """Register two groups with different geometries; return payload + caches.

    Args:
        wide_shape: shape of the second group's single layer.
        blocks_in_chunk: scheduling blocks per LMCache chunk, exactly as the
            connector computes it (``chunk tokens // engine block size``).
        tokens_per_block: ``EngineGroupInfo.tokens_per_block`` per group -- the
            *logical* tokens one of that group's block IDs covers, as vLLM
            declares it in ``kv_cache_spec.block_size``. ``0`` means "not
            reported", so the block size detected from the tensors is used.
    """
    # First Party
    from lmcache.v1.multiprocess.protocols.engine import (
        RegisterEngineDrivenContextResponse,
    )

    kv_caches = {
        "layer_0": torch.zeros(_NARROW_SHAPE),
        "layer_1": torch.zeros(_NARROW_SHAPE),
        "layer_2": torch.zeros(wide_shape),
    }
    sent: list[Any] = []

    def _send(_mq: Any, _rt: Any, args: list[Any]) -> Any:
        sent.append(args[0])
        future = MagicMock()
        future.result.return_value = RegisterEngineDrivenContextResponse(
            shm_name="lmcache_l1_pool_mixed", pool_size=4096
        )
        return future

    EngineDrivenTransferContext().register(
        instance_id=1,
        kv_caches=kv_caches,
        model_name="mixed-geometry",
        world_size=1,
        blocks_in_chunk=blocks_in_chunk,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=_send,
        engine_group_infos=[
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=(0, 1),
                tokens_per_block=tokens_per_block[0],
            ),
            EngineGroupInfo(
                engine_group_id=1,
                layer_indices=(2,),
                tokens_per_block=tokens_per_block[1],
            ),
        ],
    )
    return sent[0], kv_caches


def test_mixed_geometry_groups_register_their_own_hidden_size() -> None:
    """Each group's layout must carry that group's width, not group 0's."""
    payload, _ = _register_mixed_geometry()
    narrow, wide = _hidden_of(_NARROW_SHAPE), _hidden_of(_WIDE_SHAPE)
    assert wide == 2 * narrow, "test fixture no longer has two distinct geometries"
    assert [gl.hidden_dim_size for gl in payload.group_layouts] == [narrow, wide]
    assert [gl.num_layers for gl in payload.group_layouts] == [2, 1]


def test_mixed_geometry_registration_conserves_kv_bytes() -> None:
    """The per-group objects must add up to the KV the chunk really holds.

    This is the invariant that a factor-of-two loss cannot survive, and it is
    model-independent: with no group reporting a reduced window, the bytes the
    server will reserve for one chunk across all groups have to equal the bytes
    those same layers hold for that chunk. Sliding-window groups are the only
    licensed shortfall, and there are none here.
    """
    payload, kv_caches = _register_mixed_geometry()
    block_size, _, _, _, _, kv_size = compute_kv_layout(kv_caches)
    chunk_tokens = _BLOCKS_IN_CHUNK * block_size
    itemsize = kv_caches["layer_0"].element_size()

    registered = sum(
        kv_size * gl.num_layers * gl.hidden_dim_size * gl.window_tokens * itemsize
        for gl in payload.group_layouts
    )
    actual = sum(
        kv_size * _hidden_of(tuple(t.shape)) * chunk_tokens * itemsize
        for t in kv_caches.values()
    )
    assert all(gl.window_tokens == chunk_tokens for gl in payload.group_layouts), (
        "no group has a sliding window, so none may store a reduced token span"
    )
    assert registered == actual, (
        f"per-group objects hold {registered} B per chunk but the layers hold "
        f"{actual} B ({registered / actual:.3f}x)"
    )


def test_groups_with_unequal_block_sizes_register_the_whole_chunk() -> None:
    """Mixed block sizes must not shrink the chunk (the gemma-4 halving).

    ``blocks_in_chunk`` counts the engine's scheduling blocks, and the engine
    schedules in units of the LCM of its groups' block sizes -- 8 here, as
    gemma-4 schedules in 64 with groups of 32 and 64. So a chunk is
    ``blocks_in_chunk * lcm`` tokens, and every full-attention group must
    register exactly that many. Deriving it from the block size of whichever
    layer comes first instead registers a fraction of the chunk: the rest of
    each chunk's KV is never stored, and what is retrieved is garbage.
    """
    narrow_block = _block_size_of(_NARROW_SHAPE)
    wide_block = _block_size_of(_WIDE_PAGED_SHAPE)
    assert wide_block != narrow_block, "fixture must have two different block sizes"
    # 2, not 1: the shrunken chunk must still be divisible by every group's block
    # size, or the existing divisibility guard raises and the loss is loud. It was
    # silent for gemma-4 (128 tokens, divisible by both 32 and 64), and this is
    # the smallest fixture that reproduces that.
    blocks_in_chunk = 2
    expected_chunk = blocks_in_chunk * math.lcm(narrow_block, wide_block)

    payload, _ = _register_mixed_geometry(
        wide_shape=_WIDE_PAGED_SHAPE,
        blocks_in_chunk=blocks_in_chunk,
        tokens_per_block=(narrow_block, wide_block),
    )

    assert [gl.window_tokens for gl in payload.group_layouts] == [
        expected_chunk,
        expected_chunk,
    ], (
        f"groups registered {[gl.window_tokens for gl in payload.group_layouts]} "
        f"tokens per chunk, but the chunk is {expected_chunk} tokens "
        f"({blocks_in_chunk} scheduling block(s) of "
        f"lcm({narrow_block}, {wide_block}))"
    )

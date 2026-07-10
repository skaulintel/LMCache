# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``build_object_group_transfer_plan``."""

# Standard
from typing import Callable

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.transfer_plan import (
    KernelGroupGeometry,
    ObjectGroupGeometry,
    build_object_group_transfer_plan,
)


def _linear_tokens_to_blocks(
    blocks_per_chunk: int, tokens_per_chunk: int
) -> Callable[[int], int]:
    """Return a linear token->block map for one kernel group."""
    return lambda tokens: tokens * blocks_per_chunk // tokens_per_chunk


def _geometry(
    kernel_groups: list[tuple[int, int, int]],
    tokens_per_chunk: int = 256,
    num_chunks_in_sw: int = -1,
) -> ObjectGroupGeometry:
    """Geometry with linear token->block math (block size = tpc / bpc)."""
    return ObjectGroupGeometry(
        object_group_id=0,
        kernel_groups=tuple(
            KernelGroupGeometry(
                kernel_group_id=kg_id,
                blocks_per_chunk=bpc,
                blocks_per_window=bpw,
                tokens_to_blocks=_linear_tokens_to_blocks(bpc, tokens_per_chunk),
            )
            for kg_id, bpc, bpw in kernel_groups
        ),
        tokens_per_chunk=tokens_per_chunk,
        num_chunks_in_sw=num_chunks_in_sw,
    )


def test_full_attention_batches_cover_all_objects():
    plan = build_object_group_transfer_plan(
        _geometry([(0, 8, 8)]),
        present=[True] * 5,
        batch_size=2,
        skip_first_n_tokens=0,
        is_h2d=True,
    )
    assert [s.object_indices for s in plan] == [(0, 1), (2, 3), (4,)]
    assert [s.start_object_idx for s in plan] == [0, 2, 4]
    launches = [s.launches[0] for s in plan]
    assert [(x.start_block_pos, x.num_blocks) for x in launches] == [
        (0, 16),
        (16, 16),
        (32, 8),
    ]
    assert all(x.skip_blocks == 0 for x in launches)


def test_sliding_window_skips_leading_objects_on_h2d():
    plan = build_object_group_transfer_plan(
        _geometry([(0, 8, 8)], num_chunks_in_sw=2),
        present=[True] * 5,
        batch_size=1,
        skip_first_n_tokens=0,
        is_h2d=True,
    )
    assert [s.object_indices for s in plan] == [(3,), (4,)]
    assert [s.launches[0].start_block_pos for s in plan] == [24, 32]


def test_sliding_window_does_not_skip_on_d2h():
    plan = build_object_group_transfer_plan(
        _geometry([(0, 8, 8)], num_chunks_in_sw=2),
        present=[True] * 5,
        batch_size=5,
        skip_first_n_tokens=0,
        is_h2d=False,
    )
    assert [s.object_indices for s in plan] == [(0, 1, 2, 3, 4)]


def test_skip_first_n_tokens_drops_and_clamps_batches():
    # tokens_per_chunk=256, batch_size=1: skip 300 tokens drops chunk 0
    # entirely and clamps 44 tokens (= 1 block of 8-per-chunk math floored)
    # inside chunk 1.
    plan = build_object_group_transfer_plan(
        _geometry([(0, 8, 8)]),
        present=[True] * 3,
        batch_size=1,
        skip_first_n_tokens=300,
        is_h2d=True,
    )
    assert [s.start_object_idx for s in plan] == [1, 2]
    assert plan[0].launches[0].skip_blocks == 44 * 8 // 256
    assert plan[1].launches[0].skip_blocks == 0


def test_window_narrower_than_chunk_recalculates_skip_blocks():
    # blocks_per_window=4 < blocks_per_chunk=8: a 6-block raw skip lands
    # 2 blocks into the retained window (6 - (8 - 4)).
    plan = build_object_group_transfer_plan(
        _geometry([(0, 8, 4)]),
        present=[True] * 2,
        batch_size=1,
        skip_first_n_tokens=192,  # 6 blocks of 32 tokens
        is_h2d=True,
    )
    assert plan[0].start_object_idx == 0
    assert plan[0].launches[0].skip_blocks == 2
    assert plan[0].launches[0].num_blocks == 4


def test_absent_object_skips_batch_on_d2h():
    plan = build_object_group_transfer_plan(
        _geometry([(0, 8, 8)]),
        present=[True, False, True, True],
        batch_size=2,
        skip_first_n_tokens=0,
        is_h2d=False,
    )
    assert [s.object_indices for s in plan] == [(2, 3)]


def test_absent_object_raises_on_h2d():
    with pytest.raises(ValueError, match="cannot perform H2D copy"):
        build_object_group_transfer_plan(
            _geometry([(0, 8, 8)]),
            present=[True, None or False],
            batch_size=2,
            skip_first_n_tokens=0,
            is_h2d=True,
        )


def test_multiple_kernel_groups_launch_per_group_geometry():
    plan = build_object_group_transfer_plan(
        _geometry([(0, 8, 8), (1, 16, 16)]),
        present=[True] * 2,
        batch_size=2,
        skip_first_n_tokens=0,
        is_h2d=True,
    )
    (step,) = plan
    assert [(x.kernel_group_id, x.num_blocks) for x in step.launches] == [
        (0, 16),
        (1, 32),
    ]


def test_empty_plan_when_everything_skipped():
    assert (
        build_object_group_transfer_plan(
            _geometry([(0, 8, 8)]),
            present=[True] * 2,
            batch_size=2,
            skip_first_n_tokens=512,
            is_h2d=True,
        )
        == []
    )
    assert (
        build_object_group_transfer_plan(
            _geometry([(0, 8, 8)]),
            present=[],
            batch_size=2,
            skip_first_n_tokens=0,
            is_h2d=True,
        )
        == []
    )

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Document-level routing expectations independent of tensor execution."""

import pytest

from relax.models.qwen3_8.ops.qsa_mixed_ops.policy import flash_max_length, plan_documents


@pytest.mark.parametrize(
    "flex_limit,expected",
    [
        (16384, (((0, 2051),), ((2051, 4103), (4103, 20487)), ((20487, 36872),))),
        (0, (((0, 2051),), (), ((2051, 4103), (4103, 20487), (20487, 36872)))),
    ],
)
def test_production_boundaries(flex_limit, expected):
    # Documents: 2051, 2052, 16384, 16385. NV disables the middle category.
    assert plan_documents((0, 2051, 4103, 20487, 36872), flash_max_length(2048, -1), flex_limit) == expected


@pytest.mark.parametrize(
    "flex_limit,expected",
    [
        (16384, (((0, 1000),), ((1000, 9000),), ((9000, 29000),))),
        (0, (((0, 1000),), (), ((1000, 9000), (9000, 29000)))),
    ],
)
def test_mixed_lengths_and_empty_documents(flex_limit, expected):
    assert plan_documents((0, 0, 1000, 9000, 9000, 29000, 29000), 2051, flex_limit) == expected


def test_custom_budget_and_flash_priority():
    # Eight selected four-token blocks cover at most 35 tokens densely.
    assert flash_max_length(8 * 4, -1) == 35
    assert plan_documents((0, 35, 71, 111), 35, 39) == (((0, 35),), ((35, 71),), ((71, 111),))
    # A Flex threshold below Flash does not override dense short documents.
    assert plan_documents((0, 35, 71), 35, 20) == (((0, 35),), (), ((35, 71),))


@pytest.mark.parametrize("bounds", [(0, 0), (0, 0, 0)])
def test_empty_plan(bounds):
    assert plan_documents(bounds, 35, 16384) == ((), (), ())


@pytest.mark.parametrize("configured", [-2, 36])
def test_flash_limit_rejects_invalid_custom_budget(configured):
    with pytest.raises(ValueError, match="dense selector budget"):
        flash_max_length(32, configured)

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only document routing shared by the mixed QSA adapters."""

from typing import NamedTuple


class DocumentPlan(NamedTuple):
    flash: tuple[tuple[int, int], ...]
    flex: tuple[tuple[int, int], ...]
    long: tuple[tuple[int, int], ...]


def flash_max_length(token_budget: int, configured: int) -> int:
    """Resolve the dense prefix, including up to three causal tail tokens."""
    maximum = token_budget + 3
    limit = maximum if configured == -1 else configured
    if not 0 <= limit <= maximum:
        raise ValueError("Mixed QSA Flash maximum length exceeds the dense selector budget")
    return limit


def plan_documents(bounds: tuple[int, ...] | list[int], flash_max_len: int, flex_max_len: int = 0) -> DocumentPlan:
    """Group nonempty document spans using resolved, adapter-validated limits.

    The long group belongs to the selected adapter (Grouped or NV). Preserve
    original offsets and document order; tensor preparation stays in adapters.
    """
    flash, flex, long = [], [], []
    for start, end in zip(bounds, bounds[1:]):
        length = end - start
        if length == 0:
            continue
        if length <= flash_max_len:
            flash.append((start, end))
        elif length <= flex_max_len:
            flex.append((start, end))
        else:
            long.append((start, end))
    return DocumentPlan(tuple(flash), tuple(flex), tuple(long))

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Request-owned seed witnesses; hints never replace current-query scores."""

from dataclasses import dataclass

import torch


@dataclass
class DoSASeedState:
    """One document/layer's last completed query block, never global state.

    Create a new instance for each document/request or key remapping. Results
    stay in original document-local IDs. Native carry uses the last1536 rows.
    """

    selected: torch.Tensor | None = None
    visible_end: torch.Tensor | None = None
    key_count: int = 0

    def clear(self) -> None:
        self.selected = None
        self.visible_end = None
        self.key_count = 0

    @torch.no_grad()
    def publish(
        self,
        selected: torch.Tensor,
        offsets: torch.Tensor,
        key_count: int,
        valid: torch.Tensor | None = None,
    ) -> None:
        ids = selected[-1536:].detach()
        if valid is not None:
            ids = ids.masked_fill(~valid[-1536:], -1)
        self.selected = ids.to(torch.int32).clone().contiguous()
        self.visible_end = (offsets.amax().clamp(min=-1, max=key_count - 1) + 1).detach()
        self.key_count = key_count


@torch.no_grad()
def informative_order(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    valid: torch.Tensor,
    valid_ids: torch.Tensor,
    visible: torch.Tensor,
    seed_size: int,
    state: DoSASeedState | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vote hot witnesses into each group's common visible seed prefix.

    Cold representatives match native floor(a*(Q-1)/3), a=0..3. Compatible
    history replaces these votes, never accumulates with them. All seed scores
    are subsequently recomputed using each current query's original scorer.
    """
    import triton

    from . import dosa_parallel as parallel
    from .topk import dsa_topk_indices

    rows, width = len(query), len(key)
    common_end = (offsets.amin().clamp(min=-1, max=width - 1) + 1).contiguous()
    history = state.selected if state is not None else None
    has_history = (
        history is not None
        and history.device == query.device
        and history.dtype == torch.int32
        and state.visible_end is not None
        and state.visible_end.device == query.device
        and 0 < state.key_count <= width
        and history.ndim == 2
        and history.shape[1] <= 2048
    )
    history_end = state.visible_end if has_history else common_end
    source = (
        ((history_end <= common_end) & (history_end >= 0)).int()
        if has_history
        else torch.zeros((), device=query.device, dtype=torch.int32)
    )
    scores = query.new_empty((4, width))
    launch = {"num_warps": 4, "enable_fp_fusion": False}
    parallel.bootstrap_scores[(4, triton.cdiv(width, 128))](
        query, key, weights, valid, common_end, source, scores, rows, width, **launch
    )
    selected = dsa_topk_indices(scores).contiguous()
    votes = torch.zeros((width + 1,), device=query.device, dtype=torch.int32)
    history_count = history.numel() if has_history else 0
    parallel.seed_votes[(triton.cdiv(max(8192, history_count), 256),)](
        selected,
        history if has_history else selected,
        common_end,
        source,
        valid,
        votes,
        width,
        history_count,
        **launch,
    )
    # Positive votes are restricted to the call's common visible prefix. The
    # zero-vote fill uses the earliest valid IDs. Thus every HOT ID is visible
    # to every row with visible>S, even if short rows share its screening group.
    # Sort this shared HOT set once, rather than repeating it for every group.
    groups = triton.cdiv(rows, 4)
    priority = votes[valid_ids.long()]
    ranking = priority.argsort(dim=-1, descending=True, stable=True)
    hot = torch.zeros((width,), device=query.device, dtype=torch.int32)
    hot.scatter_(0, ranking[:seed_size], 1)
    permutation = hot.argsort(dim=-1, descending=True, stable=True)
    ordered = valid_ids[permutation][None].expand(groups, -1).contiguous()
    return ordered, source, votes

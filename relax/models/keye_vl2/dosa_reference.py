# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Slow FP64 interval oracle for the Keye DoSA certificate, not a backend.

Uses a nonnegative single-anchor projection and the triangle residual bound.
The negative aggregate may use any subset; this oracle uses all omitted
negative heads. Sampling their activation is a tightness heuristic only.
"""

import torch


def _up(x: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(x, torch.full_like(x, float("inf")))


def _down(x: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(x, torch.full_like(x, -float("inf")))


def _sum(x: torch.Tensor, dim: int, *, upper: bool) -> torch.Tensor:
    parts = x.unbind(dim)
    result = torch.zeros_like(parts[0])
    rounding = _up if upper else _down
    for part in parts:
        result = rounding(result + part)
    return result


def reference_bound(query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, heads: int = 4) -> torch.Tensor:
    """Upper enclosure of staged BF16 scores, independent of Triton code.

    Inputs are one query [16,64], document keys [N,1,64], weights [16]. This
    routine is deliberately expensive and intended only for tests.
    """
    q, k, w = query.double(), key.squeeze(1).double(), weights.double()
    norm = q.norm(dim=-1)
    positive = w > 0
    priority = w.abs() * norm
    order = sorted(range(16), key=lambda h: (-int(positive[h]), -float(priority[h]), h))
    retained = order[:heads]
    omitted = order[heads:]
    anchor = q[retained[0]]
    beta = ((q * anchor).sum(-1) / anchor.square().sum().clamp_min(2.0**-64)).clamp(0, 65536)
    beta = torch.where(positive, beta, 0)
    beta[retained] = 0
    residual = torch.maximum((q - _down(beta[:, None] * anchor)).abs(), (q - _up(beta[:, None] * anchor)).abs())
    residual = _up(residual)
    omitted_positive = positive.clone()
    omitted_positive[retained] = False
    allowance = _sum(torch.where(omitted_positive[:, None], _up(w[:, None] * residual), 0), 0, upper=True)
    numerical = _sum(_up(w[:, None].abs() * q.abs()), 0, upper=True)
    allowance = _up(allowance + _up(numerical / 64))
    fold_lo = _sum(_down(w * beta), 0, upper=False)
    fold_hi = _sum(_up(w * beta), 0, upper=True)
    upper = torch.zeros(k.shape[0], dtype=torch.float64, device=k.device)
    for j, h in enumerate(retained):
        lo = _sum(_down(k * q[h]), -1, upper=False).clamp_min(0)
        hi = _sum(_up(k * q[h]), -1, upper=True).clamp_min(0)
        wl, wh = (_down(w[h] + fold_lo), _up(w[h] + fold_hi)) if j == 0 else (w[h], w[h])
        contribution = torch.stack((_up(wl * lo), _up(wl * hi), _up(wh * lo), _up(wh * hi))).amax(0)
        upper = _up(upper + contribution)
    negative = torch.zeros_like(w)
    negative[omitted] = (-w[omitted]).clamp_min(0)
    al = _sum(_down(negative[:, None] * q), 0, upper=False)
    ah = _sum(_up(negative[:, None] * q), 0, upper=True)
    neg_lower = _sum(torch.minimum(_down(k * al), _down(k * ah)), -1, upper=False).clamp_min(0)
    upper = _up(_up(upper + _sum(_up(k.abs() * allowance), -1, upper=True)) - neg_lower)
    upper = _up(upper / 8)

    def safe(x: torch.Tensor) -> torch.Tensor:
        return (x == 0) | ((x.abs() >= 2.0**-32) & (x.abs() <= 2.0**16))

    safe_row = safe(q).all() & safe(w).all()
    safe_keys = safe(k).all(-1)
    return upper.masked_fill(~safe_keys | ~safe_row | ~upper.isfinite(), float("inf"))


@torch.no_grad()
def reference_select(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    heads: int = 4,
) -> torch.Tensor:
    """Seed, certify, refine using the original scorer; original IDs break
    ties."""
    from .indexer import indexer_scores

    ids = torch.arange(len(key), device=key.device)
    valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
    result = torch.full((len(query), min(2048, len(key))), -1, device=query.device, dtype=torch.long)
    for row in range(len(query)):
        visible = ids[valid & (ids <= offsets[row])]
        if len(visible) <= 2048:
            result[row, : len(visible)] = visible
            continue
        seed = visible[
            torch.arange(min(4096, len(visible)), device=ids.device) * len(visible) // min(4096, len(visible))
        ]
        scores = indexer_scores(query[row : row + 1], key[seed], weights[row : row + 1])[0]
        tau = scores.sort(descending=True).values[2047]
        bounds = reference_bound(query[row], key[visible], weights[row], heads)
        candidates = visible[~(bounds < tau)]
        candidates = torch.cat((seed, candidates)).unique(sorted=True)
        scores = indexer_scores(query[row : row + 1], key[candidates], weights[row : row + 1])[0]
        chosen = scores.argsort(descending=True, stable=True)[:2048]
        result[row] = candidates[chosen]
    return result

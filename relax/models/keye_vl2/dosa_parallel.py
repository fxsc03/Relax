# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Parallel, capacity-safe DoSA screening and cross-tile refinement.

The screening dot products are BF16 tensor-core operations. Their FP32
accumulation error and the aggregate's BF16 conversion error are included in a
directed-rounding L2 certificate; refinement still uses score_tile.
"""

from typing import TYPE_CHECKING

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

from .dosa_kernels import _code, _safe, _up
from .indexer_scores import score_tile


if TYPE_CHECKING:
    import torch


@triton.jit
def _norm(x):
    return lib.sqrt_ru(tl.reduce(lib.mul_ru(x, x), 0, _up))


@triton.jit
def pack_plan(PLAN, VISIBLE, VECTORS, COEFF, ERROR, R: tl.constexpr, S: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, 64)
    if tl.load(VISIBLE + row) > S:
        base = PLAN + row * (R * 66 + 193)
        al = tl.load(base + R * 66 + d)
        ah = tl.load(base + R * 66 + 64 + d)
        # Any center is valid: certify its actual BF16 representation.
        a = ((al + ah) * 0.5).to(tl.bfloat16).to(tl.float32)
        delta = tl.maximum(tl.abs(lib.add_rd(al, -a)), tl.abs(lib.add_ru(ah, -a)))
        envelope = lib.add_ru(tl.load(base + R * 66 + 128 + d), delta)
        scalar = tl.load(base + R * 66 + 192)
        allowance = lib.add_ru(scalar, _norm(envelope))
        dot_error = _norm(a)
        for j in tl.static_range(R):
            q = tl.load(base + j * 64 + d)
            wl = tl.load(base + R * 64 + j)
            wh = tl.load(base + R * 65 + j)
            tl.store(VECTORS + (row * (R + 1) + j) * 64 + d, q)
            tl.store(COEFF + row * R + j, wh)
            dot_error = lib.add_ru(dot_error, lib.mul_ru(tl.maximum(tl.abs(wl), tl.abs(wh)), _norm(q)))
        # gamma_128 using a full FP32 ulp is < 2^-15. This covers
        # 64-product FP32 accumulation, including tensor-core grouping.
        allowance = lib.add_ru(allowance, lib.mul_ru(dot_error, 2.0**-15))
        ok = (scalar < float("inf")) & (tl.sum((~_safe(a)).to(tl.int32), 0) == 0)
        tl.store(VECTORS + (row * (R + 1) + R) * 64 + d, a)
        tl.store(ERROR + row, tl.where(ok & (allowance < float("inf")), allowance, float("inf")))


@triton.jit
def key_norm(K, NORMS, N: tl.constexpr):
    t = tl.program_id(0) * 128 + tl.arange(0, 128)
    d = tl.arange(0, 64)
    k = tl.load(K + t[:, None] * 64 + d[None, :], t[:, None] < N, 0).to(tl.float32)
    norm = lib.sqrt_ru(tl.reduce(lib.mul_ru(k, k), 1, _up))
    ok = tl.sum((~_safe(k)).to(tl.int32), 1) == 0
    tl.store(NORMS + t, tl.where(ok, norm, float("inf")), t < N)


@triton.jit
def seed_codes(
    SCORES,
    SEEDS,
    TOP,
    VISIBLE,
    VALID,
    CODES,
    TAU,
    S: tl.constexpr,
    CAP: tl.constexpr,
    WORDS=None,
    LOW=None,
    LENGTHS=None,
    PACKED: tl.constexpr = False,
):
    row = tl.program_id(0)
    t = tl.arange(0, 2048)
    count = tl.load(VISIBLE + row)
    if count <= 2048:
        ids = tl.load(VALID + t, t < count, -1)
        scores = tl.full((2048,), 0, tl.bfloat16)
    else:
        selected = tl.load(TOP + row * 2048 + t)
        ids = tl.load(SEEDS + row * S + selected)
        scores = tl.load(SCORES + row * S + selected)
    if PACKED:
        tl.store(LENGTHS + row, tl.minimum(tl.maximum(count, 0), 2048))
        code = _code(scores, ids, ids >= 0).to(tl.uint64)
        score = (code >> 32).to(tl.uint32)
        original_id = (0xFFFFFFFF - (code & 0xFFFFFFFF)).to(tl.uint32) & 0xFFFFF
        tl.store(WORDS + row * (CAP + 2048) + t, ((score >> 8) << 20) | original_id)
        tl.store(LOW + row * (CAP + 2048) + t, score & 255)
    else:
        tl.store(CODES + row * (CAP + 2048) + t, _code(scores, ids, ids >= 0))
    tl.store(TAU + row, tl.min(scores.to(tl.float32), 0))


@triton.jit
def complement(t, base, S: tl.constexpr, WIDE: tl.constexpr = True):
    missing = base - S
    # Long prefixes overflow i32 in the inverse seed-map numerator.
    # N<=32768 implies the numerator is below 2^31; keep that hot path i32.
    ordinal = t.to(tl.int64) if WIDE else t
    return tl.where(
        t < missing,
        ((ordinal + 1) * base + tl.maximum(missing, 1) - 1) // tl.maximum(missing, 1) - 1,
        ordinal + S,
    )


@triton.jit
def screen(
    K,
    NORMS,
    VECTORS,
    COEFF,
    ERROR,
    TAU,
    BASE,
    VALID,
    VISIBLE,
    FLAGS,
    COUNTS,
    UNSAFE,
    Q: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
    CAP: tl.constexpr,
    TILES: tl.constexpr,
    M: tl.constexpr,
    BOUNDS=None,
    DUMP: tl.constexpr = False,
    GROUPED: tl.constexpr = False,
):
    group, block = tl.program_id(0), tl.program_id(1)
    t = block * 128 + tl.arange(0, 128)
    # Informed seeds form a contiguous prefix of the permuted ID array.
    # Only uniform seeds need the inverse sampling map and wide division.
    rank = t + S if GROUPED else complement(t, tl.load(BASE + group), S, WIDE=N > 32768)
    ids = tl.load(VALID + (group * N if GROUPED else 0) + rank, (t < CAP) & (rank < N), -1)
    d, m = tl.arange(0, 64), tl.arange(0, M)
    L: tl.constexpr = M // 4
    row = group * 4 + m // L
    head = m % L
    vec = tl.load(
        VECTORS + (row[:, None] * (R + 1) + head[:, None]) * 64 + d[None, :],
        (head[:, None] <= R) & (row[:, None] < Q),
        0,
    )
    keys = tl.load(K + ids[None, :] * 64 + d[:, None], (ids[None, :] >= 0) & (ids[None, :] < N), 0)
    dots = tl.dot(vec, keys, input_precision="ieee")
    weight = tl.load(COEFF + row * R + head, (row < Q) & (head < R), 0)
    weight = tl.where(head == R, -1.0, weight)
    contributions = lib.mul_ru(weight[:, None], tl.maximum(dots, 0))
    # Reduce the actual head dimension once. Repeated row extraction from
    # an MMA accumulator introduces expensive layout shuffles/shared traffic.
    upper = tl.reduce(contributions.reshape((4, L, 128)), 1, _up)
    norm = tl.load(NORMS + ids, (ids >= 0) & (ids < N), float("inf"))
    qr = group * 4 + tl.arange(0, 4)
    count = tl.load(VISIBLE + qr, qr < Q, 0)
    error = tl.load(ERROR + qr, qr < Q, float("inf"))
    upper = lib.mul_ru(lib.add_ru(upper, lib.mul_ru(error[:, None], norm[None, :])), 0.125)
    unsafe = ~(tl.abs(upper) < float("inf")) | ~(norm[None, :] < float("inf"))
    upper = tl.where(unsafe, float("inf"), upper)
    if DUMP:
        tl.store(BOUNDS + qr[:, None] * CAP + t[None, :], upper, (qr[:, None] < Q) & (t[None, :] < CAP))
    threshold = tl.load(TAU + qr, qr < Q, float("inf"))
    valid = (t[None, :] < CAP) & (rank[None, :] < count[:, None]) & (count[:, None] > S) & (qr[:, None] < Q)
    live = valid & ~(upper < threshold[:, None])
    tl.store(FLAGS + qr[:, None] * CAP + t[None, :], live, (qr[:, None] < Q) & (t[None, :] < CAP))
    tl.store(COUNTS + qr * TILES + block, tl.sum(live.to(tl.int32), 1), qr < Q)
    tl.store(UNSAFE + qr * TILES + block, tl.sum((unsafe & valid).to(tl.int32), 1), qr < Q)


@triton.jit
def compact(
    FLAGS,
    PREFIX,
    BASE,
    VALID,
    IDS,
    CAP: tl.constexpr,
    TILES: tl.constexpr,
    S: tl.constexpr,
    GROUPED: tl.constexpr = False,
):
    row, block = tl.program_id(0), tl.program_id(1)
    t = block * 128 + tl.arange(0, 128)
    live = tl.load(FLAGS + row * CAP + t, t < CAP, False)
    start = tl.load(PREFIX + row * TILES + block - 1, block > 0, 0)
    dest = start + tl.cumsum(live.to(tl.int32)) - 1
    rank = t + S if GROUPED else complement(t, tl.load(BASE + row // 4), S, WIDE=CAP + S > 32768)
    ids = tl.load(VALID + (row // 4 * (CAP + S) if GROUPED else 0) + rank, live, 0)
    tl.store(IDS + row * CAP + dest, ids, live)


@triton.jit
def refine(Q, K, W, IDS, PREFIX, TAU, WORDS, LOW, LENGTHS, CAP: tl.constexpr, TILES: tl.constexpr):
    """Fuse original scoring, strict-threshold filtering and five-byte
    writeback.

    Every live tile reserves only its kept count. Seed initialization resets
    the row counter on every call/replay; capacity covers all non-seeds plus
    seed K.
    """
    row, block = tl.program_id(0), tl.program_id(1)
    count = tl.load(PREFIX + row * TILES + TILES - 1)
    t = block * 128 + tl.arange(0, 128)
    if block * 128 < count:
        ids = tl.load(IDS + row * CAP + t, t < count, 0)
        h, d = tl.arange(0, 16), tl.arange(0, 64)
        q = tl.load(Q + row * 1024 + h[:, None] * 64 + d[None, :])
        w = tl.load(W + row * 16 + h)
        keys = tl.load(K + ids[None, :] * 64 + d[:, None], t[None, :] < count, 0)
        scores, _ = score_tile(q, keys, w)
        # Drop only strictly lower scores: ties and unordered values survive.
        threshold = tl.load(TAU + row)
        keep = (t < count) & ~(scores.to(tl.float32) < threshold)
        retained = tl.sum(keep.to(tl.int32), 0)
        if retained > 0:
            begin = tl.atomic_add(LENGTHS + row, retained, sem="relaxed")
            destination = begin + tl.cumsum(keep.to(tl.int32)) - 1
            code = _code(scores, ids, keep).to(tl.uint64)
            score = (code >> 32).to(tl.uint32)
            word = ((score >> 8) << 20) | ids.to(tl.uint32)
            tl.store(WORDS + row * (CAP + 2048) + destination, word, keep)
            tl.store(LOW + row * (CAP + 2048) + destination, score & 255, keep)


@triton.jit
def decode(CODES, OUT):
    t = tl.program_id(0) * 2048 + tl.arange(0, 2048)
    codes = tl.load(CODES + t).to(tl.uint64)
    tl.store(OUT + t, tl.where(codes != 0, (0xFFFFFFFF - (codes & 0xFFFFFFFF)).to(tl.int32), -1))


@triton.jit
def make_seeds(
    VALID,
    VISIBLE,
    SEEDS,
    BASE,
    N: tl.constexpr,
    Q: tl.constexpr,
    S: tl.constexpr,
    ORIGINAL=None,
    GROUPED: tl.constexpr = False,
):
    row, block = tl.program_id(0), tl.program_id(1)
    count = tl.load(VISIBLE + row)
    group = row // 4
    base = N
    for i in tl.static_range(4):
        visible = tl.load(VISIBLE + group * 4 + i, group * 4 + i < Q, 0)
        base = tl.minimum(base, tl.where(visible > S, visible, N))
    j = block * 128 + tl.arange(0, 128)
    if GROUPED:
        base = S
        rank = j
        pointer = tl.where(count > S, VALID + group * N + rank, ORIGINAL + rank)
        ids = tl.load(pointer, rank < count, -1)
    else:
        rank = tl.where(count > S, j * base // S, j)
        ids = tl.load(VALID + rank, rank < count, -1)
    tl.store(SEEDS + row * S + j, ids)
    if (row % 4 == 0) & (block == 0):
        tl.store(BASE + group, base)


@triton.jit
def radix_select(CODES, OUT, VISIBLE, N: tl.constexpr, B: tl.constexpr, HAS_VISIBLE: tl.constexpr):
    """Exact 48-bit score/ID cutoff; no repeated sorting/merging of K
    arrays."""
    row = tl.program_id(0)
    count = tl.load(VISIBLE + row) if HAS_VISIBLE else 2049
    if count <= 2048:
        direct = tl.arange(0, 2048)
        tl.store(OUT + row * 2048 + direct, tl.load(CODES + row * N + direct))
    else:
        t = tl.arange(0, B)
        bins = tl.arange(0, 256)
        prefix = tl.full((), 0, tl.uint64)
        mask = tl.full((), 0, tl.uint64)
        remaining = 2048
        for byte in tl.static_range(5, -1, -1):
            histogram = tl.full((256,), 0, tl.int32)
            for start in range(0, N, B):
                code = tl.load(CODES + row * N + start + t, start + t < N, 0).to(tl.uint64)
                bucket = tl.where((code & mask) == prefix, (code >> (8 * byte)) & 255, 256).to(tl.int32)
                histogram += tl.histogram(bucket, 256)
            above = tl.sum(histogram, 0) - tl.cumsum(histogram)
            boundary = (above < remaining) & (above + histogram >= remaining)
            chosen = tl.sum(tl.where(boundary, bins, 0), 0)
            remaining -= tl.sum(tl.where(boundary, above, 0), 0)
            prefix |= chosen.to(tl.uint64) << (8 * byte)
            mask |= tl.full((), 255, tl.uint64) << (8 * byte)
        written = 0
        for start in range(0, N, B):
            code = tl.load(CODES + row * N + start + t, start + t < N, 0).to(tl.uint64)
            live = (code >= prefix) & (code != 0) & (start + t < N)
            destination = written + tl.cumsum(live.to(tl.int32)) - 1
            tl.store(OUT + row * 2048 + destination, code, live & (destination < 2048))
            written += tl.sum(live.to(tl.int32), 0)


def select_codes(codes: "torch.Tensor", visible: "torch.Tensor | None" = None) -> "torch.Tensor":
    """Select exact 48-bit BF16 score/original-ID codes; padding stays zero."""
    import torch

    rows, capacity = codes.shape
    result = torch.zeros((rows, 2048), device=codes.device, dtype=torch.int64)
    radix_select[(rows,)](
        codes, result, visible, capacity, min(2048, triton.next_power_of_2(capacity)), visible is not None, num_warps=4
    )
    return result


@triton.jit
def select_records(WORDS, LOW, LENGTHS, BOUNDARY, OUT, STRIDE: tl.constexpr, B: tl.constexpr):
    """Score-first selection on logical lengths; IDs resolve only exact ties.

    Keye stores the entire BF16 score in two bytes. Unlike the reference FP32
    coarse cells, this exact representation needs no boundary-score replay.
    Boundary storage covers every possible record, including all-equal scores.
    """
    row = tl.program_id(0)
    count = tl.load(LENGTHS + row)
    t = tl.arange(0, B)
    bins = tl.arange(0, 256)
    direct = tl.arange(0, 2048)
    tl.store(OUT + row * 2048 + direct, -1)
    if count <= 2048:
        word = tl.load(WORDS + row * STRIDE + direct, direct < count, 0)
        tl.store(OUT + row * 2048 + direct, (word & 0xFFFFF).to(tl.int32), direct < count)
    else:
        histogram = tl.full((256,), 0, tl.int32)
        for start in range(0, count, B):
            word = tl.load(WORDS + row * STRIDE + start + t, start + t < count, 0)
            bucket = tl.where(start + t < count, word >> 20, 256)
            histogram += tl.histogram(bucket.to(tl.int32), 256)
        above = tl.sum(histogram, 0) - tl.cumsum(histogram)
        boundary = (above < 2048) & (above + histogram >= 2048)
        high = tl.sum(tl.where(boundary, bins, 0), 0)
        higher = tl.sum(tl.where(bins > high, histogram, 0), 0)
        fine = tl.full((256,), 0, tl.int32)
        for start in range(0, count, B):
            word = tl.load(WORDS + row * STRIDE + start + t, start + t < count, 0)
            equal = (start + t < count) & ((word >> 20) == high)
            low = tl.load(LOW + row * STRIDE + start + t, equal, 0).to(tl.int32)
            fine += tl.histogram(tl.where(equal, low, 256), 256)
        above = tl.sum(fine, 0) - tl.cumsum(fine)
        boundary = (above < 2048 - higher) & (above + fine >= 2048 - higher)
        low_cut = tl.sum(tl.where(boundary, bins, 0), 0)
        cutoff = (high << 8) | low_cut
        written, tied = 0, 0
        for start in range(0, count, B):
            valid = start + t < count
            word = tl.load(WORDS + row * STRIDE + start + t, valid, 0)
            low = tl.load(LOW + row * STRIDE + start + t, valid, 0).to(tl.int32)
            score = ((word >> 20) << 8) | low
            better = valid & (score > cutoff)
            equal = valid & (score == cutoff)
            dst = written + tl.cumsum(better.to(tl.int32)) - 1
            tl.store(OUT + row * 2048 + dst, (word & 0xFFFFF).to(tl.int32), better)
            tie_dst = tied + tl.cumsum(equal.to(tl.int32)) - 1
            tl.store(BOUNDARY + row * STRIDE + tie_dst, word & 0xFFFFF, equal)
            written += tl.sum(better.to(tl.int32), 0)
            tied += tl.sum(equal.to(tl.int32), 0)
        # Compaction changes lane ownership; publish all boundary IDs to the CTA.
        tl.debug_barrier()
        needed = 2048 - written
        if needed == tied:
            ids = tl.load(BOUNDARY + row * STRIDE + direct, direct < needed, 0)
            tl.store(OUT + row * 2048 + written + direct, ids.to(tl.int32), direct < needed)
        else:
            prefix = tl.full((), 0, tl.uint32)
            mask = tl.full((), 0, tl.uint32)
            remaining = needed
            # IDs are exact 20-bit document-local IDs, not compact ordinals.
            for byte in tl.static_range(2, -1, -1):
                hist = tl.full((256,), 0, tl.int32)
                for start in range(0, tied, B):
                    ids = tl.load(BOUNDARY + row * STRIDE + start + t, start + t < tied, 0)
                    code = 0xFFFFF - ids
                    active = (start + t < tied) & ((code & mask) == prefix)
                    bucket = tl.where(active, (code >> (8 * byte)) & 255, 256)
                    hist += tl.histogram(bucket.to(tl.int32), 256)
                above = tl.sum(hist, 0) - tl.cumsum(hist)
                edge = (above < remaining) & (above + hist >= remaining)
                chosen = tl.sum(tl.where(edge, bins, 0), 0).to(tl.uint32)
                remaining -= tl.sum(tl.where(bins > chosen, hist, 0), 0)
                prefix |= chosen << (8 * byte)
                mask |= tl.full((), 255, tl.uint32) << (8 * byte)
            emitted = 0
            for start in range(0, tied, B):
                ids = tl.load(BOUNDARY + row * STRIDE + start + t, start + t < tied, 0)
                live = (start + t < tied) & (0xFFFFF - ids >= prefix)
                dst = written + emitted + tl.cumsum(live.to(tl.int32)) - 1
                tl.store(OUT + row * 2048 + dst, ids.to(tl.int32), live)
                emitted += tl.sum(live.to(tl.int32), 0)


@triton.jit
def bootstrap_scores(Q, K, W, VALID, COMMON, SOURCE, OUT, ROWS: tl.constexpr, N: tl.constexpr):
    anchor, block = tl.program_id(0), tl.program_id(1)
    t = block * 128 + tl.arange(0, 128)
    end = tl.load(COMMON)
    result = tl.full((128,), float("-inf"), tl.bfloat16)
    if (tl.load(SOURCE) == 0) & (block * 128 < end):
        row = anchor * (ROWS - 1) // 3
        h, d = tl.arange(0, 16), tl.arange(0, 64)
        q = tl.load(Q + row * 1024 + h[:, None] * 64 + d[None, :])
        w = tl.load(W + row * 16 + h)
        live = (t < N) & (t < end) & tl.load(VALID + t, t < N, False)
        keys = tl.load(K + t[None, :] * 64 + d[:, None], live[None, :], 0)
        scores, _ = score_tile(q, keys, w)
        result = tl.where(live, scores, float("-inf"))
    tl.store(OUT + anchor * N + t, result, t < N)


@triton.jit
def seed_votes(COLD, HISTORY, COMMON, SOURCE, VALID, VOTES, N: tl.constexpr, HISTORY_COUNT: tl.constexpr):
    t = tl.program_id(0) * 256 + tl.arange(0, 256)
    if tl.load(SOURCE) != 0:
        ids = tl.load(HISTORY + t, t < HISTORY_COUNT, -1).to(tl.int32)
    else:
        ids = tl.load(COLD + t, t < 8192, -1).to(tl.int32)
    live = (ids >= 0) & (ids < N) & (ids < tl.load(COMMON))
    live &= tl.load(VALID + ids, live, False)
    tl.atomic_add(VOTES + ids, 1, live, sem="relaxed")

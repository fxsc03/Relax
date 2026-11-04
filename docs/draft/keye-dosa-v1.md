# Keye-VL2 DoSA v1 implementation notes

This is an opt-in implementation and validation record, not a production
performance recommendation. The default remains `dsa`. The parallel optimization
revision and its new measurements are documented below; the original v1 results
are retained explicitly as historical evidence.

## Baseline and scope

Work stays on `feat/fengxiaoshi/DoSA`, at original HEAD
`bba4a30735e8f1acb7946e7118766370cb72409b`. The fetched main reference is
`a155b620251079e98ed5a4db8622f8450b02d3e0`. Its `relax/models/keye_vl2/`
contents are identical to HEAD before these edits. The branch was not rebased
or merged; unrelated uncommitted changes were preserved. Main's additional
CI/test-directory reorganization is not included in this working-tree change.

References:

- [DoSA paper, especially §2 and Appendix C](https://arxiv.org/html/2610.13990v1)
- [Original Keye integration](https://github.com/redai-studio/Relax/pull/498)
- [Author implementation](https://github.com/llm-sys/dosa-sglang), inspected at
  `582a8a151d5b14bd3b9d79e932a6390638d11e5a`. Its build requires SM100a,
  DeepGEMM headers, FP8 Q/K, 32/64 heads and dimension 128. Those kernels are
  not transplanted into Keye's BF16 16-head, dimension-64 model.

The shared selector is called after the existing training CP gather and from
SGLang's prefill/extend branch. Decode and its graph path are unchanged.
Training only enters DoSA if **every actual indexer parameter is frozen**.
`no_grad`, inference mode, a zero auxiliary coefficient, and the absence of
an auxiliary loss are not substitutes for freezing. Trainable indexers keep
the original scorer, KL loss, AutoScaler and backward, including checkpoint
recomputation. Neither model parameters nor the optimization objective change.

## Configuration

Use the existing `--custom-config-path /path/to/keye-dosa.yaml` mechanism:

```yaml
keye_dsa_backend: dosa
keye_dosa_heads: 4
sglang_json_model_override_args: '{"keye_dsa_backend":"dosa","keye_dosa_heads":4}'
```

The first two fields are explicitly consumed by `KeyeModelProvider.configure_runtime`;
they do not depend on `bridge_keys`. The JSON override reaches SGLang's model
config through its existing server-argument plumbing; each Keye attention
layer copies these fields from that config. Merge these keys with any other
existing JSON model overrides rather than overwriting them.

Training must also freeze the indexer, for example with the existing
`--freeze-params-name-list '\.sa_indexer\.'` option. No automatic freezing is
performed. `keye_dosa_heads` accepts 1–15, defaults to 4, and the configured seed capacity
is 4096 (the benchmark can also test other supported capacities). The two backends can be switched independently for diagnosis.
To disable, use `dsa` in both configurations or remove the new fields.

## Parallel selector (optimization revision)

The original streaming implementation is retained in `dosa_kernels.py` for
reference; `dosa_select` now dispatches `dosa_parallel.py`. Default configuration
and train/decode gates are unchanged. No experimental setting was applied to the
running eight-GPU experiment.

- Four queries share uniformly spaced seeds from the smallest visible prefix
  among group members with more than S valid keys. Shorter members use all their
  visible keys. Seeds are distinct, valid, visible and ordered by original ID.
  The default S remains 4096; the selector/benchmark additionally accept 2048
  and 8192 for measurement, without adding training arguments.
- Seed refinement uses the original scorer; native DSA Top-K establishes tau.
  A fused metadata kernel avoids intermediate `[Q,S]` int64 rank arrays.
- Query preparation retains the original positive projection, negative aggregate
  and numerical envelope. A second small kernel packs BF16 screening vectors
  and a conservative scalar L2 allowance. Key norms are computed once per call.
- Screening runs over query groups × non-seed key tiles. Each group's matrix
  has four padded groups of retained heads plus a negative aggregate. A single
  BF16 tensor-core dot reuses the key tile across queries; the head dimension is
  reduced once with directed rounding. There is no per-head coordinate interval
  dot in the normal screening path and no `[Q,H,N]` intermediate.
- Seed ranks are excluded **before** loading keys or executing screening MMA.
  For base prefix B and S seeds, complement ordinal t maps to
  `ceil((t+1)*B/(B-S))-1` if `t < B-S`, otherwise `t+S`.
  This is the complement of `floor(j*B/S)`, not an approximate seed mask.
- A GPU prefix sum plus parallel scatter compacts survivors across original
  tiles into a contiguous per-query array of original token IDs. Refinement
  CTAs load full 128-key tiles from this array; only its last tile is padded.
  CTAs beyond the device-side count skip MMA. Retained screening dots are not
  reused for exact scoring, preserving the original scorer's arithmetic.
- Final selection uses six 256-bin radix passes over the 48-bit ordered BF16
  score/original-ID code. This obtains the exact global Kth code without
  repeatedly sorting K-element arrays. A prefix scan packs codes above the
  cutoff into K slots. Each pass has one CTA per query, while screening and
  refinement remain query × tile parallel. This final stage is still a
  scalability bottleneck at larger N. Invalid code zero decodes to ID -1;
  compact-buffer positions never decide a tie. Rows with <=K visible keys
  directly copy their valid seed IDs and skip the radix passes.


Candidate capacity is exactly `N-S` per query: the full complement of the seed
set. All survivors, including all-tie/unsafe cases, fit without truncation or a
CPU capacity decision. Scratch is now O(Q(N-S)), deliberately trading memory
for parallelism: bool flags, int32 candidate IDs and tile counts, int64 packed
score/ID records for seed Top-K plus survivors, and final K selected codes.
Unused records are zero initialized; rejected keys are never fully scored.
This is not an original-domain dense score matrix. Peak allocated workspace is
reported by the benchmark. Very large Q×N can still exhaust memory; there is
no hidden OOM-catching allocator or undersized candidate pool.

N<=2048 directly returns visible valid IDs. By default 2049<=N<=4096 requests
original DSA because no pruning is possible. In mixed long batches, rows with
<=K visible keys skip scoring and plan preparation; their direct IDs pass
through the batched selection buffers. The native seed-selection launch still has batched workspace overhead for
these rows; seed scoring and final radix passes are skipped.
Padding, CP zigzag gathering, document isolation, logical causal offsets,
SGLang prefix/cache writing and the main attention backward remain unchanged.
There is no persistent activation-dependent cache across calls or training steps.

## Numerical certificate

`indexer_scores.score_tile` is shared by the original dense/paged scorer, seed
scoring and candidate refinement. It preserves the 16×128×64 BF16 MMA geometry,
BF16 dot cast, scale cast, BF16 weighted-ReLU cast, FP32 head reduction, and
final BF16 score cast. Candidate compaction changes columns, not the reduction
geometry for each dot product.

The screened real score includes the factor 1/8. To dominate the **computed**
score, the coordinate allowance also includes

```text
J[d] = 2^-6 * sum_h abs(w[h]) * abs(q[h,d])
numerical allowance = (1/8) * sum_d J[d] * abs(k[d])
```

This coefficient is not an empirical epsilon. For admitted BF16 inputs define
`u = 2^-8`, `v = 2^-23`, and `gamma_m = m*v/(1-m*v)`. Allowing up to 128
single-precision rounding errors in the 64-term MMA (including truncation),
16 in the head reduction, and three BF16 roundings gives

```text
(1 + gamma_128) * (1 + u)^3 * (1 + gamma_16) - 1 < 2^-6.
```

The scale by 1/8 is exact in the admitted range. The bound is relative to the
sum of absolute contributions, so it covers cancellation and negative weights.
The reference BF16 score may differ from an exact-real dot near a midpoint;
the certificate accounts for that difference rather than redefining the scorer.

Each nonzero input coordinate/weight admitted to the certificate has magnitude
in `[2^-32, 2^16]`. BF16 input quantization then gives a smallest input quantum
of `2^-39`; even a cancelling dot, scaling and weighting remain on a grid no
finer than `2^-120`, above the normal FP32/BF16 minimum. Intermediate upper
magnitudes also remain below overflow. This excludes underflow/overflow from
the relative-error argument for the original scorer.

Preparation uses the **stored** nonnegative projection coefficient, not an
assumed exact least-squares optimum. Directed arithmetic constructs folded
weight intervals `[wl, wh]`, aggregate intervals `[al, ah]` and a nonnegative
coordinate envelope E (including J above).

The new Tensor Core screen has an additional, separately certified allowance.
Let a_hat be the **actual stored BF16** midpoint of `[al, ah]`, and
`delta[d] = max(abs(al[d]-a_hat[d]), abs(ah[d]-a_hat[d]))`, rounded outward.
With upward-rounded L2 norms and `g = 2^-15 > gamma_128`, compute once per query:

```text
C = ||E + delta||_2
    + g * (sum_retained max(abs(wl), abs(wh)) * ||q||_2 + ||a_hat||_2)
U = (sum_retained wh * ReLU(dot_TC(q,k))
     - ReLU(dot_TC(a_hat,k)) + C * ||k||_2) / 8
```

Cauchy–Schwarz bounds both the coordinate residual and aggregate conversion
error. ReLU is 1-Lipschitz, so each MMA dot error contributes at most
`g*||q||_2*||k||_2`, multiplied by the maximum coefficient magnitude. Using wh
is safe even for negative weights, because ReLU is nonnegative. All operations
after the MMA, including reductions, squared norms and square roots, round
upward; fusion is disabled. The dot allowance uses the same full-FP32-ulp
contract as the original certificate. This avoids coordinate interval work
per candidate, at the cost of a potentially looser L2 bound.

Only `U < tau` rejects a key. Nonfinite or out-of-range queries/weights/aggregate
vectors invalidate the whole row certificate; nonfinite/out-of-range keys
invalidate that key's certificate. Such bounds become +inf and those candidates
are fully scored, rather than lost. Certificate recovery, backend fallback and
candidate-buffer capacity are separate concepts in the benchmark. No empirical
epsilon, FP8 error coefficient or CPU tensor read is used.

The numerical contract relies on SM80/SM90 BF16 MMA accumulating to at least
single precision. The [PTX precision contract](https://docs.nvidia.com/cuda/parallel-thread-execution/)
does not specify accumulation order or a rounding direction; the allowance
therefore uses a full FP32 ulp bound rather than assuming round-to-nearest.
Other device families are not opted in. Unsupported device,
dtype, geometry or length (>262144) returns to the caller's original DSA path.

## Original v1 reproduction and historical measurements

Use the existing matching training environment, with Megatron on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:${MEGATRON:-/root/Megatron-LM}:${PYTHONPATH:-}"
python -m pytest -q tests/models/keye_vl2/test_dosa.py \
  tests/models/keye_vl2/test_indexer_scores.py \
  tests/models/keye_vl2/test_cp_attention.py::test_keye_cp2_matches_cp1_outputs_and_cross_rank_gradients

python examples/models/keye-vl2/benchmark_dosa.py \
  --checkpoint "$KEYE_CHECKPOINT" --length 8192 --rows 4 \
  --heads 4 --repeats 5 --output /tmp/keye-dosa-layer0.json
```

`--text /path/to/history.txt` uses a real prompt/tool-history text and repeats
tokens if necessary to reach the requested length. This lightweight checkpoint
benchmark computes actual layer-0 indexer inputs from token embeddings and
the first input norm; it does not claim deeper-layer measurements. For captures
from a full model forward use `--inputs /path/to/indexer-inputs.pt`: a dictionary,
or list of layer dictionaries, containing post-RoPE/Hadamard `iq`, `ik`, `iw`,
logical `offsets`, and optional document-local `valid` mask. Do not save stale
summaries across training steps. The benchmark checks compact selection before
timing and writes GPU counters, peak extra allocation and kernel timing data.

The streamed kernel fuses screening, compaction, refinement and running Top-K;
its time is reported as a combined stage, not fabricated per-stage estimates.
Seed scoring, seed Top-K and plan preparation appear as separate GPU kernels.
Absolute timings on an occupied GPU are diagnostic only. Selector timing is
not a claim about full-model prefill, rollout latency or training step time.

## Historical v1 validation record and limitations

Hardware available during development: NVIDIA H20-3e, 143771 MiB per device,
PyTorch 2.13.0+cu129, Triton 3.7.1. All eight devices already hosted an existing
training experiment; it was not stopped or reconfigured. Small GPU tests run
alongside it. Full-model training-step benchmarking requires a separate idle
node and has not been substituted with an indexer microbenchmark.

The initial implementation favors correctness and bounded storage. Per-query
SIMT interval evaluation and repeated streaming Top-K merges can cost more than
the full tensor-core scorer. Head budget, seed quality and real layer activation
statistics matter. Production launch scripts remain unchanged.

### Measured results (2026-11-04)

The following are real layer-0 indexer inputs from the existing Keye checkpoint,
using recorded ALFWorld prompt/response text, repeated or truncated to the stated
length. All inputs are BF16, H=16, D=64, K=2048; seed capacity is 4096. Times
include selector metadata, scoring, selection and final `compact_indices`.
The median is over seven warmed repetitions on an occupied H20-3e.

| Q | Keys | Retained heads | DSA ms | DoSA ms | Non-seed pruning |
|---:|---:|---:|---:|---:|---:|
| 4 | 2048 | 4 | 0.0846 | 0.0873 | not applicable |
| 4 | 8192 | 4 | 0.1076 | 1.3504 | 0% |
| 4 | 16384 | 4 | 0.2507 | 2.3604 | 0% |
| 4 | 32768 | 4 | 0.2349 | 4.2466 | 0% |
| 128 | 8192 | 4 | 0.1789 | 1.5482 | 0% |
| 1024 | 8192 | 4 | 0.8369 | 18.9409 | 0.000082% |
| 128 | 8192 | 15 | 0.1777 | 5.4853 | 35.14% |
| 1024 | 8192 | 15 | 0.8364 | 46.8918 | 34.72% |

The 4096-key seed-only implementation measured 0.2828 ms versus 0.1039 ms for
DSA before adding the short-prefix fallback. This case now requests DSA.
Short-prefix skipping is not reported as DoSA pruning or as a speedup.

For Q=128/N=8192/R=4, warmed CUDA-event launch intervals were: seed scoring
0.0289 ms, seed Top-K 0.1140 ms, bound-plan preparation 0.0085 ms, fused
screen/compact/refine/streaming-Top-K 1.2134 ms. These single-profile intervals
need not sum to the median total: metadata operations and Python launch gaps
are also included in total timing. CUPTI was unavailable, so the benchmark
uses CUDA events around warmed launches, without claiming internal per-stage
timings for the fused kernel. For Q=1024 the fused stage took 17.6878 ms.

Peak extra PyTorch allocator usage was 10,617,344 bytes for Q=128 and 84,021,248
bytes for Q=1024 at N=8192. Numerical-certificate bypass occurred on 0% of rows
in these captures. Constructed out-of-range/nonfinite tests separately exercise
the retain-and-score recovery. No global candidate-capacity recovery is needed:
all-survivor, all-tie and non-contiguous-ID cases fit the bounded streaming design.

**No measured configuration above accelerated the selector.** Increasing the
head budget tightened the certificate but made interval screening more costly.
The 4-head single-anchor/coordinate residual bound was too loose for this
layer. Batching Top-K merges reduced an earlier 10.8 ms implementation to about
1.35 ms at Q=4/N=8192, but that is still slower than DSA. Do not enable this
backend in production on the basis of its pruning rate or theoretical FLOPs.

Local raw artifacts are in `/tmp/keye-dosa-validation/`: `sweep.json`,
`batches.json`, and the small reproduction drivers. The checked-in benchmark
accepts the same history via `--text` and supports captured deeper-layer inputs
via `--inputs`.

### Executed validation

- The main GPU regression command above, plus `test_decode.py` and `test_topk.py`,
  passed 69 tests after the short-prefix policy and batched Top-K merge changes.
- A supplementary SGLang run passed all five tests: short and 4113-token
  prefill, 8192-token-prefix extend with permuted physical cache slots, and
  actual model-constructor propagation for both `dsa` and `dosa`. This brings
  the suite to 72 distinct tests; the supplementary run repeats two existing
  cases rather than adding five distinct cases.
- Checkpoint validation uses a 4113-token document, confirms that both the
  no-grad first forward and grad-enabled recomputation actually enter the long
  DoSA path, and compares their selected IDs bitwise. Main forward outputs are
  bitwise equal to DSA. Main-attention gradients use `atol=1e-9, rtol=1e-5` to
  accommodate the unchanged fused SDPA backward's parallel reductions.
- CP2 compares against CP1 with a 4160-token long document, another short
  document, per-document padding and three-axis mRoPE. Frozen-indexer outputs
  and cross-rank main-model gradients pass. Trainable-indexer checkpoint tests
  assert DoSA is never called and that auxiliary KL gradients remain nonzero.
- Tests cover lengths below/equal/above K and 4096, non-contiguous tensors and
  candidate IDs, empty causal rows, BF16 midpoint inputs, negative weights,
  zero/all-tie scores, three head budgets, nonfinite/out-of-range values,
  and unsupported-input fallback. GPU bound matrices are materialized **only
  in tests** and compared elementwise against the original BF16 scorer.
- Targeted pre-commit hooks and `git diff --check` pass. Repository-wide
  formatting is intentionally not run over unrelated existing local edits.

### Historical v1 remaining validation and optimization

This is a correctness-oriented first implementation, **not an achieved
acceleration result**. Full-model prefill/extend timing, complete frozen-indexer
training step time, multiple real deep-layer captures, and TP/SP end-to-end
validation remain to be run on an idle matching node. CP2 and attention-level
prefill/extend tests do not substitute for those measurements. There was no
distributed training launch, model deployment, dependency/driver upgrade,
push or merge in this task.

Useful next work is tighter multi-anchor/joint residual preparation, grouped
query screening with a proven numerical enclosure, and accumulation of live
keys across tiles before Tensor Core refinement. Optimize against captured
real layers and complete selector time; the unchanged main attention backward
and CP communication still limit end-to-end benefit.

## Optimization validation (2026-11-04, current code)

The original working tree, source hashes and untracked DoSA implementation were
saved before editing in `/tmp/keye-dosa-v2/baseline/`. Work remains on
`feat/fengxiaoshi/DoSA`, HEAD `bba4a30735e8f1acb7946e7118766370cb72409b`.
There was no commit, branch switch, fetch/merge, deployment or dependency upgrade.
All recorded benchmarks ran alongside the existing experiment. After recording
these results, the user authorized stopping ALFWorld for exclusive validation,
then cancelled further validation because the machine was being reclaimed.
No exclusive-run performance result is claimed.

Changed this revision: `dosa.py`, `dosa_kernels.py` (skip seed-only empty tiles),
new `dosa_parallel.py`, `test_dosa.py`, `benchmark_dosa.py`, new
`capture_dosa_inputs.py`, this note and the machine-readable results below.
Provider/attention/SGLang integration changes from v1 remain in place without
further changes to their training or decode gates.

### Inputs and timing conditions

`capture_dosa_inputs.py` produced **real layers 0, 23 and 47** from the local
Keye checkpoint using a text-only, layer-streamed TP1 Relax DSA forward. This
loads one transformer layer at a time, uses the shared main sparse attention,
original BF16 scorer and Relax MoE arithmetic, and avoids loading the entire
30B model alongside the experiment. Input is recorded ALFWorld prompt/response
history repeated to 16384 tokens. These are not independent random hidden states
fed to deep indexers. N=8192 benchmarks slice the causal prefix of the same
capture. This collector is not an end-to-end SGLang engine benchmark.

All measurements use BF16, 16 indexer heads, dimension 64, K=2048 on H20-3e,
Torch 2.13.0+cu129 / Triton 3.7.1. GPU 7 was shared with the existing eight-GPU
training task. The normal selector plus `compact_indices` is timed with CUDA
events, three warmup pairs and seven interleaved DSA/DoSA measurements. Old and
new selectors replay identical saved tensors; their runs are separate shared
GPU windows. Individual stage diagnostics are additional instrumented calls,
not an additive partition of the total. Allocations, metadata, prefix sum and
all selection stages are included in the normal total.

[Machine-readable results](./keye-dosa-v2-results.json) contain 24 old-DoSA
and 60 new-DoSA measurements, all timing samples, min/max, counters, stage
intervals, compiler register/spill/shared-memory metadata, peak extra allocator
bytes and capture SHA256 hashes. Raw tensors, full row counters, logs and the
immutable old source snapshot were originally under `/tmp/keye-dosa-v2/` and
have now been copied to the shared repository filesystem at
`examples/models/keye-vl2/scripts/log/dosa-optimization-20261104/`. Shared-load
outliers are retained: for example layer 23/Q128/N8192/R4 ranged from 0.397 to
3.044 ms, while its median was 0.398 ms. These are not exclusive-GPU numbers.

### Complete selector comparison

Default R=4, S=4096; milliseconds. Ratio is **DSA / new DoSA**, so less than
one means slower than DSA. The old/new comparison must not be called a speedup
over DSA.

| Layer | Q | N | DSA | Old DoSA | New DoSA | DSA/new |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 128 | 8192 | 0.181 | 1.543 | 0.397 | 0.45× |
| 0 | 1024 | 8192 | 0.884 | 18.979 | 1.536 | 0.58× |
| 0 | 128 | 16384 | 0.239 | 5.167 | 0.588 | 0.41× |
| 0 | 1024 | 16384 | 1.293 | 37.093 | 5.051 | 0.26× |
| 23 | 128 | 8192 | 0.180 | 1.543 | 0.398 | 0.45× |
| 23 | 1024 | 8192 | 0.831 | 18.982 | 1.428 | 0.58× |
| 23 | 128 | 16384 | 0.235 | 5.164 | 0.587 | 0.40× |
| 23 | 1024 | 16384 | 1.287 | 37.095 | 5.052 | 0.25× |
| 47 | 128 | 8192 | 0.220 | 1.511 | 0.639 | 0.34× |
| 47 | 1024 | 8192 | 0.878 | 18.964 | 1.543 | 0.57× |
| 47 | 128 | 16384 | 0.236 | 5.164 | 0.586 | 0.40× |
| 47 | 1024 | 16384 | 1.285 | 37.098 | 5.052 | 0.25× |

**None of the 60 new real-activation configurations beat DSA.** The best ratio
was approximately 0.606×. Comparisons include R=4/8/15 and S=2048/4096/8192,
with Q=128/1024 and N=8192/16384 (R=8 and S=8192 are supplementary Q1024 cases).
Short-prefix fallback/all-valid shortcuts are not included as pruning gains.

For Q1024/N8192/S4096/R15, non-seed pruning was 30.82%, 27.66%, and 4.21% at
layers 0/23/47; the actual fraction of **all visible keys** not fully scored
was only 14.38%, 12.91%, and 1.97%. R4 remained essentially unpruned.
S2048 gives a weak threshold and almost no pruning in these captures. S8192
at N16384/R15 improved non-seed pruning (e.g. 50.85% at layer 23), but the
extra seed work and screening/selection overhead still lost to DSA.

### Bottleneck evidence and what changed

- Old Q1024/N8192/R4 fused scan: 17.76 ms, 255 registers/thread and zero compiler
  spill slots. Old R15: approximately 45.79 ms, 255 registers and **512 spill
  slots**. The old scan fuses bounds, local compaction, MMA and sorting; these
  timings do not isolate the interval arithmetic alone. Code inspection
  confirms per-head FP32 interval dot/reduction and screening of already
  scored seeds, plus serial per-query scan/merge work.
- At N8192 with S4096 and no pruning, the old code executes 32 seed MMA tiles
  plus about 64 half-filled scan/refine tiles per fully visible query. The
  new path executes 32 seed tiles plus 32 densely packed refinement tiles.
  These counts follow the executed conditional/tile geometry, not a hardware
  instruction-counter trace. Actual aggregate tile counts are recorded from
  device-side survivor counts. Q1024/N8192/R4 refinement fill was 98.26%; R15
  fill was 97.51–98.18% across captured layers.
- New layer0/Q1024/N8192/R4 diagnostic: seed metadata 0.054 ms; seed scoring
  0.201; seed Top-K 0.177; query preparation 0.021; bound packing 0.010; key
  norms 0.007; screening 0.182; prefix sum 0.006; compaction 0.035; refinement
  0.172; final radix selection 0.297. Normal complete selector: **1.536 ms**.
  Stage wrappers are not summed to manufacture a total; final compaction and
  allocation/launch gaps are included only in the complete measurement.
- Grouped screening uses 128 registers (R4) and no spill in the recorded
  configuration. Replacing repeated extraction of head rows from the MMA
  accumulator with one head-axis reduction removed an initially measured
  10.14 ms R15 screening bottleneck (down to about 0.169 ms at Q1024/N8192).
- Final selection avoids repeated 4096-element sorts and avoids data-dependent
  slow native int64 Top-K. A diagnostic sweep on the same captured candidate
  records chose radix block 2048 / 4 warps: 128 registers, zero spill,
  approximately 0.299/0.711 ms for N8192/16384. Block 4096 / 4 warps used
  255 registers and was slower. **Isolated-kernel numbers do not replace full
  selector timings:** normal N16384/Q1024 calls still showed roughly 3 ms
  final-selection intervals under shared execution. Occupancy, cache misses
  and memory traffic were not measured with hardware counters; their exact
  contribution to this gap remains unverified.
- Parallelism costs memory: Q1024/N8192/R4 peak extra allocated workspace was
  135.15 MiB, versus about 80 MiB in the old implementation. The largest new
  tested workspace was 258.25 MiB. Every real case used the DoSA path, with
  zero backend fallback and zero numerical-certificate bypass; no candidate
  was dropped to fit a smaller buffer. Full-capacity/all-tie tests cover the
  no-pruning capacity limit.

The remaining problem is both bound tightness and execution cost. With 16
heads, 4 retained heads plus the aggregate already cost five screening dot
products per non-seed key. Even ignoring preparation and selection, their
work requires more than 5/16 non-seed pruning to repay full-head refinement;
these real R4 captures do not provide it. R15 plus its aggregate has the same
number of screening heads as the original scorer. Larger seeds tighten tau
but consume more full scoring. Better execution alone does not fix these
tradeoffs. Next work should target tighter small-head residual bounds **and**
parallel final selection, using these same captures. Keep default DSA; do not
justify enabling DoSA with future aux training or theoretical FLOPs.

### Verification and reproducible commands

Executed on the final implementation:

- **55 passed**: `test_dosa.py`, `test_indexer_scores.py`,
  `test_keye_vl2_attention.py`. Includes original-scorer midpoint alignment,
  signed/zero/ReLU cases, q/k/w NaN/Inf recovery, direct elementwise GPU
  certificate dominance at R1/4/15, seed sizes 2048/4096/8192, saturated
  candidate capacity, cross-tile compaction, large ties with shuffled original
  IDs, radix tails at capacities 2049/8193/32769, frozen checkpoint IDs/output/
  gradients, trainable indexer KL/backward, config gates and prefill/extend.
- **1 passed**: CP2 DoSA vs CP1 on GPUs 5/6, packed long/short documents,
  per-document padding, mRoPE, zigzag and cross-rank main-model gradients.
- Targeted pre-commit checks and `git diff --check` pass. No unrelated files
  were formatted or committed.

Not executed: new four-GPU TP/SP end-to-end validation, actual multimodal
image/video encoder capture, complete SGLang model prefill/extend timing, or
full frozen-indexer training step timing. The user cancelled further validation before any new full-model experiment
was launched. Attention-level output tests and
layer-streamed captures do not substitute for these end-to-end measurements.
There is no claim that indexer speed translates to unchanged CP communication
or main-attention backward speed.

```bash
export PYTHONPATH="$PWD:/root/Megatron-LM:${PYTHONPATH:-}"
export ARTIFACTS="$PWD/examples/models/keye-vl2/scripts/log/dosa-optimization-20261104"
# Set CHECKPOINT and HISTORY to the local model and recorded ALFWorld text.
CUDA_VISIBLE_DEVICES=7 /opt/venv/bin/python examples/models/keye-vl2/capture_dosa_inputs.py \
  --checkpoint "$CHECKPOINT" --text "$HISTORY" --length 16384 --layers 0 23 47 \
  --output-dir "$ARTIFACTS/captures"

CUDA_VISIBLE_DEVICES=7 /opt/venv/bin/python examples/models/keye-vl2/benchmark_dosa.py \
  --inputs "$ARTIFACTS/captures/layer-23-n16384.pt" \
  --length 8192 --rows 1024 --heads 4 --seed-size 4096 --repeats 7 \
  --output "$ARTIFACTS/replay-new.json"

# Same input, immutable pre-edit source. Its seed capacity is fixed at 4096.
CUDA_VISIBLE_DEVICES=7 /opt/venv/bin/python examples/models/keye-vl2/benchmark_dosa.py \
  --inputs "$ARTIFACTS/captures/layer-23-n16384.pt" \
  --length 8192 --rows 1024 --heads 4 --repeats 7 \
  --baseline-dir "$ARTIFACTS/baseline/relax/models/keye_vl2" \
  --output "$ARTIFACTS/replay-old.json"

CUDA_VISIBLE_DEVICES=6 /opt/venv/bin/python -m pytest -q \
  tests/models/keye_vl2/test_dosa.py tests/models/keye_vl2/test_indexer_scores.py \
  tests/models/keye_vl2/test_keye_vl2_attention.py
CUDA_VISIBLE_DEVICES=5,6 /opt/venv/bin/python -m pytest -q \
  tests/models/keye_vl2/test_cp_attention.py -k dosa
```

Use `--heads 8/15`, `--seed-size 2048/8192`, `--length 16384`, and
`--rows 128` as applicable to reproduce the additional cases. On another node
choose available GPU IDs explicitly; none of these commands shuts down Ray
or an existing model service. The archived exploratory `final48-sweep.py` and `extra-sweep.py` retain their
original temporary paths; use the maintained benchmark commands above with
`ARTIFACTS` to replay captures after container replacement.

### Handoff before machine reclamation

Further validation was stopped at the user's request. Final status:

- At the validation handoff, source and reports were uncommitted on
  `feat/fengxiaoshi/DoSA`; subsequent publication is recorded separately in Git history.
- The target Ray job `keye-alfworld-temp06-prefixcache-20261103-203354`
  was confirmed `STOPPED`; its detached Serve applications were also shut down
  after verifying they all belonged to job `07000000`. No experiment restart
  or automatic checkpoint resume was performed. Existing checkpoint files
  were not modified by this cleanup.
- No exclusive performance sweep, additional TP/SP run or full-model
  prefill/extend/training-step benchmark was completed after that stop.
  The shared-load measurements above remain the final performance evidence.
- Baseline source/diff/manifest, three actual activation captures, benchmark
  JSON/logs, correctness/CP logs, original ALFWorld history, job-stop metadata
  and exploration drivers are preserved under the shared filesystem directory
  `examples/models/keye-vl2/scripts/log/dosa-optimization-20261104/`.
  `artifact-inventory.json` lists the archived files and sizes. This log
  directory is Git-ignored: it survives container replacement **if this shared
  mount is retained**, but a fresh Git clone does not contain these artifacts.
  Capture hashes are in `docs/draft/keye-dosa-v2-results.json`.

The implementation addresses expensive per-head screening, seed re-screening,
local-only packing and serial refinement. It is still an experimental opt-in
backend: no tested real-input configuration beat DSA, so DSA remains the default.

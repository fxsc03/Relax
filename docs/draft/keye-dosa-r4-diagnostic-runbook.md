# Keye DoSA R=4 剪枝诊断起跑文档

日期：2026-11-04。目标分支：`feat/fengxiaoshi/DoSA`。当前实现基准提交：`45d38ea`。

本文是新机器上的执行交接，不是已完成的统计报告。本轮仅编写文档，不启动 GPU 测试、安装依赖或提交训练任务。起跑前记录实际 commit 和工作区 diff，不能只依赖分支名。

## 1. 要回答的问题

历史真实激活测试中，默认 `R=4` 的非 seed 剪枝率接近零。需要区分三个原因：

1. 输入本身有大量接近 Top-K 边界或并列的分数。
2. seed 的第 K 大分数过低，能够安全淘汰的候选本来就少。
3. 当前筛选上界过松，放过了大量真实分数低于 seed 阈值的候选。

本次使用 **Keye 真实激活的 selector 诊断 microbenchmark**。主结果是分布统计，不是整模型吞吐、生成质量、训练加速或论文性能复现。历史共享 H20 计时与新机器的结果分开记录。

## 2. 申请机器与使用限制

| 资源 | 首轮建议 |
| --- | --- |
| GPU | 单张独占 H800 80GB，或可用的同代 Hopper GPU；当前 selector 仅准入 SM80/SM90 |
| CPU / 内存 | 建议至少 8 核、64GB RAM；这是准备数据和读取权重的资源预算，未做峰值测量 |
| 存储 | 建议至少 120GB 可用持久盘，容纳约 62GB 模型权重、输入、缓存和输出；镜像空间另计 |
| 软件环境 | 与当前 Keye 分支匹配的 Relax 完整环境，优先复用已验证镜像及 digest |
| GPU 时间 | 首轮累计最多 15 分钟，包括 GPU 预检、JIT、激活生成和统计；不是每条命令各 15 分钟 |

数据下载、tokenization、诊断代码准备和静态检查应在占用 GPU 前完成。开始 GPU 阶段后只使用一张指定卡，整个阶段由同一个 900 秒期限控制；每完成一个 case 即落盘。剩余时间不足时停止新增 case，保存部分结果并退出，不为凑齐矩阵自动延长。

脚本退出即释放 GPU，不保留后台服务或占卡进程。此次不需要 Ray、SGLang server 或训练启动脚本，也不执行集群清理命令。深层采集不自动纳入这 15 分钟窗口。

## 3. 环境与资产准备

### 环境要求

参考 [Keye 示例](../../examples/models/keye-vl2/README.md) 和 [镜像说明](../../docker/README.md)。示例要求匹配 SGLang 0.5.21 的训练环境，同时包含 Megatron/Bridge、Transformer Engine、Triton、FlashInfer 和 `fast_hadamard_transform`。单独升级 SGLang 版本号不能证明环境兼容；不在新机器上猜测或重建 CUDA/驱动栈。

上一台机器的环境失败记录是：Torch 2.9.1+cu129、Triton 3.5.1、SGLang 0.5.9、FlashInfer 0.6.3；Hadamard 扩展缺失。已有 55 项测试中，23 通过，22 项因 `top_k` 不支持 `deterministic` 参数失败，5 项缺 Hadamard，5 项缺 `KVWriteLoc`。不能把该环境直接视为可起跑环境。

新机器的最低预检项目：

- 确认指定 GPU 空闲、型号和计算能力；记录驱动、Torch/CUDA、Triton、SGLang、FlashInfer、MCore、Bridge、TE 版本及镜像 digest。
- 能导入 `relax.models.keye_vl2.indexer.KeyeIndexer`、原始 scorer 和 DoSA kernel。
- 能导入并实际执行 Hadamard 扩展，以及小规模 BF16 scorer / screening kernel。
- 若执行现有完整 selector benchmark，必须通过原始 FlashInfer Top-K 调用，不能去掉 deterministic/tie-break 参数来“跑通”。
- 深层采集另需检查当前 SGLang 接口；第一层统计成功不代表深层环境已验证。

本轮文档不授权安装依赖。若准备新环境需要安装或升级软件，先按仓库 AGENTS.md 确认范围。

### 模型与旧资产

模型为 `Kwai-Keye/Keye-VL-2.0-30B-A3B`，保留 config、tokenizer、索引 JSON 和全部 safetensors shard。转移时记录模型 revision 或权重校验信息。模型目录由执行者提供，不把旧机器绝对路径写入诊断代码。

旧的三层激活曾归档在 `examples/models/keye-vl2/scripts/log/dosa-optimization-20261104/`，该目录被 Git 忽略，当前 checkout 不包含它。若可以恢复，先与 [历史结果 JSON](./keye-dosa-v2-results.json) 中的 `capture_sha256` 比对；否则不能宣称逐点重现旧输入。

现有临时结果中标记为“checkpoint indexer weights + embedding replay”的深层测试，不能当作真实深层激活。真正的第 23/47 层输入必须经过前面各层的模型前向。

## 4. Benchmark 矩阵与输入约束

首轮采用以下矩阵；执行时先完成 8K/16K，再向长上下文扩展。

| 项目 | 设置 |
| --- | --- |
| 模型层 | 第 0 层，真实 embedding → input norm → indexer → RoPE/Hadamard 输出 |
| 主输入 | InfiniteBench En.QA 长文本、Code.Debug 代码，各选 2 个不同 context |
| 重复输入对照 | 1 条记录来源的 ALFWorld 短历史，显式重复扩展；不是旧样本时标为新对照 |
| 长度 N | 8192、16384、65536、131072，按本地 Keye tokenizer 计数 |
| Query | 每个前缀末尾连续 128 个 query，保留各自真实 causal offset |
| 主配置 | H=16、D=64、K=2048、R=4、S=4096、BF16 |
| 总量 | 4 个主输入 + 1 个重复对照，各 4 个长度，最多 20 个主 case |

数据来源：[InfiniteBench 仓库](https://github.com/OpenBMB/InfiniteBench)、[数据集](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench)。只使用 context，不进行任务回答评分。En.QA 的书籍内容带有实体替换，不能称为未经处理的原始书籍。

选择样本时按固定 ID 顺序挑选满足长度要求的不同 context，并记录 dataset revision、sample ID、context SHA256、tokenizer 信息和 token 数。不同题目可能共用 context，必须按 context hash 去重。主输入使用自然连续前缀，不拼接无关样本、不循环补足；不足目标长度则换样本或记录缺失。所有长度尽量来自同一批足够长的 context，使长度对照保持配对。

短历史对照需要记录原始 token 数和实际重复规则。若旧 ALFWorld 历史无法恢复，使用来源清楚的新历史，不声称它是原始历史复现。

ALFWorld 重复输入与书籍/代码同时改变了领域和重复方式，差异不能全部归因于重复。若初步结果指向输入因素，下一轮应增加同源配对：同一长 context 的自然前缀，对比它最初 512 个 token 重复到相同长度。该消融不自动扩展首轮 GPU 预算。

主结果固定 seed 规则、head selection 和数值保护项。R=15、全前缀采样、最佳 anchor 等只能作为后续单变量消融，不能混入 R=4 主结果。若第一层足以定位现象，先交付首轮报告；是否采集深层另行决定。

## 5. 统计定义

### 同一个候选集合

对每个 query q 定义：

- Ωq：原始 token ID 满足 causal offset 且 valid 的可见集合。
- Sq：当前实现实际生成的 seed 集合，大小为 4096。
- Cq = Ωq \ Sq：可见、有效、非 seed 的候选集合。
- s(q,i)：`dense_indexer_scores` 返回的原始 BF16 分数。
- τ_seed(q)：Sq 内第 2048 大的原始分数。
- τ_full(q)：Ωq 内第 2048 大的原始分数。
- U(q,i)：当前 `dosa_parallel.screen` 实际计算的数值上界，包含数值保护项。

三个主要指标如下：

| 指标 | 定义 | 解释 |
| --- | --- | --- |
| 当前阈值可剪比例 | `count(i∈Cq: s(q,i) < τ_seed(q)) / len(Cq)` | 即使上界完全准确，在当前阈值下能拒绝多少 |
| 实际上界剪枝比例 | `count(i∈Cq: U(q,i) < τ_seed(q)) / len(Cq)` | 当前 R=4 的实际拒绝比例 |
| seed 阈值差距 | `τ_full(q) - τ_seed(q)` | seed 相对全量选择损失了多少阈值 |

必须使用严格 `<`；等于阈值的项保留。不能把 seed 或不可见 padding 放进分母，不能把短前缀直接返回、backend fallback 当作剪枝。

补充报告以下指标：

- `count(s < τ_full) / len(Cq)`：完整阈值下的 score-only 可拒绝比例；和当前阈值可剪比例的差反映 seed 造成的机会损失。
- `count(s < τ_seed 且 U >= τ_seed) / len(Cq)`：当前上界错过的安全剪枝机会，不是选错 Top-K。
- `s == τ_seed`、`s == τ_full` 的比例，解释 BF16 并列边界的影响。
- `U-s` 与 `U-τ_seed` 的 P10/P50/P90/P99，以及正/负权重 head 数。
- 每个 query 的两个阈值及差值；跨 query 给出中位数、P90、最小/最大值。先转换为 FP32 再减，避免额外 BF16 舍入。
- 非有限 score、无穷上界、certificate bypass、fallback 的数量；不静默删除这些行。

原始分数出现 NaN/Inf 的 case 标为异常，不将其并入正常阈值归因。有限分数对应 +Inf 上界时仍计入候选分母，按无法剪枝处理并单列数量。候选为空时比例为 null，不填零。若做尺度归一化，可用逐 query 分数 IQR；IQR 为零时归一化结果置 null，保留原始差值。

聚合时同时提供 query 比例分布和 `sum(拒绝数)/sum(候选数)` 的总体比例。不能把逐 query 百分比的简单平均误称为总体比例，也不把所有输入混为一行。

### 数值与映射检查

1. 必须复用原始 BF16 scorer，包括 dot、缩放、weighted ReLU、head reduction 的舍入边界；不能改用普通 FP32 einsum 充当基准。
2. seed 是四 query 分组规则生成的实际 seed。不能给每个 query 另造一组“近似相同”的均匀 seed。
3. `screen` 的 BOUNDS 列按非 seed complement ordinal 排列，不是原始 token ID。用该分组的 BASE、`complement` 和 VALID_IDS 恢复原 ID，再与 dense score 对齐，并应用逐 query 的 causal/valid mask。
4. 对有限分数检查 `U >= s`、`τ_seed <= τ_full`、实际上界剪枝比例不超过当前阈值可剪比例；任一失败先排查正确性，不解释性能。
5. 从同一个 kernel 输出的 live counts 核对统计剪枝数，避免统计映射有误却得到合理百分比。

这里只需要标量第 K 大分数时，可对原始 BF16 scorer 的输出转 FP32 后用 PyTorch 精确选择阈值，ties 不改变阈值数值。这样不必为了统计依赖 FlashInfer 的索引 tie-break 接口；但这不是原始 Top-K 性能测试，不能据此报告 selector 加速比。完整 selector 的 ID 等价测试仍需原始选择契约。

## 6. 当前工具能做什么、还缺什么

| 入口 | 当前能力 |
| --- | --- |
| [benchmark_dosa.py](../../examples/models/keye-vl2/benchmark_dosa.py) | 生成真实 layer-0 输入或读取捕获张量，校验 compact IDs，测 selector 耗时与候选计数；尚未输出本文三个归因指标 |
| [capture_dosa_inputs.py](../../examples/models/keye-vl2/capture_dosa_inputs.py) | 逐层加载权重采集真实深层激活；依赖匹配 SGLang，完整深层前向不纳入默认短时首轮 |
| [test_dosa.py](../../tests/models/keye_vl2/test_dosa.py) 的 `test_grouped_tensorcore_certificate` | 展示 `screen(..., BOUNDS=..., DUMP=True)` 的上界观测入口；其特殊同 offset 测试映射不能直接代替真实 causal batch 映射 |
| [dosa.py](../../relax/models/keye_vl2/dosa.py) | 当前 seed、valid IDs、分组、candidate capacity 和调用流程的依据 |

**TODO(agent)：起跑前实现只用于诊断的驱动。当前仓库没有可一键输出本报告的命令。** 不把下面的 smoke 命令误认为三个统计指标已经实现。

诊断驱动最低要求：

- 复用已有 layer-0 生成逻辑；不需要把 48 层模型全部放进 GPU。主输入预先验证长度，禁止悄悄触发现有 helper 的重复补长行为。
- 记录输入 manifest，捕获实际 seeds、threshold、BASE、valid IDs 和 screen bound；保留生产参数与数值运算，新增观测不修改模型后端。
- 使用小批量 query 计算 dense score 与上界，不构造 `[Q,H,N]` 大矩阵；统计后释放 GPU tensor，只保存可复核的行级摘要和必要的小样本。
- 一次 GPU 进程内顺序执行，接收总截止时间，逐 case 原子落盘；完成后退出。JIT 编译也计入预算。
- 先完成一个 8K case，验证映射与不等式，再扩展矩阵。优先覆盖各输入的 8K/16K，再跑 64K/128K。
- 预检失败立即记录并退出，不启动重试循环，不以安装缺失依赖的过程占住 GPU。

## 7. 新机器起跑顺序

### A. CPU 准备阶段

确认代码、镜像、模型和数据均已就绪，补齐并静态检查诊断驱动。以下变量由执行者填写，命令在仓库根目录执行：

```bash
export KEYE_CHECKPOINT=/path/to/Keye-VL-2.0-30B-A3B
export KEYE_DIAG_INPUT=/path/to/validated-context.txt
export KEYE_DIAG_OUTPUT=/path/to/persistent/keye-dosa-r4-diagnostics
export KEYE_DIAG_GPU=0
export MEGATRON_PATH=/path/to/Megatron-LM
export PYTHONPATH="$PWD:$MEGATRON_PATH:${PYTHONPATH:-}"
mkdir -p "$KEYE_DIAG_OUTPUT"
git rev-parse HEAD > "$KEYE_DIAG_OUTPUT/commit.txt"
git diff --binary > "$KEYE_DIAG_OUTPUT/worktree.patch"
python -m pip freeze > "$KEYE_DIAG_OUTPUT/packages.txt"
```

保存数据/tokenizer/模型 manifest、镜像 digest 和驱动入口。先确认输入至少有 8192 个 Keye token；长输入各长度使用同一 context 的前缀。

### B. GPU 预检与一个小样本

下面只展示**已有工具的可选完整 selector smoke**，需要匹配 FlashInfer；它检查输入生成与索引等价并产生小规模计时，不输出本文的归因统计。必须纳入总预算，不能作为额外的 GPU 窗口。

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv \
  > "$KEYE_DIAG_OUTPUT/gpu-before.csv"

CUDA_VISIBLE_DEVICES="$KEYE_DIAG_GPU" OMP_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=10s 120s \
  python examples/models/keye-vl2/benchmark_dosa.py \
  --checkpoint "$KEYE_CHECKPOINT" --text "$KEYE_DIAG_INPUT" \
  --length 8192 --rows 128 --heads 4 --seed-size 4096 --repeats 1 \
  --output "$KEYE_DIAG_OUTPUT/selector-smoke.json" \
  > "$KEYE_DIAG_OUTPUT/selector-smoke.log" 2>&1
```

120 秒是该子步骤的上限，不是运行时间预测。外层仍须管理累计 900 秒预算和自己的进程组清理。若只执行独立统计驱动，可在驱动内直接做原 scorer、seed 与 bound 一致性预检，不强行运行完整 selector smoke。

### C. 运行诊断与退出

诊断驱动实现后，将其实际命令补入执行记录；不要把未实现的 CLI 选项当作现有接口。运行剩余矩阵前计算总预算剩余时间，用外层 timeout 限制该驱动，并给落盘/退出预留时间。

到期或全部完成后，确认自己的 GPU 子进程退出，保存 GPU after 状态，再在 CPU 上生成统计表和图片。未完成 case 写明 `not_run_time_budget`、`input_too_short` 或具体错误，不伪装成 0% 剪枝。

## 8. 报告与交付物

在持久目录保存环境清单、输入 manifest、每 case/per-query JSON 或 CSV、stdout/stderr、未完成 case 清单、静态 PNG/SVG 图以及 Markdown 报告。不要只放 `/tmp`；原始大张量如需保存也放持久盘并记录 SHA256。

报告至少包含下表，每个样本和长度一行，附逐 query 分布：

| 输入 / ID | Layer | N / Q | 当前阈值可剪率 | 实际剪枝率 | 全量阈值可剪率 | 阈值差 P50/P90 | 并列比例 | 异常 / 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 待测 | 0 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

推荐图表：按长度展示“当前阈值可剪率 vs 实际剪枝率”，按输入展示 seed/full 阈值差分布，以及 `U-s` 和分数边界并列比例。图中注明样本数、层号和 GPU，不用一条总体均值掩盖层内/输入间差异。

归因按以下证据判断：

| 观察 | 支持的解释 | 下一步 |
| --- | --- | --- |
| 全量阈值下也少有严格低分项，且边界并列多 | 输入/分数分布限制 score-only 剪枝 | 比较自然长 context 与重复对照 |
| 全量阈值可剪率高，seed 阈值可剪率明显低 | seed 阈值过弱 | 保持上界不变，单独比较 seed 策略 |
| seed 阈值可剪率高，实际上界剪枝率接近零 | 上界余量是直接瓶颈 | 分析正残差、负聚合、数值保护项，再做单变量消融 |
| 出现 `U < s` 或 seed/full 阈值顺序错误 | 正确性或诊断映射错误 | 停止归因，先修复验证 |

第一层首轮报告不得推断全部 48 层，也不能宣称已经复现论文剪枝率。若需要解释历史第 23/47 层，优先恢复旧捕获；无法恢复时，单独申请真实深层采集时间。更好的剪枝率也不自动等于 selector 加速，紧凑写回和最终选择成本需要后续计时验证。

## 9. 给执行者的交接指令

> 在 `feat/fengxiaoshi/DoSA` 上按本文准备 R=4 归因统计。先核对模型、输入和环境，实现尚缺的诊断驱动；不能拿原 benchmark 的时间 JSON 冒充三个归因指标。保持原始 BF16 scorer、seed 规则和上界数值保护不变。CPU 准备完成后，仅使用一张已确认空闲的卡，首轮累计 GPU 窗口最多 15 分钟，逐 case 保存并在到期后退出。输出统计报告及原始行级结果，区分输入/seed 阈值/上界损失；如未完成或仅覆盖 layer 0，明确写出边界。不要启动训练、服务或修改已有实验。

相关记录：[实现与历史测试](./keye-dosa-v1.md)、[历史计时 JSON](./keye-dosa-v2-results.json)、[DoSA 论文](https://arxiv.org/html/2610.13990v1)。

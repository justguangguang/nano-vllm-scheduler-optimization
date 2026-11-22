# nano-vLLM Scheduler Optimization

**面向大语言模型推理的 Scheduler 调度优化与可复现实验**

本项目基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 进行研究性扩展，针对 FIFO 等待队列的 **Head-of-Line（HOL）Blocking**、Prefill Token Budget 利用不足，以及请求绕行可能造成的 **KV Cache 资源竞争**，迭代实现三种调度策略：**Queue Bypass（v1）**、**Bounded Candidate Scanning（v2）** 和 **Decode-Aware KV Cache Admission（v3）**。

> 项目定位：聚焦 Scheduler 与 KV Cache 准入策略的离线实验，不是完整的在线推理服务，也不是 nano-vLLM 官方版本。所有性能结论均限定于下述实验负载和环境。

## 1. 核心结果

基于 **RTX 3090 24GB + Qwen3-0.6B**，在相同环境下分别执行 Baseline / v2 / v3，每个场景测量 **10 次**，比较吞吐量中位数及请求延迟：

| 场景 | 对比版本 | 关键结果 | 说明 |
|---|---|---|---|
| **E1：Token Budget 队头阻塞** | Baseline → v3 | **输出吞吐量 +17.79%**；Short TTFT P95 **-23.64%** | Prefill 调度轮数由 48 降至 36 |
| **E2：受限 KV Cache 准入** | v2 → v3 | **输出吞吐量 +2.06%**；Preemptions **1 → 0** | 避免 47 个重复 Prefill Token |
| **A/B/C/D：常规离线负载** | Baseline → v3 | 10 个场景吞吐量变化 **-0.58%～+0.26%** | 整体基本持平，并非普遍提升 |

**关键权衡：** E2 中 v3 相比**原始 Baseline** 的吞吐量仍下降 **8.26%**（545.419 → 500.361 tok/s）；Short 请求显著提前执行，但 Medium 与已有 Running 请求的延迟有所增加。因此，不能把 E2 描述为“所有请求吞吐和延迟同时提升”。

- [完整复现指南与脚本](benchmarks/README.md)
- [Baseline 常规 Benchmark 报告](docs/experiments/benchmark_baseline.md) · [v3 常规 Benchmark 报告](docs/experiments/benchmark_v3.md)
- [E1 对比记录](docs/experiments/e1_comparison.txt) · [E2 对比记录](docs/experiments/e2_comparison.txt)

## 2. 问题背景与优化设计

### 2.1 为什么会发生队头阻塞？

原始 nano-vLLM 采用以 FIFO 为基础的 Waiting Queue 调度。在 Prefill Token Budget 受限时，若队首请求无法装入本轮剩余预算，调度可能停止，尽管队列后面仍存在可以执行的短请求。

例如，设每轮 Prefill Token Budget 为 **1024**，等待队列依次包含 **Long(768) → Medium(512) → Short(256)**：

```text
原始 FIFO：  选择 Long(768) → 剩余 256 → Medium(512) 不适配 → 停止
Queue Bypass：选择 Long(768) → 跳过 Medium(512) → 选择 Short(256)

本轮 Prefill Token Budget 使用量：768/1024 → 1024/1024
```

这种绕行能够改善特定负载下的 Token Budget 利用率，但也可能增大候选搜索开销，并挤占正在 Decode 的请求所需的 KV Block。

### 2.2 三个迭代版本

| 版本 | 核心方法 | 解决的问题 |
|---|---|---|
| **v1 · Queue Bypass** | 跳过本轮暂不可执行的候选，继续尝试后续请求 | 缓解 Token Budget / KV 条件造成的队头阻塞 |
| **v2 · Bounded Candidate Scanning** | 限制额外扫描候选数量，结合保守的固定 KV Block 预留 | 控制绕行搜索成本与缓存竞争风险 |
| **v3 · Decode-Aware KV Cache Admission** | 根据 Running 请求下一次 KV Block 扩展需求估计 Decode Headroom，动态限制绕行准入 | 降低新 Prefill 对后续 Decode KV 空间的挤占与抢占风险 |

**实现边界：** v3 主要保护 **KV Cache 空间**，不等于实现了 Prefill/Decode 同批混合执行、Decode 优先调度或 P/D 分离。该项目不宣称解决所有 Prefill 对 Decode 的执行时间干扰。

## 3. 实验设置与结果

### 3.1 实验环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090，24GB，单卡 |
| 模型 | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |
| Python / PyTorch | Python 3.12 / PyTorch 2.5.1+cu124 |
| CUDA Runtime | 12.4 |
| Transformers / FlashAttention | 4.51.3 / 2.7.4.post1 |
| 随机种子 / 重复次数 | `20261107` / 每场景 10 次 |
| 调度器版本 | 原始 Baseline、v2、v3 独立 Git 分支 |

下表均使用**新 AutoDL 环境**的测试结果，未与旧服务器数据混合计算。具体参数、脚本和输出目录详见 [Benchmark 复现文档](benchmarks/README.md)。

### 3.2 E1：Token Budget 队头阻塞

E1 构造 **24 组 Long(768) / Medium(512) / Short(256)**，共 **72 条请求**；Prefill Token Budget 为 **1024**，每条请求生成 **64 tokens**，用于隔离验证 Queue Bypass 在队头阻塞场景下的效果。

| 指标（10 次测量中位数） | Baseline | v3 | 变化 |
|---|---:|---:|---:|
| 输出吞吐量（tok/s） | 2163.390 | 2548.267 | **+17.79%** |
| Prefill Steps | 48 | 36 | **-25.00%** |
| Short TTFT P95（ms） | 1352.991 | 1033.178 | **-23.64%** |
| Short E2E P95（ms） | 2129.890 | 1808.191 | **-15.10%** |
| 抢占次数 | 0 | 0 | 无变化 |

首轮调度轨迹显示：Baseline 只选择 `['long']`，v3 选择 `['long', 'short']`；v3 在此受限预算场景中更充分地利用了本轮 Prefill 容量。Long、Medium、Short 请求的 TTFT P95 在该负载下均下降，但这一结果**不能外推至任意请求分布或在线到达流量**。

### 3.3 E2：Decode-Aware KV Cache 准入

E2 是受控资源压力实验：将 BlockManager **可访问的逻辑 KV Block 限制为 7 个**，以区分不同绕行准入策略。**实际 GPU KV Cache Tensor 并未缩小到 7 个 Block**，因此不能把这一场景描述为默认显存容量下的生产部署。

| 指标（10 次测量中位数） | Baseline | v2 | v3 |
|---|---:|---:|---:|
| 输出吞吐量（tok/s） | **545.419** | 490.281 | 500.361 |
| Prefill Steps | **2** | 3 | 3 |
| Preemptions | **0** | 1 | **0** |
| 重复 Prefill Tokens | **0** | 47 | **0** |
| Short TTFT（ms） | 142.757 | 174.952 | **31.178** |
| Medium TTFT（ms） | 142.764 | **30.914** | 175.852 |
| Owner E2E 均值（ms） | **142.857** | 174.704 | 176.279 |

首次注入晚到请求后的调度选择：

```text
Baseline: decode  ['owner_a', 'owner_b']  → preemptions = 0
v2:       prefill ['medium']              → preemptions = 1
v3:       prefill ['short']               → preemptions = 0
```

**v3 相对 v2**：吞吐量 **+2.06%**，在这个固定场景中避免 **1 次 Medium 请求抢占**和 **47 个重复 Prefill Token**。已有 Running Owner 请求在两版中都没有被抢占，不能误称为“Owner 抢占由 1 降至 0”。

**v3 相对 Baseline**：整体吞吐量 **-8.26%**，Short 请求 TTFT 更低，但 Medium TTFT 和 Owner E2E 更高，体现了调度准入与不同请求类别之间的性能权衡。

### 3.4 常规 Benchmark：A/B/C/D

A/B/C/D 覆盖随机输入输出长度、不同 Batch Size、长输入以及混合请求。下表为输出吞吐量中位数（tok/s）：

| 场景 | Baseline | v3 | 变化 |
|---|---:|---:|---:|
| A · Random Length | 4582.76 | 4587.18 | +0.10% |
| B1 | 313.09 | 312.82 | -0.09% |
| B16 | 4020.24 | 4015.30 | -0.12% |
| B64 | 9065.97 | 9071.76 | +0.06% |
| B128 | 11020.74 | 10956.35 | -0.58% |
| B256 | 12004.31 | 12036.08 | +0.26% |
| C512 | 4634.08 | 4641.80 | +0.17% |
| C1024 | 2758.12 | 2751.73 | -0.23% |
| C2048 | 1477.02 | 1480.72 | +0.25% |
| D · Mixed Requests | 3192.02 | 3195.44 | +0.11% |

多数变化幅度较小，**不能据此声称 v3 普遍加速**。另外，D 采用 Scheduler Step 级插桩，测量口径与 A/B/C 不完全相同，不应将不同场景的吞吐量直接横向对比。

## 4. 代码结构与版本追溯

```text
nano-vllm-scheduler-optimization/
├── nanovllm/engine/
│   ├── scheduler.py                 # 调度策略实现（随 Git 分支变化）
│   └── block_manager.py             # KV Block 管理
├── benchmarks/
│   ├── README.md                    # 完整运行与复现说明
│   ├── patches/
│   │   └── transformers_dtype_compat.patch
│   └── phase1/                     # A/B/C/D、E1、E2、对比和诊断脚本
├── docs/
│   ├── phase1_results.md            # 历史阶段性结果
│   └── experiments/                # 本次 AutoDL 汇总及对比记录
├── README.md
└── LICENSE
```

| 分支 | 版本说明 |
|---|---|
| [`phase1/baseline`](https://github.com/justguangguang/nano-vllm-scheduler-optimization/tree/phase1/baseline) | 原始 FIFO Scheduler |
| [`feat/queue-bypass-v1`](https://github.com/justguangguang/nano-vllm-scheduler-optimization/tree/feat/queue-bypass-v1) | Queue Bypass |
| [`feat/queue-bypass-v2-guarded`](https://github.com/justguangguang/nano-vllm-scheduler-optimization/tree/feat/queue-bypass-v2-guarded) | Bounded Scan + 固定 KV 预留 |
| [`feat/queue-bypass-v3-decode-headroom`](https://github.com/justguangguang/nano-vllm-scheduler-optimization/tree/feat/queue-bypass-v3-decode-headroom) | 动态 Decode Headroom；第一阶段最终版 |

实验对应的 Scheduler 源文件校验前缀：Baseline `05cc3479307d`、v2 `d2eb95025dd8`、v3 `b7d1872ab99d`（SHA-256 前 12 位）。当前仓库主分支用于展示最终方案和实验工具，不影响历史分支的调度策略对照。

## 5. 快速开始与复现

模型权重不包含在仓库中。准备可运行的 nano-vLLM Python 环境，并从项目根目录执行：

```bash
export CUDA_VISIBLE_DEVICES=0
export NANO_VLLM_MODEL="/absolute/path/to/Qwen3-0.6B"

python -m pip install -e . --no-deps

# 提前复制测试脚本到 Git 忽略的 results/，以便跨历史分支复用同一版本
mkdir -p results/phase1_tools
cp benchmarks/phase1/*.py results/phase1_tools/

# 历史代码与 Transformers 4.51.3 的 dtype 兼容处理；仅在尚未应用时执行一次
git apply benchmarks/patches/transformers_dtype_compat.patch

# 切换原始 Scheduler 并运行一个示例（其他版本和实验命令见复现指南）
git switch phase1/baseline
python -u results/phase1_tools/bench_budget_hol.py \
  --model "$NANO_VLLM_MODEL" \
  --output-dir results/reproduce_e1_baseline \
  --repeats 10 --seed 20261107 \
  --triplets 24 --budget 1024 --output-tokens 64 --max-model-len 4096
```

**注意：** 上述 `pip install -e . --no-deps` 假设依赖已安装；完整环境、各分支切换、全部测试命令及注意事项请以 [benchmarks/README.md](benchmarks/README.md) 为准。如果 dtype 修复已存在，请勿重复执行 `git apply`。输出目录应使用未存在的新路径，避免覆盖实验结果。

## 6. 局限性与后续工作

- **场景限定：** E1、E2 是专门构造的离线负载；尚未评估生产级请求到达率、在线 P95/P99 延迟和端到端 SLO。
- **E2 资源约束：** 限制逻辑 Block 分配，并非直接压缩物理 GPU KV Cache 容量。
- **调度权衡：** 避免抢占不保证整体吞吐量优于 FIFO，也不保证所有请求的 TTFT、TPOT 和 E2E 同时改善。
- **指标口径：** 插桩得到的 TTFT/E2E 与估计 TPOT 不等同于生产服务逐 Token 的精确 ITL 或在线 SLO。
- **后续方向：** 探索面向动态到达请求的 Prefill/Decode 平衡调度、公平性与 SLO 感知策略；**这些功能尚未实现**。

## 7. 上游项目与许可证

本仓库基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 二次开发。原项目作者为 **Xingkai Yu 和其他贡献者**；原始推理引擎的相关实现及版权归原作者和贡献者所有。本项目新增的内容主要是 Scheduler 调度策略研究、实验脚本与性能分析。

遵循上游 **MIT License**，保留原项目版权信息与仓库根目录中的 [`LICENSE`](LICENSE)。本项目为独立研究扩展，**不代表上游官方项目**。

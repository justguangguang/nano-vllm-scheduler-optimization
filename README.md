# nano-vLLM Scheduler Optimization

基于 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的大语言模型推理调度优化项目，重点研究 **Head-of-Line Blocking（队头阻塞）**、**Prefill Token Budget 利用率**与 **KV Cache 资源竞争**。

本项目在保留原始推理引擎架构的基础上，对 `Scheduler` 进行扩展，实现 **Queue Bypass**、**Bounded Candidate Scanning** 和 **Decode-Aware KV Cache Admission**，并通过常规负载、专项压力测试及消融实验评估策略效果。

## 1. 项目背景

原始 nano-vLLM 采用以 FIFO 为基础、Prefill 优先的调度策略。在部分资源受限负载下，Waiting Queue 存在队头阻塞（Head-of-Line Blocking）：后续可执行请求无法被及时选中，导致剩余 Prefill Token Budget 未得到充分利用。

直接引入请求绕行虽然能够缓解队头阻塞，但可能增加无效 KV Block 分配检查、KV Cache 资源竞争、请求抢占（Preemption）和重复 Prefill 计算。本项目针对上述问题，迭代实现三版调度策略。

## 2. 核心优化

### 2.1 Queue Bypass（v1）

允许跳过暂时无法调度的队首请求，继续检查后续满足 Token Budget 或 KV Cache 条件的请求，提高特定队头阻塞场景下的 Prefill 批次资源利用效率。

### 2.2 Bounded Candidate Scanning（v2）

对队头阻塞后的额外候选扫描设置上限，并引入保守的 KV Block 预留，减少无效 `can_allocate()` 检查和过度资源竞争。该策略用于修正 v1 在常规随机负载下出现的吞吐量回退。

### 2.3 Decode-Aware KV Cache Admission（v3）

根据 Running 请求距下一次 KV Block 扩展的距离，估算近期 Decode 缓存需求，在绕行准入时动态预留空闲 Block，降低新 Prefill 请求过度占用缓存、引发额外抢占与重计算的风险。

该机制主要保护 **KV Cache 空间**，并未实现 Prefill/Decode 同批执行或 Decode 优先调度。

## 3. 实验结果

**原实验环境：** NVIDIA RTX 3090 24GB、Qwen3-0.6B、PyTorch 2.6.0+cu126、CUDA Runtime 12.6。

> **说明：** 以下数据来自原实验环境。E1 和 E2 使用不同的专门构造的工作负载，不能将两组吞吐量直接横向比较；新服务器的运行结果需要重新测量。

### 3.1 E1：Token Budget 队头阻塞（Baseline vs v3）

E1 采用 **72 条混合长度请求**，Prefill Token Budget 设置为 **1024**，专门验证 Queue Bypass 对队头阻塞的改善效果。下表为三轮测试的中位数结果。

| 评价指标 | 原始 Baseline | Queue Bypass v3 | 变化 |
|---|---:|---:|---:|
| 输出吞吐量（tok/s） | 2053.85 | 2471.89 | **+20.35%** |
| Prefill 调度轮数 | 48 | 36 | **-25.00%** |
| Short TTFT P95（ms） | 1468.35 | 1092.16 | **-25.62%** |

在 E1 场景下，原始 FIFO 在剩余 Token Budget 无法容纳下一条请求时停止选择；v3 可以绕过该请求，接纳后续更短的请求，减少 Prefill 调度轮数。

### 3.2 E2：KV Cache 准入消融（v2 vs v3）

E2 人工限制调度器可分配的 KV Block 数量，构造 KV Cache 资源竞争场景，用于隔离验证 **Decode Headroom 相比固定 Block 预留策略**的作用。下表为五轮测试的中位数结果。

| 评价指标 | v2：固定 Block 预留 | v3：动态 Decode Headroom | 变化 |
|---|---:|---:|---:|
| 输出吞吐量（tok/s） | 491.72 | 514.44 | **+4.62%** |
| Preemption 次数 | 1 | 0 | **减少 1 次** |
| 额外 Prefill Token | 47 | 0 | **减少 47 Token** |

v3 在此场景中改变了绕行候选的准入顺序，避免了 v2 中一次 **Medium 请求**的抢占；不能将其表述为已有 Running Owner 请求的抢占次数下降，因为 Owner 在两个版本中均未发生抢占。

**必要的性能权衡：** 与原始 Baseline 相比，E2 中 v3 的输出吞吐量从 **546.47 降至 514.44 tok/s（-5.86%）**，已有 Owner 请求的 TPOT 估计与端到端延迟也变差。v3 虽然保护了 KV Cache 空间，但仍保留 Prefill 优先执行路径，无法消除 Prefill 对 Decode 的执行时间干扰。

### 3.3 常规 Benchmark：A/B/C/D

常规测试覆盖随机输入/输出长度、不同 Batch Size、长输入以及混合请求。最终 v3 与原始 Baseline 的 **10 个场景输出吞吐量变化均在约 ±0.6% 内**，整体可视为基本持平，而非普遍吞吐量提升。

更完整的数值可查看 [`docs/phase1_results.md`](docs/phase1_results.md)（若该文档未包含某项指标，以对应实验日志为准）。

## 4. 版本演进

| 版本 / 分支 | 核心修改 |
|---|---|
| [`phase1/baseline`](https://github.com/justguangguang/nano-vllm-scheduler-optimization/tree/phase1/baseline) | 原始 nano-vLLM FIFO 调度基线 |
| [`feat/queue-bypass-v1`](https://github.com/justguangguang/nano-vllm-scheduler-optimization/tree/feat/queue-bypass-v1) | Queue Bypass，缓解队头阻塞 |
| [`feat/queue-bypass-v2-guarded`](https://github.com/justguangguang/nano-vllm-scheduler-optimization/tree/feat/queue-bypass-v2-guarded) | Bounded Scan + 固定 KV Block 预留 |
| [`feat/queue-bypass-v3-decode-headroom`](https://github.com/justguangguang/nano-vllm-scheduler-optimization/tree/feat/queue-bypass-v3-decode-headroom) | 动态 Decode Headroom，改善绕行准入 |

以上阶段通过独立 Git 分支保留，方便比较调度策略的变化及实验结果。

## 5. 复现说明

当前仓库已公开 Scheduler 源码与实验结果汇总；**Benchmark 脚本尚未提交到仓库**，因此此处暂不提供未经验证的“一键复现”命令。计划在整理、去除服务器专用绝对路径并完成新环境回归测试后，提交 A/B/C/D、E1 和 E2 测试脚本与运行说明。

模型：[`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B)。模型权重不包含在本仓库中。

## 6. 后续研究

计划研究 **Prefill/Decode Balanced Scheduling**，探索在动态请求到达场景下兼顾 TTFT、TPOT、吞吐量和调度公平性的策略。**该功能尚未实现。**

## 7. 上游项目与开源许可

本项目基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 二次开发。

- **Original project:** nano-vLLM
- **Original author:** Xingkai Yu and contributors
- **Original license:** MIT License
- **This repository:** 在原始框架基础上新增 Scheduler 调度优化及实验分析

本仓库保留原始项目的版权声明和 [`LICENSE`](LICENSE) 中的 MIT 许可条款。原始推理引擎的实现归原作者及贡献者所有，本项目不是 nano-vLLM 官方版本。

# nano-vLLM Scheduler Optimization

基于 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的大语言模型推理调度优化项目，重点研究 **Head-of-Line Blocking、Prefill Token Budget 利用率与 KV Cache 资源竞争**。

本项目在保留原始推理引擎架构的基础上，对 Scheduler 进行扩展，实现 Queue Bypass、Bounded Candidate Scanning 和 Decode-Aware KV Cache Admission，并构建性能测试与消融实验。

## 1. 项目背景

原始 nano-vLLM 的 FIFO Prefill 调度在部分资源受限场景下存在队头阻塞，导致后续可执行请求无法充分利用剩余 Token Budget。

直接引入请求绕行虽然能够改善队头阻塞，但可能增加 KV Cache 资源竞争、无效分配检查、请求抢占和重复 Prefill 开销。

本项目围绕上述问题进行三轮调度策略优化。

## 2. 核心优化

### Queue Bypass

允许跳过暂时无法调度的队首请求，选择后续满足资源条件的请求，提高 Prefill Batch 的 Token Budget 利用率。

### Bounded Candidate Scanning

限制队头阻塞后的额外候选扫描数量，降低无效 KV Cache 分配检查带来的 CPU 调度开销。

### Decode-Aware KV Cache Admission

根据 Running 请求距离下一次 KV Block 扩展的距离，估算近期 Decode 缓存需求，动态预留空闲 Block，避免绕行请求过度占用缓存资源。

## 3. 实验结果

**测试环境：** RTX 3090 24GB、Qwen3-0.6B、PyTorch 2.6.0、CUDA 12.6。

| 测试场景 | Baseline | 优化版本 v3 | 结果 |
|---|---:|---:|---|
| E1 输出吞吐量 | 2053.85 tok/s | 2471.89 tok/s | +20.35% |
| E1 Prefill 轮数 | 48 | 36 | -25% |
| E1 Short TTFT P95 | 1468.35 ms | 1092.16 ms | -25.62% |
| E2 Preemption（对比 v2） | 1 次 | 0 次 | 避免 1 次抢占 |
| E2 重复 Prefill Token（对比 v2） | 47 | 0 | 减少 47 Token |

E1 是 72 条混合长度请求、Prefill Token Budget 为 1024 的专项测试；E2 是人工限制 KV Block 可分配容量的资源压力测试。

在完整常规 Benchmark 中，v3 的吞吐量与原始 nano-vLLM 基本持平。E2 相比原始版仍存在总体吞吐量和部分 Decode 延迟回退，当前方案并不保证对所有负载均有收益。

## 4. 版本演进

| 版本 | 核心修改 |
|---|---|
| Baseline | 原始 nano-vLLM FIFO 调度 |
| v1 | Queue Bypass |
| v2 | Bounded Scan + 固定 KV Block 预留 |
| v3 | Decode-Aware Dynamic KV Headroom |

各阶段保留独立 Git 分支，用于功能回归及消融实验。

## 5. 后续研究

后续计划研究 Prefill/Decode Balanced Scheduling，探索在动态请求到达场景下改善 TTFT、TPOT、吞吐量与请求公平性的调度策略。该功能尚未实现。

## 6. 上游项目与开源许可

本项目基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 二次开发。

- Original project: nano-vLLM
- Original author: Xingkai Yu and contributors
- Original license: MIT License
- Scheduler optimizations and benchmark extensions: This repository

保留原始项目的版权声明和 MIT License。本仓库的调度策略修改与实验工作属于在原始项目基础上的扩展，并非 nano-vLLM 官方版本。
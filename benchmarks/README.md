# Phase 1 Benchmark Reproduction

Benchmark and diagnostic tools for [nano-vLLM Scheduler Optimization](https://github.com/justguangguang/nano-vllm-scheduler-optimization), a research extension of [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

This directory contains **A/B/C/D** offline benchmarks, **E1** token-budget head-of-line (HOL) blocking tests, **E2** constrained KV-cache admission tests, and comparison/diagnostic tools. All experiments compare separate Scheduler revisions while holding the model, software environment, workload, and parameters constant.

## 1. Environment

The recorded AutoDL experiments (November 22, 2026) used:

| Component | Configuration |
|---|---|
| GPU | NVIDIA RTX 3090, 24 GB, single GPU |
| Model | Qwen3-0.6B |
| Python | 3.12 |
| PyTorch | 2.5.1+cu124 |
| CUDA runtime (PyTorch) | 12.4 |
| FlashAttention | 2.7.4.post1 |
| Transformers | 4.51.3 |
| Visible GPU | `CUDA_VISIBLE_DEVICES=0` |

Other environments may produce different absolute performance. Use the same environment for all versions being compared. Prepare the nano-vLLM dependencies and model locally; the model weights are **not** included in this repository. For an environment with dependencies already installed, run from the repository root:

```bash
python -m pip install -e . --no-deps
export CUDA_VISIBLE_DEVICES=0
export NANO_VLLM_MODEL="/absolute/path/to/Qwen3-0.6B"
```

### ModelRunner dtype compatibility

The historical Scheduler revisions use a `model_runner.py` that accesses `hf_config.dtype`, while the tested Transformers/Qwen3 configuration exposes `hf_config.torch_dtype`. A narrowly scoped patch is provided at `benchmarks/patches/transformers_dtype_compat.patch`.

**Before switching Scheduler branches**, apply the patch from the repository root if it is not already applied:

```bash
git apply --check benchmarks/patches/transformers_dtype_compat.patch
git apply benchmarks/patches/transformers_dtype_compat.patch
```

To **check whether the patch is already applied**, without changing anything:

```bash
git apply --reverse --check benchmarks/patches/transformers_dtype_compat.patch
```

Do not apply the patch twice. This is an environment compatibility fix, **not** a Scheduler optimization. The tests below assume the same patch is present in every compared version.

## 2. Preserve the benchmark tools across Git branches

The benchmark scripts were added to the optimized project's `main` / v3 branch, but the historical Baseline and v2 branches do not contain this folder. While on the branch containing `benchmarks/`, copy them to a Git-ignored local directory **before switching branches**:

```bash
mkdir -p results/phase1_tools results/run_logs
cp benchmarks/phase1/*.py results/phase1_tools/
python -m compileall -q results/phase1_tools
```

The `results/` directory is ignored by Git and survives `git switch`. Run **all** versions with this same frozen copy of the benchmark scripts. Run commands from the repository root. The scripts require `CUDA_VISIBLE_DEVICES=0`.

If switching branches refuses because of uncommitted changes, inspect `git status` first; do not discard the dtype compatibility patch.

## 3. Full A/B/C/D benchmark

**A:** 256 requests with random input/output lengths. **B:** batch sizes 1, 16, 64, 128 and 256. **C:** input lengths 512, 1024 and 2048 (batch size 64). **D:** mixed-length request scheduling with step-level instrumentation. Each workload has a shape-matched warmup and **10** measured repeats; seed = `20261107`, max model length = `4096`.

**Baseline:**

```bash
git switch phase1/baseline
python -u results/phase1_tools/run_benchmark_v1.py \
  --model "$NANO_VLLM_MODEL" \
  --output-dir results/reproduce_baseline \
  --repeats 10 --seed 20261107 --max-model-len 4096
```

**v3:**

```bash
git switch feat/queue-bypass-v3-decode-headroom
python -u results/phase1_tools/run_benchmark_v1.py \
  --model "$NANO_VLLM_MODEL" \
  --output-dir results/reproduce_v3 \
  --repeats 10 --seed 20261107 --max-model-len 4096
```

Outputs include `summary/summary.md`, `summary/summary.json`, `summary/summary.csv`, `raw/trial_results.csv`, and `raw/D_request_records.csv`. The CSV writer combines fields from all trial types (including D's `scheduler_steps`).

**Measurement caveat:** A/B/C use `generate()` timing, whereas D uses Scheduler step instrumentation. Do not directly compare D throughput against A/B/C throughput. D's TTFT/E2E are observed at Scheduler-step boundaries; TPOT is an average estimate, not token-level inter-token latency (ITL).

## 4. E1 — token-budget HOL blocking

**Workload:** 24 repeating groups of Long (768 input tokens), Medium (512), and Short (256); 72 requests in total, 64 output tokens/request, Prefill token budget = **1024**. This controlled test checks whether Queue Bypass can use the token budget left over after selecting a Long request.

**Baseline:**

```bash
git switch phase1/baseline
python -u results/phase1_tools/bench_budget_hol.py \
  --model "$NANO_VLLM_MODEL" \
  --output-dir results/reproduce_e1_baseline \
  --repeats 10 --seed 20261107 \
  --triplets 24 --budget 1024 --output-tokens 64 --max-model-len 4096
```

**v3:**

```bash
git switch feat/queue-bypass-v3-decode-headroom
python -u results/phase1_tools/bench_budget_hol.py \
  --model "$NANO_VLLM_MODEL" \
  --output-dir results/reproduce_e1_v3 \
  --repeats 10 --seed 20261107 \
  --triplets 24 --budget 1024 --output-tokens 64 --max-model-len 4096
```

**Compare:**

```bash
python results/phase1_tools/compare_budget_hol.py \
  --main results/reproduce_e1_baseline --v3 results/reproduce_e1_v3
```

## 5. E2 — constrained KV-cache admission

**Workload:** Five named requests, two incumbent Running requests and three late requests (`head`, `medium`, `short`). The allocator's accessible metadata pool is constrained to **7 KV blocks** (256 tokens/block) to induce resource competition. This does **not** shrink the actual allocated GPU KV-cache tensor. The experiment contrasts FIFO, fixed reserve (v2), and decode-aware reserve (v3).

**Baseline:**

```bash
git switch phase1/baseline
python -u results/phase1_tools/bench_e2_kv.py \
  --model "$NANO_VLLM_MODEL" \
  --output-dir results/reproduce_e2_baseline \
  --repeats 10 --seed 20261107 --kv-blocks 7
```

**v2:**

```bash
git switch feat/queue-bypass-v2-guarded
python -u results/phase1_tools/bench_e2_kv.py \
  --model "$NANO_VLLM_MODEL" \
  --output-dir results/reproduce_e2_v2 \
  --repeats 10 --seed 20261107 --kv-blocks 7
```

**v3:**

```bash
git switch feat/queue-bypass-v3-decode-headroom
python -u results/phase1_tools/bench_e2_kv.py \
  --model "$NANO_VLLM_MODEL" \
  --output-dir results/reproduce_e2_v3 \
  --repeats 10 --seed 20261107 --kv-blocks 7
```

**Compare:**

```bash
python results/phase1_tools/compare_e2.py \
  --main results/reproduce_e2_baseline \
  --v2 results/reproduce_e2_v2 --v3 results/reproduce_e2_v3
```

## 6. Recorded results on AutoDL (10 repeats)

### Full A/B/C/D — v3 vs Baseline

| Scenario | Baseline tok/s | v3 tok/s | Change |
|---|---:|---:|---:|
| A | 4582.76 | 4587.18 | +0.10% |
| B1 | 313.09 | 312.82 | -0.09% |
| B16 | 4020.24 | 4015.30 | -0.12% |
| B64 | 9065.97 | 9071.76 | +0.06% |
| B128 | 11020.74 | 10956.35 | -0.58% |
| B256 | 12004.31 | 12036.08 | +0.26% |
| C512 | 4634.08 | 4641.80 | +0.17% |
| C1024 | 2758.12 | 2751.73 | -0.23% |
| C2048 | 1477.02 | 1480.72 | +0.25% |
| D | 3192.02 | 3195.44 | +0.11% |

### E1 — Baseline vs v3

| Metric | Baseline | v3 | Change |
|---|---:|---:|---:|
| Output throughput | 2163.390 tok/s | 2548.267 tok/s | +17.79% |
| Prefill steps | 48 | 36 | -25.00% |
| Short TTFT P95 | 1352.991 ms | 1033.178 ms | -23.64% |

The first Prefill batch is `['long']` in Baseline and `['long', 'short']` in v3, directly showing Queue Bypass under the 1024-token budget.

### E2 — Baseline, v2, and v3

| Metric | Baseline | v2 | v3 |
|---|---:|---:|---:|
| Output throughput (tok/s) | 545.419 | 490.281 | 500.361 |
| Preemptions | 0 | 1 | 0 |
| Redundant Prefill tokens | 0 | 47 | 0 |
| Short TTFT (ms) | 142.757 | 174.952 | 31.178 |
| Medium TTFT (ms) | 142.764 | 30.914 | 175.852 |

In this constrained test, **v3 vs v2** improved output throughput by **2.06%** and prevented the single medium-request preemption and 47 repeated Prefill tokens. **v3 vs Baseline** reduced output throughput by approximately **8.26%** and delayed some other requests. These are workload-specific trade-offs, not a universal speedup.

Detailed reports are available in `docs/experiments/` at repository root.

## 7. Reproducibility and limitations

- Run each version in a fresh Python process and use identical workload configuration.
- Confirm the Scheduler branch/SHA and verify that the dtype compatibility fix is present for each variant.
- Output directories must be new; do not overwrite recorded results.
- The benchmarks evaluate offline batches and controlled pressure workloads, **not** production traffic with online arrivals.
- The E2 seven-block constraint applies to allocator metadata, not actual VRAM capacity.
- TTFT/TPOT in instrumented tests should not be described as an online serving SLO.

**Attribution:** This project is derived from [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), originally authored by Xingkai Yu and contributors, under the MIT License. The original license and attribution are retained in the repository root `LICENSE`.

# nano-vLLM Benchmark v1.0 — Baseline Report

- Model: `/root/autodl-tmp/nano_vllm/models/Qwen3-0.6B`
- GPU: `Physical GPU 0` (logical device: `NVIDIA GeForce RTX 3090`)
- PyTorch: `2.5.1+cu124`; CUDA runtime: `12.4`
- Max model length: `4096`
- Repeats per case: `10`; warmups per case: `1`
- Sampling: temperature `0.6`, ignore_eos `True`
- Prompt policy: fixed length profile per case; fresh random token-ID content for warmup and every measured repeat to avoid reusing identical prefixes.

## A/B/C throughput summary

| Scenario | Label | Requests | Input tokens/run | Output tokens/run | Median output tok/s | Mean tok/s | CV | Median time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | random_length_256 | 256 | 146429 | 142226 | 4587.18 | 4588.38 | 0.430% | 31.0051 |
| B1 | batch_size_1 | 1 | 128 | 128 | 312.82 | 312.87 | 0.287% | 0.4092 |
| B16 | batch_size_16 | 16 | 2048 | 2048 | 4015.30 | 4017.20 | 0.188% | 0.5100 |
| B64 | batch_size_64 | 64 | 8192 | 8192 | 9071.76 | 9068.89 | 0.169% | 0.9030 |
| B128 | batch_size_128 | 128 | 16384 | 16384 | 10956.35 | 10961.79 | 0.540% | 1.4954 |
| B256 | batch_size_256 | 256 | 32768 | 32768 | 12036.08 | 12094.84 | 0.881% | 2.7225 |
| C512 | input_length_512_batch_64 | 64 | 32768 | 8192 | 4641.80 | 4643.48 | 0.218% | 1.7648 |
| C1024 | input_length_1024_batch_64 | 64 | 65536 | 8192 | 2751.73 | 2746.98 | 0.469% | 2.9770 |
| C2048 | input_length_2048_batch_64 | 64 | 131072 | 8192 | 1480.72 | 1480.71 | 0.048% | 5.5325 |

## D mixed-request scheduling summary

- Requests/run: 56; input tokens/run: 20480; output tokens/run: 6144
- Median output throughput: 3195.44 tok/s
- Mean output throughput: 3195.55 tok/s
- Throughput CV: 0.055%

| Request group | Count | TTFT P50 (ms) | TTFT P95 (ms) | E2E P50 (ms) | E2E P95 (ms) | TPOT estimate P50 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| short | 32 | 342.822 | 342.870 | 837.755 | 837.890 | 7.851 |
| medium | 16 | 272.197 | 342.893 | 1278.639 | 1278.734 | 7.928 |
| long | 8 | 272.192 | 272.235 | 1922.622 | 1922.672 | 6.472 |

### Measurement boundary

TTFT/E2E timestamps are sampled after each scheduler step and CUDA synchronization. TPOT is an estimated average interval from first-token step to finish step, not token-level ITL. This is an offline batch benchmark, not an online arrival benchmark.

## Metric definitions

- Output throughput = actual generated output tokens / measured wall time.
- Request throughput = completed requests / measured wall time.
- TTFT = first observed output-token step time − request enqueue time.
- E2E = observed finish-step time − request enqueue time.
- TPOT estimate = (finish-step time − first-output-step time) / (output tokens − 1).
- P50/P95 are calculated within each repeat; the displayed D group values are medians of the 10 repeat-level percentiles.
- CV = sample standard deviation / arithmetic mean × 100%.

## Important interpretation notes

- A, B and C use the generate() API; D uses step-level instrumentation. Do not compare D throughput directly with A/B/C throughput.
- D is an offline batch submission test. It does not represent sustained online arrivals or prove online scheduling fairness.
- TTFT/E2E are scheduler-step-boundary measurements; TPOT is an estimate, not exact per-token ITL.
- This report is valid only for this recorded source revision, hardware, model and runtime configuration.

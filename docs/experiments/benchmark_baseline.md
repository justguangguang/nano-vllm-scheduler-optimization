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
| A | random_length_256 | 256 | 146429 | 142226 | 4582.76 | 4587.15 | 0.242% | 31.0350 |
| B1 | batch_size_1 | 1 | 128 | 128 | 313.09 | 312.61 | 0.436% | 0.4088 |
| B16 | batch_size_16 | 16 | 2048 | 2048 | 4020.24 | 4023.27 | 0.195% | 0.5094 |
| B64 | batch_size_64 | 64 | 8192 | 8192 | 9065.97 | 9067.24 | 0.197% | 0.9036 |
| B128 | batch_size_128 | 128 | 16384 | 16384 | 11020.74 | 11018.52 | 0.074% | 1.4867 |
| B256 | batch_size_256 | 256 | 32768 | 32768 | 12004.31 | 12001.49 | 0.110% | 2.7297 |
| C512 | input_length_512_batch_64 | 64 | 32768 | 8192 | 4634.08 | 4634.19 | 0.057% | 1.7678 |
| C1024 | input_length_1024_batch_64 | 64 | 65536 | 8192 | 2758.12 | 2759.61 | 0.228% | 2.9701 |
| C2048 | input_length_2048_batch_64 | 64 | 131072 | 8192 | 1477.02 | 1469.31 | 1.047% | 5.5463 |

## D mixed-request scheduling summary

- Requests/run: 56; input tokens/run: 20480; output tokens/run: 6144
- Median output throughput: 3192.02 tok/s
- Mean output throughput: 3192.08 tok/s
- Throughput CV: 0.181%

| Request group | Count | TTFT P50 (ms) | TTFT P95 (ms) | E2E P50 (ms) | E2E P95 (ms) | TPOT estimate P50 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| short | 32 | 345.010 | 345.067 | 840.195 | 840.329 | 7.862 |
| medium | 16 | 269.128 | 345.077 | 1281.147 | 1281.231 | 7.969 |
| long | 8 | 269.123 | 269.164 | 1924.697 | 1924.736 | 6.492 |

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

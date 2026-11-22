#!/usr/bin/env python3
"""Unified, reproducible nano-vLLM benchmark suite (A/B/C/D).

Run from the nano-vllm project root. Results are written under
results/benchmark_v1/ with raw trial data and compact summaries.
"""

import argparse
import csv
import gc
import json
import math
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
from nanovllm import LLM, SamplingParams


DEFAULT_MODEL = os.environ.get("NANO_VLLM_MODEL")
DEFAULT_OUT = "results/benchmark_v1"
REPEATS = 10
WARMUPS = 1
BASE_SEED = 20261107
MAX_MODEL_LEN = 4096
TEMPERATURE = 0.6
IGNORE_EOS = True

D_GROUPS = [
    {"name": "short", "count": 32, "input_len": 128, "output_len": 64},
    {"name": "medium", "count": 16, "input_len": 512, "output_len": 128},
    {"name": "long", "count": 8, "input_len": 1024, "output_len": 256},
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", default=DEFAULT_OUT)
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument("--seed", type=int, default=BASE_SEED)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    return p.parse_args()


def make_prompts(input_lengths, seed):
    """Create valid token-ID prompts; content differs by warmup/trial seed."""
    rng = random.Random(seed)
    return [
        [rng.randint(100, 10000) for _ in range(length)]
        for length in input_lengths
    ]


def make_sampling_params(output_lengths):
    return [
        SamplingParams(
            temperature=TEMPERATURE,
            ignore_eos=IGNORE_EOS,
            max_tokens=length,
        )
        for length in output_lengths
    ]


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - pos) + values[hi] * (pos - lo))


def stats(values):
    vals = [float(x) for x in values]
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {
        "mean": round(mean, 4),
        "median": round(statistics.median(vals), 4),
        "std_sample": round(sd, 4),
        "cv_percent": round(sd / mean * 100, 4) if mean else 0.0,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(list((fields) or []) + [key for row in rows for key in row.keys()])))
        writer.writeheader()
        writer.writerows(rows)


def run_generate_case(llm, case_id, label, input_lengths, output_lengths,
                      seed, repeats, trial_rows):
    """One shape-matched warmup, then N measured generate() trials."""
    assert len(input_lengths) == len(output_lengths)
    expected_output = sum(output_lengths)
    input_tokens = sum(input_lengths)
    nreq = len(input_lengths)

    # Shape-matched warmup with a content seed not reused by measured trials.
    warm_seed = seed + 900_000_000
    warm_prompts = make_prompts(input_lengths, warm_seed)
    warm_params = make_sampling_params(output_lengths)
    torch.manual_seed(warm_seed)
    torch.cuda.manual_seed_all(warm_seed)
    warm_outputs = llm.generate(warm_prompts, warm_params, use_tqdm=False)
    warm_count = sum(len(x["token_ids"]) for x in warm_outputs)
    if warm_count != expected_output:
        raise RuntimeError(
            f"{case_id} warmup token mismatch: expected {expected_output}, got {warm_count}"
        )

    case_rows = []
    for run_idx in range(1, repeats + 1):
        # Same length profile in every repeat, different token contents.
        trial_seed = seed + run_idx * 1_000_003
        prompts = make_prompts(input_lengths, trial_seed)
        sampling = make_sampling_params(output_lengths)
        torch.manual_seed(trial_seed)
        torch.cuda.manual_seed_all(trial_seed)

        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        actual_output = sum(len(x["token_ids"]) for x in outputs)
        if actual_output != expected_output:
            raise RuntimeError(
                f"{case_id} run {run_idx} output mismatch: "
                f"expected {expected_output}, got {actual_output}"
            )

        row = {
            "scenario": case_id,
            "label": label,
            "run": run_idx,
            "request_count": nreq,
            "input_tokens": input_tokens,
            "output_tokens": actual_output,
            "time_seconds": round(elapsed, 6),
            "output_throughput_tok_s": round(actual_output / elapsed, 4),
            "request_throughput_req_s": round(nreq / elapsed, 6),
            "prompt_content_seed": trial_seed,
        }
        case_rows.append(row)
        trial_rows.append(row)
        print(
            f"[{case_id} {run_idx:02d}/{repeats}] "
            f"{row['output_throughput_tok_s']:.2f} tok/s | "
            f"{elapsed:.4f} s | {nreq / elapsed:.3f} req/s",
            flush=True,
        )

    summary = {
        "scenario": case_id,
        "label": label,
        "request_count": nreq,
        "input_tokens_per_run": input_tokens,
        "output_tokens_per_run": expected_output,
        "warmup_runs": WARMUPS,
        "repeats": repeats,
        "output_throughput_tok_s": stats([r["output_throughput_tok_s"] for r in case_rows]),
        "request_throughput_req_s": stats([r["request_throughput_req_s"] for r in case_rows]),
        "time_seconds": stats([r["time_seconds"] for r in case_rows]),
        "runs": case_rows,
    }
    print(
        f"[{case_id} SUMMARY] median throughput="
        f"{summary['output_throughput_tok_s']['median']:.2f} tok/s; "
        f"CV={summary['output_throughput_tok_s']['cv_percent']:.3f}%",
        flush=True,
    )
    return summary


def make_d_workload(seed):
    rng = random.Random(seed)
    workload = []
    for idx in range(max(g["count"] for g in D_GROUPS)):
        for group in D_GROUPS:
            if idx >= group["count"]:
                continue
            workload.append({
                "group": group["name"],
                "request_id": f"{group['name']}_{idx:02d}",
                "input_len": group["input_len"],
                "output_len": group["output_len"],
                "prompt_ids": [rng.randint(100, 10000) for _ in range(group["input_len"])],
                "sampling_params": SamplingParams(
                    temperature=TEMPERATURE,
                    ignore_eos=IGNORE_EOS,
                    max_tokens=group["output_len"],
                ),
            })
    return workload


def latency_summary(rows):
    ttft = [r["ttft_ms"] for r in rows]
    e2e = [r["e2e_ms"] for r in rows]
    tpot = [r["tpot_est_ms"] for r in rows]
    return {
        "request_count": len(rows),
        "ttft_p50_ms": round(percentile(ttft, 0.50), 3),
        "ttft_p95_ms": round(percentile(ttft, 0.95), 3),
        "e2e_p50_ms": round(percentile(e2e, 0.50), 3),
        "e2e_p95_ms": round(percentile(e2e, 0.95), 3),
        "tpot_est_p50_ms": round(percentile(tpot, 0.50), 3),
        "tpot_est_p95_ms": round(percentile(tpot, 0.95), 3),
        "e2e_max_ms": round(max(e2e), 3),
    }


def run_d_case(llm, seed, repeats, trial_rows, request_rows):
    total_input = sum(g["count"] * g["input_len"] for g in D_GROUPS)
    total_output = sum(g["count"] * g["output_len"] for g in D_GROUPS)
    total_requests = sum(g["count"] for g in D_GROUPS)
    token_budget = llm.scheduler.max_num_batched_tokens

    if total_input <= token_budget:
        raise RuntimeError(
            f"D workload input tokens ({total_input}) do not exceed "
            f"max_num_batched_tokens ({token_budget}); inspect scheduler config."
        )

    # Shape-matched warmup with unrelated prompt contents.
    warmup = make_d_workload(seed + 900_000_001)
    warm_outputs = llm.generate(
        [x["prompt_ids"] for x in warmup],
        [x["sampling_params"] for x in warmup],
        use_tqdm=False,
    )
    warm_count = sum(len(x["token_ids"]) for x in warm_outputs)
    if warm_count != total_output:
        raise RuntimeError(f"D warmup output mismatch: {warm_count} != {total_output}")

    # Wrap step only to observe sequence-level first-token/finish transitions.
    state = {"records": {}}
    original_step = llm.step

    def tracked_step():
        result = original_step()
        # Keep identical step-boundary synchronization for original/optimized runs.
        torch.cuda.synchronize()
        now = time.perf_counter()
        for rec in state["records"].values():
            seq = rec["seq"]
            n = seq.num_completion_tokens
            if n > rec["observed_tokens"]:
                if rec["first_token_time"] is None:
                    rec["first_token_time"] = now
                rec["observed_tokens"] = n
            if seq.is_finished and rec["finish_time"] is None:
                rec["finish_time"] = now
        return result

    llm.step = tracked_step
    run_summaries = []

    for run_idx in range(1, repeats + 1):
        trial_seed = seed + run_idx * 1_000_003
        workload = make_d_workload(trial_seed)
        records = {}
        state["records"] = records

        torch.manual_seed(trial_seed)
        torch.cuda.manual_seed_all(trial_seed)
        torch.cuda.synchronize()
        run_start = time.perf_counter()

        # Submit all requests as one offline mixed batch; arrival times are
        # recorded individually at the add_request boundary.
        for item in workload:
            arrival = time.perf_counter()
            llm.add_request(item["prompt_ids"], item["sampling_params"])
            seq = llm.scheduler.waiting[-1]
            records[seq.seq_id] = {
                "seq": seq,
                "request_id": item["request_id"],
                "group": item["group"],
                "input_len": item["input_len"],
                "expected_output_len": item["output_len"],
                "arrival_time": arrival,
                "first_token_time": None,
                "finish_time": None,
                "observed_tokens": 0,
            }

        steps = 0
        while not llm.is_finished():
            llm.step()
            steps += 1
            if steps > 10000:
                raise RuntimeError("D scheduler exceeded 10000 steps")

        torch.cuda.synchronize()
        run_end = time.perf_counter()
        elapsed = run_end - run_start
        current_rows = []

        for rec in records.values():
            seq = rec["seq"]
            if rec["first_token_time"] is None or rec["finish_time"] is None:
                raise RuntimeError(f"Missing timestamp for {rec['request_id']}")
            actual = seq.num_completion_tokens
            expected = rec["expected_output_len"]
            if actual != expected:
                raise RuntimeError(
                    f"D output mismatch {rec['request_id']}: {actual} != {expected}"
                )

            ttft_ms = (rec["first_token_time"] - rec["arrival_time"]) * 1000
            e2e_ms = (rec["finish_time"] - rec["arrival_time"]) * 1000
            # This is a step-boundary average estimate, not token-level ITL.
            tpot_est_ms = (
                (rec["finish_time"] - rec["first_token_time"])
                / max(actual - 1, 1)
            ) * 1000

            row = {
                "run": run_idx,
                "request_id": rec["request_id"],
                "group": rec["group"],
                "input_len": rec["input_len"],
                "expected_output_len": expected,
                "actual_output_tokens": actual,
                "ttft_ms": round(ttft_ms, 3),
                "e2e_ms": round(e2e_ms, 3),
                "tpot_est_ms": round(tpot_est_ms, 3),
            }
            current_rows.append(row)
            request_rows.append(row)

        actual_output = sum(r["actual_output_tokens"] for r in current_rows)
        if len(current_rows) != total_requests or actual_output != total_output:
            raise RuntimeError("D request/token count validation failed")

        run_row = {
            "scenario": "D",
            "label": "mixed_request_scheduling",
            "run": run_idx,
            "request_count": total_requests,
            "input_tokens": total_input,
            "output_tokens": actual_output,
            "time_seconds": round(elapsed, 6),
            "output_throughput_tok_s": round(actual_output / elapsed, 4),
            "request_throughput_req_s": round(total_requests / elapsed, 6),
            "scheduler_steps": steps,
            "prompt_content_seed": trial_seed,
        }
        trial_rows.append(run_row)

        group_metrics = {}
        for group in ("short", "medium", "long"):
            group_metrics[group] = latency_summary(
                [r for r in current_rows if r["group"] == group]
            )
        overall_metrics = latency_summary(current_rows)

        run_summary = {
            **run_row,
            "overall_latency": overall_metrics,
            "group_latency": group_metrics,
        }
        run_summaries.append(run_summary)
        print(
            f"[D {run_idx:02d}/{repeats}] {actual_output / elapsed:.2f} tok/s | "
            f"E2E P95={overall_metrics['e2e_p95_ms']:.2f} ms | steps={steps}",
            flush=True,
        )

    # Restore original method after D measurements.
    llm.step = original_step

    throughput_stats = stats([r["output_throughput_tok_s"] for r in run_summaries])
    group_final = {}
    for group in ("short", "medium", "long"):
        group_final[group] = {}
        for metric in (
            "ttft_p50_ms", "ttft_p95_ms",
            "e2e_p50_ms", "e2e_p95_ms",
            "tpot_est_p50_ms", "tpot_est_p95_ms",
        ):
            group_final[group][metric] = round(
                statistics.median([r["group_latency"][group][metric] for r in run_summaries]),
                3,
            )
        group_final[group]["request_count"] = next(
            g["count"] for g in D_GROUPS if g["name"] == group
        )

    overall_final = {}
    for metric in (
        "ttft_p50_ms", "ttft_p95_ms",
        "e2e_p50_ms", "e2e_p95_ms",
        "tpot_est_p50_ms", "tpot_est_p95_ms",
    ):
        overall_final[metric] = round(
            statistics.median([r["overall_latency"][metric] for r in run_summaries]),
            3,
        )

    return {
        "scenario": "D",
        "label": "mixed_request_scheduling",
        "request_count": total_requests,
        "input_tokens_per_run": total_input,
        "output_tokens_per_run": total_output,
        "max_num_batched_tokens": token_budget,
        "warmup_runs": WARMUPS,
        "repeats": repeats,
        "output_throughput_tok_s": throughput_stats,
        "request_throughput_req_s": stats([r["request_throughput_req_s"] for r in run_summaries]),
        "time_seconds": stats([r["time_seconds"] for r in run_summaries]),
        "overall_latency_median_across_runs": overall_final,
        "group_latency_median_across_runs": group_final,
        "runs": run_summaries,
        "measurement_note": (
            "TTFT/E2E timestamps are sampled after each scheduler step and CUDA synchronization. "
            "TPOT is an estimated average interval from first-token step to finish step, "
            "not token-level ITL. This is an offline batch benchmark, not an online arrival benchmark."
        ),
    }


def write_markdown(path, result):
    s = result["summary"]
    lines = [
        "# nano-vLLM Benchmark v1.0 — Baseline Report",
        "",
        f"- Model: `{result['model']}`",
        f"- GPU: `{result['gpu']}` (logical device: `{result['logical_gpu']}`)",
        f"- PyTorch: `{result['pytorch']}`; CUDA runtime: `{result['cuda_runtime']}`",
        f"- Max model length: `{result['max_model_len']}`",
        f"- Repeats per case: `{result['repeats']}`; warmups per case: `{result['warmup_runs']}`",
        f"- Sampling: temperature `{TEMPERATURE}`, ignore_eos `{IGNORE_EOS}`",
        "- Prompt policy: fixed length profile per case; fresh random token-ID content for warmup and every measured repeat to avoid reusing identical prefixes.",
        "",
        "## A/B/C throughput summary",
        "",
        "| Scenario | Label | Requests | Input tokens/run | Output tokens/run | Median output tok/s | Mean tok/s | CV | Median time (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in s["A_B_C"]:
        lines.append(
            f"| {case['scenario']} | {case['label']} | {case['request_count']} | "
            f"{case['input_tokens_per_run']} | {case['output_tokens_per_run']} | "
            f"{case['output_throughput_tok_s']['median']:.2f} | "
            f"{case['output_throughput_tok_s']['mean']:.2f} | "
            f"{case['output_throughput_tok_s']['cv_percent']:.3f}% | "
            f"{case['time_seconds']['median']:.4f} |"
        )
    d = s["D"]
    lines += [
        "",
        "## D mixed-request scheduling summary",
        "",
        f"- Requests/run: {d['request_count']}; input tokens/run: {d['input_tokens_per_run']}; output tokens/run: {d['output_tokens_per_run']}",
        f"- Median output throughput: {d['output_throughput_tok_s']['median']:.2f} tok/s",
        f"- Mean output throughput: {d['output_throughput_tok_s']['mean']:.2f} tok/s",
        f"- Throughput CV: {d['output_throughput_tok_s']['cv_percent']:.3f}%",
        "",
        "| Request group | Count | TTFT P50 (ms) | TTFT P95 (ms) | E2E P50 (ms) | E2E P95 (ms) | TPOT estimate P50 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("short", "medium", "long"):
        m = d["group_latency_median_across_runs"][group]
        lines.append(
            f"| {group} | {m['request_count']} | {m['ttft_p50_ms']:.3f} | "
            f"{m['ttft_p95_ms']:.3f} | {m['e2e_p50_ms']:.3f} | "
            f"{m['e2e_p95_ms']:.3f} | {m['tpot_est_p50_ms']:.3f} |"
        )
    lines += [
        "",
        "### Measurement boundary",
        "",
        d["measurement_note"],
        "",
        "## Metric definitions",
        "",
        "- Output throughput = actual generated output tokens / measured wall time.",
        "- Request throughput = completed requests / measured wall time.",
        "- TTFT = first observed output-token step time − request enqueue time.",
        "- E2E = observed finish-step time − request enqueue time.",
        "- TPOT estimate = (finish-step time − first-output-step time) / (output tokens − 1).",
        "- P50/P95 are calculated within each repeat; the displayed D group values are medians of the 10 repeat-level percentiles.",
        "- CV = sample standard deviation / arithmetic mean × 100%.",
        "",
        "## Important interpretation notes",
        "",
        "- A, B and C use the generate() API; D uses step-level instrumentation. Do not compare D throughput directly with A/B/C throughput.",
        "- D is an offline batch submission test. It does not represent sustained online arrivals or prove online scheduling fairness.",
        "- TTFT/E2E are scheduler-step-boundary measurements; TPOT is an estimate, not exact per-token ITL.",
        "- This report is valid only for this recorded source revision, hardware, model and runtime configuration.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be >= 2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Run: export CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device (physical GPU 0)")
    if not Path(args.model).is_dir():
        raise FileNotFoundError(args.model)

    out = Path(args.output_dir)
    if (out / "summary" / "summary.json").exists():
        raise FileExistsError(
            f"A completed benchmark already exists at {out}. "
            "Use a new --output-dir to avoid overwriting prior results."
        )
    env_dir = out / "environment"
    raw_dir = out / "raw"
    summary_dir = out / "summary"
    logs_dir = out / "logs"
    for d in (env_dir, raw_dir, summary_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Capture environment and source fingerprint before running.
    (env_dir / "runtime.txt").write_text(
        "\n".join([
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
            f"Python={__import__('platform').python_version()}",
            f"PyTorch={torch.__version__}",
            f"CUDA runtime={torch.version.cuda}",
            f"GPU={torch.cuda.get_device_name(0)}",
            f"GPU memory bytes={torch.cuda.get_device_properties(0).total_memory}",
            f"Model={args.model}",
            f"Repeats={args.repeats}",
            f"Warmups={WARMUPS}",
            f"Max model len={args.max_model_len}",
            f"Temperature={TEMPERATURE}",
            f"Ignore EOS={IGNORE_EOS}",
        ]) + "\n",
        encoding="utf-8",
    )
    try:
        gpu_info = subprocess.run(
            ["nvidia-smi", "-i", "1"], capture_output=True, text=True, check=False
        ).stdout
        (env_dir / "gpu_before.txt").write_text(gpu_info, encoding="utf-8")
    except FileNotFoundError:
        (env_dir / "gpu_before.txt").write_text("nvidia-smi unavailable\n", encoding="utf-8")

    source_files = sorted(Path("nanovllm").rglob("*.py"))
    hash_lines = []
    import hashlib
    for file in source_files:
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        hash_lines.append(f"{digest}  {file.as_posix()}")
    (env_dir / "source_sha256.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    try:
        pip_freeze = subprocess.run(
            ["python", "-m", "pip", "freeze"],
            capture_output=True, text=True, check=False
        ).stdout
        (env_dir / "requirements.txt").write_text(pip_freeze, encoding="utf-8")
    except Exception as exc:
        (env_dir / "requirements.txt").write_text(
            f"Could not capture pip freeze: {exc}\n", encoding="utf-8"
        )
    try:
        git_info = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False
        )
        git_text = git_info.stdout.strip() if git_info.returncode == 0 else "unavailable (not a git checkout)"
        (env_dir / "git_revision.txt").write_text(git_text + "\n", encoding="utf-8")
    except Exception as exc:
        (env_dir / "git_revision.txt").write_text(f"unavailable: {exc}\n", encoding="utf-8")

    # Fixed case definitions. A's random length profile is generated once and
    # reused across repeats; only prompt token contents change from run to run.
    len_rng = random.Random(args.seed + 11)
    a_input_lengths = [len_rng.randint(100, 1024) for _ in range(256)]
    a_output_lengths = [len_rng.randint(100, 1024) for _ in range(256)]

    cases = []
    cases.append(("A", "random_length_256", a_input_lengths, a_output_lengths))
    for n in (1, 16, 64, 128, 256):
        cases.append((f"B{n}", f"batch_size_{n}", [128] * n, [128] * n))
    for inp in (512, 1024, 2048):
        cases.append((f"C{inp}", f"input_length_{inp}_batch_64", [inp] * 64, [128] * 64))

    config = {
        "benchmark_version": "1.0",
        "model": args.model,
        "physical_gpu": 1,
        "visible_devices": "1",
        "max_model_len": args.max_model_len,
        "repeats": args.repeats,
        "warmups_per_case": WARMUPS,
        "base_seed": args.seed,
        "temperature": TEMPERATURE,
        "ignore_eos": IGNORE_EOS,
        "enforce_eager": False,
        "prompt_policy": "fixed length profile per case; new random token-ID contents for warmup and each measured repeat",
        "scenarios": {
            "A": "256 requests; input and output lengths independently sampled once from [100,1024] and held fixed across repeats",
            "B": "batch sizes 1,16,64,128,256; input=128 and output=128",
            "C": "batch=64; input lengths 512,1024,2048; output=128",
            "D": D_GROUPS,
        },
    }
    (out / "benchmark_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("=" * 72)
    print("nano-vLLM Benchmark v1.0")
    print("Physical GPU 0:", torch.cuda.get_device_name(0))
    print("Model:", args.model)
    print("Output directory:", out)
    print("=" * 72)

    print("Loading model...")
    llm = LLM(args.model, enforce_eager=False, max_model_len=args.max_model_len)
    token_budget = llm.scheduler.max_num_batched_tokens
    max_num_seqs = getattr(llm.scheduler, "max_num_seqs", None)
    print("max_num_batched_tokens:", token_budget)

    all_trial_rows = []
    summaries = []
    for idx, (case_id, label, input_lengths, output_lengths) in enumerate(cases):
        print("\n" + "=" * 72)
        print(f"Scenario {case_id}: {label}; requests={len(input_lengths)}")
        case_seed = args.seed + (idx + 1) * 10_000_019
        summaries.append(
            run_generate_case(
                llm, case_id, label, input_lengths, output_lengths,
                case_seed, args.repeats, all_trial_rows
            )
        )

    print("\n" + "=" * 72)
    print("Scenario D: mixed request scheduling")
    d_summary = run_d_case(
        llm, args.seed + 700_000_001, args.repeats,
        all_trial_rows, d_request_rows := []
    )

    # Save raw trial records and request-level D records.
    write_csv(raw_dir / "trial_results.csv", all_trial_rows)
    write_csv(raw_dir / "D_request_records.csv", d_request_rows)

    result = {
        "benchmark_version": "1.0",
        "model": args.model,
        "gpu": "Physical GPU 0",
        "logical_gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": token_budget,
        "max_num_seqs": max_num_seqs,
        "repeats": args.repeats,
        "warmup_runs": WARMUPS,
        "sampling": {"temperature": TEMPERATURE, "ignore_eos": IGNORE_EOS},
        "prompt_policy": config["prompt_policy"],
        "summary": {"A_B_C": summaries, "D": d_summary},
        "raw_files": [
            "raw/trial_results.csv",
            "raw/D_request_records.csv",
        ],
    }
    (raw_dir / "all_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (summary_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Compact, one-row-per-case summary CSV (D is additionally split by group).
    compact = []
    for x in summaries:
        compact.append({
            "scenario": x["scenario"],
            "label": x["label"],
            "request_count": x["request_count"],
            "input_tokens_per_run": x["input_tokens_per_run"],
            "output_tokens_per_run": x["output_tokens_per_run"],
            "median_output_tok_s": x["output_throughput_tok_s"]["median"],
            "mean_output_tok_s": x["output_throughput_tok_s"]["mean"],
            "std_output_tok_s": x["output_throughput_tok_s"]["std_sample"],
            "cv_percent": x["output_throughput_tok_s"]["cv_percent"],
            "median_time_seconds": x["time_seconds"]["median"],
        })
    d = d_summary
    compact.append({
        "scenario": "D", "label": d["label"],
        "request_count": d["request_count"],
        "input_tokens_per_run": d["input_tokens_per_run"],
        "output_tokens_per_run": d["output_tokens_per_run"],
        "median_output_tok_s": d["output_throughput_tok_s"]["median"],
        "mean_output_tok_s": d["output_throughput_tok_s"]["mean"],
        "std_output_tok_s": d["output_throughput_tok_s"]["std_sample"],
        "cv_percent": d["output_throughput_tok_s"]["cv_percent"],
        "median_time_seconds": d["time_seconds"]["median"],
    })
    write_csv(summary_dir / "summary.csv", compact)
    write_markdown(summary_dir / "summary.md", result)

    try:
        gpu_after = subprocess.run(
            ["nvidia-smi", "-i", "1"], capture_output=True, text=True, check=False
        ).stdout
        (env_dir / "gpu_after.txt").write_text(gpu_after, encoding="utf-8")
    except FileNotFoundError:
        pass

    print("\n" + "=" * 72)
    print("BENCHMARK V1.0 COMPLETED")
    print("Final report:", summary_dir / "summary.md")
    print("Final summary JSON:", summary_dir / "summary.json")
    print("Compact summary CSV:", summary_dir / "summary.csv")
    print("Raw trials:", raw_dir / "trial_results.csv")
    print("D request records:", raw_dir / "D_request_records.csv")
    print("=" * 72)

    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A-only replay of frozen nano-vLLM Benchmark v1.0, with optional scheduler tracing.

Run from the nano-vllm-main project root. Does not change upstream benchmark or source.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def load_frozen_benchmark(project: Path):
    candidates = [
        project / 'benchmarks/phase1/run_benchmark_v1.py',
        project / 'results/phase1_tools/run_benchmark_v1.py',
    ]
    bench_path = next((p for p in candidates if p.is_file()), None)
    if bench_path is None:
        raise FileNotFoundError(
            'Benchmark v1 script not found in benchmarks/phase1 or results/phase1_tools'
        )
    spec = importlib.util.spec_from_file_location('frozen_nano_vllm_benchmark_v1', bench_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, bench_path


def profile_scheduler(scheduler):
    """Wrap a Scheduler instance, not source files. Activate only after shape warmup."""
    from collections import Counter
    counts = Counter()
    durations_ns = []
    prefill_batch_sizes = []
    decode_batch_sizes = []
    prefill_step_tokens = []
    prefill_step_utilization = []
    original_schedule = scheduler.schedule
    original_preempt = scheduler.preempt
    bm = scheduler.block_manager
    original_can_allocate = bm.can_allocate
    budget = scheduler.max_num_batched_tokens

    def timed_can_allocate(seq):
        counts['can_allocate_calls'] += 1
        result = original_can_allocate(seq)
        if result == -1:
            counts['can_allocate_failures'] += 1
        return result

    def tracked_preempt(seq):
        counts['preempt_calls'] += 1
        return original_preempt(seq)

    def tracked_schedule():
        waiting_before = list(scheduler.waiting)
        running_before = len(scheduler.running)
        counts['max_waiting_len'] = max(counts['max_waiting_len'], len(waiting_before))
        counts['max_running_len'] = max(counts['max_running_len'], running_before)
        counts['min_free_blocks'] = min(
            counts.get('min_free_blocks', len(bm.free_block_ids)), len(bm.free_block_ids)
        )
        started_ns = time.perf_counter_ns()
        chosen, is_prefill = original_schedule()
        elapsed_ns = time.perf_counter_ns() - started_ns
        durations_ns.append(elapsed_ns)
        counts['schedule_calls'] += 1
        if is_prefill:
            counts['prefill_steps'] += 1
            prefill_batch_sizes.append(len(chosen))
            tokens = sum(s.num_scheduled_tokens for s in chosen)
            prefill_step_tokens.append(tokens)
            prefill_step_utilization.append(tokens / budget)
            counts['prefill_tokens_scheduled'] += tokens
            # A bypass was observed if a scheduled request has a preceding
            # waiting-queue request not scheduled in this prefill step.
            selected = {id(s) for s in chosen}
            unselected_predecessor = False
            bypassed_this_step = False
            for seq in waiting_before:
                if id(seq) in selected:
                    if unselected_predecessor:
                        counts['requests_selected_after_bypass'] += 1
                        bypassed_this_step = True
                else:
                    unselected_predecessor = True
            if bypassed_this_step:
                counts['steps_with_bypass'] += 1
        else:
            counts['decode_steps'] += 1
            decode_batch_sizes.append(len(chosen))
        return chosen, is_prefill

    bm.can_allocate = timed_can_allocate
    scheduler.preempt = tracked_preempt
    scheduler.schedule = tracked_schedule

    def percentile(values, fraction):
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * fraction
        lo = int(k)
        hi = min(lo + 1, len(sorted_values) - 1)
        return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)

    def snapshot():
        # Values are cumulative since the trace was attached.
        d = dict(counts)
        d.update({
            'scheduler_cpu_time_ms': round(sum(durations_ns) / 1e6, 4),
            'scheduler_cpu_mean_ms_per_call': round(statistics.mean(durations_ns) / 1e6, 5) if durations_ns else 0.0,
            'scheduler_cpu_p95_ms_per_call': round(percentile(durations_ns, 0.95) / 1e6, 5),
            'mean_prefill_batch_size': round(statistics.mean(prefill_batch_sizes), 3) if prefill_batch_sizes else 0.0,
            'mean_decode_batch_size': round(statistics.mean(decode_batch_sizes), 3) if decode_batch_sizes else 0.0,
            'mean_prefill_tokens_per_step': round(statistics.mean(prefill_step_tokens), 3) if prefill_step_tokens else 0.0,
            'mean_prefill_token_budget_utilization_percent': round(100 * statistics.mean(prefill_step_utilization), 3) if prefill_step_utilization else 0.0,
        })
        return d

    def reset():
        counts.clear()
        durations_ns.clear()
        prefill_batch_sizes.clear()
        decode_batch_sizes.clear()
        prefill_step_tokens.clear()
        prefill_step_utilization.clear()

    def detach():
        scheduler.schedule = original_schedule
        scheduler.preempt = original_preempt
        bm.can_allocate = original_can_allocate

    return snapshot, reset, detach


def main():
    parser = argparse.ArgumentParser(description='Replay frozen Benchmark v1.0 Scenario A only')
    parser.add_argument('--model', default=os.environ.get('NANO_VLLM_MODEL'))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--profile', action='store_true', help='Measure scheduler behavior; adds CPU tracing overhead')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Use physical GPU 0: export CUDA_VISIBLE_DEVICES=0')
    if args.repeats < 1:
        raise ValueError('--repeats must be >= 1')
    project = Path.cwd()
    if not (project / 'nanovllm/engine/scheduler.py').is_file():
        raise RuntimeError('Run from nano-vllm-main root')
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f'Output directory already exists: {output}. Use a new directory.')
    output.mkdir(parents=True)
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    benchmark, bench_path = load_frozen_benchmark(project)
    import torch
    from nanovllm import LLM
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one visible CUDA device (physical GPU 0)')
    if not Path(args.model).is_dir():
        raise FileNotFoundError(args.model)

    import random
    lengths_rng = random.Random(args.seed + 11)
    input_lengths = [lengths_rng.randint(100, 1024) for _ in range(256)]
    output_lengths = [lengths_rng.randint(100, 1024) for _ in range(256)]
    case_seed = args.seed + 1 * 10_000_019  # A is first in frozen v1.0 case ordering
    expected_output = sum(output_lengths)
    assert sum(input_lengths) == 146429 and expected_output == 142226, 'A lengths changed from frozen baseline'
    if hasattr(benchmark, 'BASE_SEED'):
        assert benchmark.BASE_SEED == 20261107, 'Frozen benchmark seed has changed'

    git_revision = subprocess.run(['git','rev-parse','--short','HEAD'], capture_output=True, text=True).stdout.strip()
    scheduler_sha = hashlib.sha256((project / 'nanovllm/engine/scheduler.py').read_bytes()).hexdigest()
    print(f'Git revision: {git_revision} | profile={args.profile}', flush=True)
    print(f'Scheduler sha256: {scheduler_sha}', flush=True)
    print(f'GPU: {torch.cuda.get_device_name(0)} | Torch {torch.__version__} CUDA {torch.version.cuda}', flush=True)
    print(f'Workload: A, inputs={sum(input_lengths)}, outputs={expected_output}, repeats={args.repeats}', flush=True)
    print(f'Benchmark imported from: {bench_path}', flush=True)
    llm = LLM(args.model, enforce_eager=False, max_model_len=args.max_model_len)

    warm_seed = case_seed + 900_000_000
    warm_prompts = benchmark.make_prompts(input_lengths, warm_seed)
    warm_params = benchmark.make_sampling_params(output_lengths)
    torch.manual_seed(warm_seed)
    torch.cuda.manual_seed_all(warm_seed)
    warm_outputs = llm.generate(warm_prompts, warm_params, use_tqdm=False)
    warm_count = sum(len(x['token_ids']) for x in warm_outputs)
    if warm_count != expected_output:
        raise RuntimeError(f'Warmup token mismatch: {warm_count} != {expected_output}')

    snapshot = reset = detach = None
    if args.profile:
        snapshot, reset, detach = profile_scheduler(llm.scheduler)
    rows = []
    try:
        for idx in range(1, args.repeats + 1):
            trial_seed = case_seed + idx * 1_000_003
            prompts = benchmark.make_prompts(input_lengths, trial_seed)
            sampling_params = benchmark.make_sampling_params(output_lengths)
            torch.manual_seed(trial_seed)
            torch.cuda.manual_seed_all(trial_seed)
            if reset:
                reset()
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            actual_tokens = sum(len(x['token_ids']) for x in outputs)
            if len(outputs) != 256 or actual_tokens != expected_output:
                raise RuntimeError(f'Run {idx}: invalid outputs {len(outputs)}, tokens {actual_tokens}')
            row = {
                'run': idx,
                'seed': trial_seed,
                'elapsed_seconds': round(elapsed, 6),
                'output_tokens': actual_tokens,
                'throughput_tok_s': round(actual_tokens / elapsed, 4),
            }
            if snapshot:
                row['scheduler'] = snapshot()
            rows.append(row)
            s = row.get('scheduler', {})
            print(
                f"[A {idx:02d}/{args.repeats}] {row['throughput_tok_s']:.2f} tok/s | "
                f"{elapsed:.3f} s | prefills={s.get('prefill_steps', '-')} | "
                f"decodes={s.get('decode_steps', '-')} | "
                f"bypass_steps={s.get('steps_with_bypass', '-')} | "
                f"preemptions={s.get('preempt_calls', '-')}",
                flush=True,
            )
    finally:
        if detach:
            detach()
    summary = {
        'scenario': 'A',
        'label': 'random_length_256',
        'profile': args.profile,
        'source_commit': git_revision,
        'scheduler_sha256': scheduler_sha,
        'benchmark_source_sha256': hashlib.sha256(bench_path.read_bytes()).hexdigest(),
        'gpu': torch.cuda.get_device_name(0),
        'pytorch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'seed': args.seed,
        'repeats': args.repeats,
        'model': args.model,
        'input_tokens_per_run': sum(input_lengths),
        'output_tokens_per_run': expected_output,
        'median_throughput_tok_s': round(statistics.median(r['throughput_tok_s'] for r in rows), 4),
        'mean_throughput_tok_s': round(statistics.mean(r['throughput_tok_s'] for r in rows), 4),
        'rows': rows,
    }
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"A-only median throughput: {summary['median_throughput_tok_s']:.2f} tok/s", flush=True)
    print(f'Saved: {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""E1: real-GPU prefilling token-budget HOL test for baseline vs bypass.

Runs against the user's nano-vLLM checkout, does not modify its files.
Only a controlled budget-induced bypass test, NOT a KV-memory-pressure or online-arrival benchmark.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path


def pct(values, p):
    a = sorted(values)
    if not a:
        return None
    q = (len(a) - 1) * p
    i = int(q)
    j = min(i + 1, len(a) - 1)
    return round(a[i] + (a[j] - a[i]) * (q - i), 3)


def create_workload(seed, triplets, out_tokens):
    rng = random.Random(seed)
    result = []
    for k in range(triplets):
        for tag, n in [('long', 768), ('medium', 512), ('short', 256)]:
            result.append({
                'request_id': f'{tag}_{k:02d}', 'group': tag,
                'input_len': n, 'output_len': out_tokens,
                'prompt': [rng.randint(100, 10000) for _ in range(n)],
            })
    return result


def main():
    p = argparse.ArgumentParser(description='E1: budget head-of-line bypass real-GPU test')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--model', default=os.environ.get("NANO_VLLM_MODEL"))
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--triplets', type=int, default=24)
    p.add_argument('--budget', type=int, default=1024)
    p.add_argument('--max-model-len', type=int, default=4096)
    p.add_argument('--output-tokens', type=int, default=64)
    args = p.parse_args()

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise SystemExit('Please set CUDA_VISIBLE_DEVICES=0 (physical GPU 0).')
    if args.budget < 768 or args.triplets < 1 or args.repeats < 1 or args.output_tokens < 2:
        raise SystemExit('Require budget >= 768, triplets >= 1, repeats >= 1, output_tokens >= 2.')
    repo = Path.cwd()
    if not (repo / 'nanovllm/engine/scheduler.py').is_file():
        raise SystemExit('Run from nano-vllm-main project root.')
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        raise SystemExit(f'Output already exists: {out_dir}. Choose a new directory.')
    out_dir.mkdir(parents=True)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    import torch
    from nanovllm import LLM, SamplingParams
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit('Expected exactly 1 visible GPU.')
    if not Path(args.model).is_dir():
        raise SystemExit(f'Model directory not found: {args.model}')
    rev = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    sha = hashlib.sha256((repo/'nanovllm/engine/scheduler.py').read_bytes()).hexdigest()
    dirty = bool(subprocess.run(['git', 'diff', '--quiet', '--', 'nanovllm/engine/scheduler.py']).returncode)
    config = vars(args).copy()
    config.update({'git_revision': rev, 'scheduler_sha256': sha, 'scheduler_modified_uncommitted': dirty,
                   'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                   'cuda': torch.version.cuda, 'workload_order': 'long768,medium512,short256 repeated'})
    print('Git:', rev, '(uncommitted Scheduler changes)' if dirty else '', '| SHA:', sha, flush=True)
    print('GPU:', config['gpu'], '| budget:', args.budget, '| triplets:', args.triplets, flush=True)
    llm = LLM(model=args.model, enforce_eager=False, max_model_len=args.max_model_len)
    if args.budget > llm.scheduler.max_num_batched_tokens:
        raise SystemExit(f'Requested budget {args.budget} exceeds initialized budget')
    # Experimental throttle of the scheduler only; model runner and kernel layout unchanged.
    llm.scheduler.max_num_batched_tokens = args.budget
    print('Physical KV blocks:', len(llm.scheduler.block_manager.blocks), flush=True)

    def params(work):
        return [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=x['output_len']) for x in work]

    # One shape-matched warmup with separate seed to avoid exact prefix reuse.
    warm = create_workload(args.seed + 900_000_000, args.triplets, args.output_tokens)
    outputs = llm.generate([x['prompt'] for x in warm], params(warm), use_tqdm=False)
    exp_tokens = len(warm) * args.output_tokens
    if len(outputs) != len(warm) or sum(len(x['token_ids']) for x in outputs) != exp_tokens:
        raise RuntimeError('Warmup invalid output counts')

    sch = llm.scheduler
    original_schedule = sch.schedule
    original_preempt = sch.preempt
    state = {}

    def tracked_schedule():
        before = list(sch.waiting)
        start = time.perf_counter_ns()
        selected, is_prefill = original_schedule()
        elapsed = time.perf_counter_ns() - start
        state['schedule_ns'] += elapsed
        state['schedule_calls'] += 1
        chosen_ids = {id(s) for s in selected}
        if is_prefill:
            state['prefill_steps'] += 1
            state['prefill_tokens'] += sum(s.num_scheduled_tokens for s in selected)
            skipped = False
            bypassed = 0
            for s in before:
                if id(s) in chosen_ids:
                    if skipped:
                        bypassed += 1
                else:
                    skipped = True
            if bypassed:
                state['bypass_steps'] += 1
                state['bypassed_requests_selected'] += bypassed
            if state['prefill_steps'] == 1:
                state['first_prefill_selected'] = [state['request_group_by_seq_id'].get(s.seq_id, '?') for s in selected]
        else:
            state['decode_steps'] += 1
        return selected, is_prefill

    def tracked_preempt(seq):
        state['preemptions'] += 1
        return original_preempt(seq)

    sch.schedule = tracked_schedule
    sch.preempt = tracked_preempt
    all_trials, all_requests = [], []
    try:
        for run in range(1, args.repeats + 1):
            seed = args.seed + run * 1000003
            work = create_workload(seed, args.triplets, args.output_tokens)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            state.clear()
            state.update({'schedule_ns': 0, 'schedule_calls': 0, 'prefill_steps': 0,
                          'decode_steps': 0, 'prefill_tokens': 0,
                          'bypass_steps': 0, 'bypassed_requests_selected': 0,
                          'preemptions': 0, 'first_prefill_selected': [],
                          'request_group_by_seq_id': {}})
            records = {}
            torch.cuda.synchronize()
            run_start = time.perf_counter()
            for item in work:
                t = time.perf_counter()
                llm.add_request(item['prompt'], SamplingParams(
                    temperature=0.6, ignore_eos=True, max_tokens=item['output_len']))
                seq = sch.waiting[-1]
                state['request_group_by_seq_id'][seq.seq_id] = item['group']
                records[seq.seq_id] = {
                    'seq': seq, 'request_id': item['request_id'], 'group': item['group'],
                    'arrival': t, 'first': None, 'end': None, 'n_seen': 0}
            steps = 0
            while not llm.is_finished():
                llm.step()
                torch.cuda.synchronize()
                now = time.perf_counter()
                for r in records.values():
                    seq = r['seq']
                    n = seq.num_completion_tokens
                    if n > r['n_seen']:
                        if r['first'] is None:
                            r['first'] = now
                        r['n_seen'] = n
                    if seq.is_finished and r['end'] is None:
                        r['end'] = now
                steps += 1
                if steps > args.output_tokens * 10 + args.triplets * 10 + 1000:
                    raise RuntimeError('Step guard exceeded; likely no progress')
            torch.cuda.synchronize()
            end = time.perf_counter()
            local = []
            for r in records.values():
                if r['first'] is None or r['end'] is None:
                    raise RuntimeError(f'Missing timing for {r["request_id"]}')
                if r['seq'].num_completion_tokens != args.output_tokens:
                    raise RuntimeError(f'Output length mismatch for {r["request_id"]}')
                row = {
                    'run': run, 'request_id': r['request_id'], 'group': r['group'],
                    'ttft_ms': round((r['first']-r['arrival'])*1000, 3),
                    'e2e_ms': round((r['end']-r['arrival'])*1000, 3),
                    'tpot_est_ms': round((r['end']-r['first'])*1000/(args.output_tokens-1), 3)
                }
                local.append(row)
                all_requests.append(row)
            elapsed = end - run_start
            row = {
                'run': run, 'elapsed_sec': round(elapsed, 6),
                'throughput_tok_s': round(exp_tokens/elapsed, 3),
                'schedule_calls': state['schedule_calls'],
                'prefill_steps': state['prefill_steps'], 'decode_steps': state['decode_steps'],
                'prefill_tokens': state['prefill_tokens'],
                'schedule_cpu_ms': round(state['schedule_ns']/1e6, 3),
                'bypass_steps': state['bypass_steps'],
                'bypassed_requests_selected': state['bypassed_requests_selected'],
                'preemptions': state['preemptions'],
                'first_prefill_selected': state['first_prefill_selected'],
            }
            for group in ('long','medium','short'):
                group_rows = [x for x in local if x['group'] == group]
                row[group+'_ttft_p50_ms'] = pct([x['ttft_ms'] for x in group_rows], .50)
                row[group+'_ttft_p95_ms'] = pct([x['ttft_ms'] for x in group_rows], .95)
                row[group+'_e2e_p95_ms'] = pct([x['e2e_ms'] for x in group_rows], .95)
            all_trials.append(row)
            print(f'[E1 {run}/{args.repeats}] tok/s={row["throughput_tok_s"]:.2f} '
                  f'prefill={row["prefill_steps"]} bypass={row["bypass_steps"]} '
                  f'short TTFT P50={row["short_ttft_p50_ms"]:.2f}ms '
                  f'preempt={row["preemptions"]} '
                  f'first_batch={row["first_prefill_selected"][:6]}', flush=True)
    finally:
        sch.schedule = original_schedule
        sch.preempt = original_preempt

    metrics = ['throughput_tok_s','prefill_steps','decode_steps','schedule_cpu_ms',
               'bypass_steps','bypassed_requests_selected','preemptions',
               'long_ttft_p50_ms','long_ttft_p95_ms','long_e2e_p95_ms',
               'medium_ttft_p50_ms','medium_ttft_p95_ms','medium_e2e_p95_ms',
               'short_ttft_p50_ms','short_ttft_p95_ms','short_e2e_p95_ms']
    summary = {
        'config': config, 'total_requests_per_run': args.triplets * 3,
        'input_tokens_per_run': args.triplets * 1536,
        'output_tokens_per_run': exp_tokens,
        'metrics_median': {k: round(statistics.median(x[k] for x in all_trials),3) for k in metrics},
        'trials': all_trials,
    }
    (out_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (out_dir/'trials.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_trials[0].keys()); writer.writeheader(); writer.writerows(all_trials)
    with (out_dir/'requests.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_requests[0].keys()); writer.writeheader(); writer.writerows(all_requests)
    print('Saved:', out_dir/'summary.json', flush=True)
    print('Median throughput:', summary['metrics_median']['throughput_tok_s'], 'tok/s', flush=True)


if __name__ == '__main__':
    main()

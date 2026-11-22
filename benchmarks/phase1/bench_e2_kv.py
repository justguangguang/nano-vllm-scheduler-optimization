#!/usr/bin/env python3
"""E2: Real GPU experiment isolating queue-bypass decode headroom, not an online service benchmark.

One deterministic resource-pressure episode per repetition:
  - seven *logical accessible* KV-cache blocks (256 tokens/block, full model KV tensor stays allocated)
  - two incumbent running requests with prompts of 510 tokens, each holding 2 blocks
  - after incumbent prefill, three late waiting requests, in this order:
      head: 900 tokens -> needs 4 blocks (unallocatable with 3 free)
      medium: 300 -> 2 blocks
      short: 128 -> 1 block
  - both running owners reach a new KV block in ~3 decode steps.
  - baseline: can't bypass head; v2: admits 2-block medium;
    v3: reserves two decode blocks, admits 1-block short.

All physical block indices allocated by BlockManager remain below the model's real
KV-cache tensor size. Does NOT shrink the GPU tensor; only constrains the allocator.
Run each version in a fresh Python process to avoid stale scheduler references.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

MODEL = os.environ.get('NANO_VLLM_MODEL')
PROJECT = Path(__file__).resolve().parents[2]
WORKLOAD = [('owner_a', 'owner', 510, 32),
            ('owner_b', 'owner', 510, 32),
            ('head', 'head', 900, 32),
            ('medium', 'medium', 300, 32),
            ('short', 'short', 128, 32)]


def percentile(nums, q):
    vals = sorted(nums)
    if not vals:
        raise ValueError('empty percentile')
    z = (len(vals) - 1) * q
    i = int(z)
    j = min(i + 1, len(vals)-1)
    return vals[i] + (vals[j]-vals[i]) * (z-i)


def run_git(*args):
    return subprocess.run(['git', *args], text=True, capture_output=True, check=True).stdout.strip()


def make_prompts(seed):
    r = random.Random(seed)
    return {name: [r.randint(100, 10000) for _ in range(n)] for name, _, n, _ in WORKLOAD}


def main():
    ap = argparse.ArgumentParser(description='E2 KV pressure: baseline vs v2 vs v3')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--seed', type=int, default=20261107)
    ap.add_argument('--kv-blocks', type=int, default=7)
    args = ap.parse_args()
    if args.repeats < 2 or args.kv_blocks != 7:
        ap.error('This fixed E2 scenario requires --kv-blocks 7 and --repeats >= 2.')
    if os.getenv('CUDA_VISIBLE_DEVICES') != '0':
        ap.error('Set CUDA_VISIBLE_DEVICES=0 for physical GPU 0.')
    root = Path.cwd().resolve()
    if root != Path(PROJECT).resolve() or not (root/'nanovllm/engine/scheduler.py').exists():
        ap.error(f'Run from the correct project root: {PROJECT}')
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        ap.error(f'Output directory exists: {output_dir}. Never overwrite previous results.')
    if not args.model or not Path(args.model).is_dir():
        ap.error(f'Model not found: {args.model}')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.block_manager import BlockManager

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        ap.error('Expected precisely one visible GPU.')
    branch = run_git('branch', '--show-current')
    revision = run_git('rev-parse','--short','HEAD')
    if subprocess.run(['git','diff','--quiet','--','nanovllm/engine/scheduler.py']).returncode:
        ap.error('Scheduler has uncommitted edits; commit or restore before benchmarking.')
    sha = hashlib.sha256((root/'nanovllm/engine/scheduler.py').read_bytes()).hexdigest()
    output_dir.mkdir(parents=True)
    print('E2 KV-Cache pressure test | Git:', branch, revision, '| SHA:', sha[:12], flush=True)
    print('GPU:', torch.cuda.get_device_name(0), '| CUDA:', torch.version.cuda, '| torch:',torch.__version__,flush=True)
    llm = LLM(model=args.model, max_model_len=4096, enforce_eager=False)
    sch = llm.scheduler
    block_size = sch.block_manager.block_size
    real_kv_blocks = len(sch.block_manager.blocks)
    if block_size != 256 or real_kv_blocks < args.kv_blocks:
        raise RuntimeError(f'Incompatible model KV cache: block_size={block_size}, physical_blocks={real_kv_blocks}')
    print(f'Physical KV allocation available: {real_kv_blocks} blocks. Restrict metadata manager to {args.kv_blocks}.',flush=True)

    def params(name):
        out = next(x[3] for x in WORKLOAD if x[0] == name)
        return SamplingParams(temperature=0.6, max_tokens=out, ignore_eos=True)

    rows = []
    request_rows = []
    original_schedule = sch.schedule
    original_preempt = sch.preempt
    state = {}

    def wrapped_schedule():
        waiting_before = list(sch.waiting)
        t0 = time.perf_counter_ns()
        seqs, is_prefill = original_schedule()
        state['sched_ns'] += time.perf_counter_ns() - t0
        state['steps'] += 1
        if is_prefill:
            state['prefill_steps'] += 1
            state['prefill_tokens'] += sum(s.num_scheduled_tokens for s in seqs)
            state['prefill_max_batch'] = max(state['prefill_max_batch'],len(seqs))
        else:
            state['decode_steps'] += 1
        if state.get('injected'):
            if state['first_post_inject'] is None:
                state['first_post_inject'] = ('prefill' if is_prefill else 'decode',
                    [state['group_by_id'].get(s.seq_id,'?') for s in seqs])
            if is_prefill:
                chosen = {id(s) for s in seqs}
                seen_skipped = False
                bypassed = 0
                for s in waiting_before:
                    if id(s) in chosen:
                        if seen_skipped:
                            bypassed += 1
                    else:
                        seen_skipped = True
                if bypassed:
                    state['bypass_steps'] += 1
                    state['bypassed_requests'] += bypassed
        return seqs, is_prefill

    def wrapped_preempt(seq):
        state['preemptions'] += 1
        group = state['group_by_id'].get(seq.seq_id,'?')
        state['preempt_by_group'][group] = state['preempt_by_group'].get(group,0) + 1
        return original_preempt(seq)

    def reset_state():
        state.clear()
        state.update(dict(sched_ns=0, steps=0, prefill_steps=0,
            decode_steps=0, prefill_tokens=0, prefill_max_batch=0,
            bypass_steps=0, bypassed_requests=0, preemptions=0,
            preempt_by_group={}, first_post_inject=None, injected=False,
            group_by_id={}))

    def once(seed, run_number, record):
        if not llm.is_finished():
            raise RuntimeError('LLM not idle before run')
        # After all previous sequences are finished, switch to a fresh *smaller*
        # metadata BlockManager. No physical GPU memory is released or moved.
        sch.block_manager = BlockManager(args.kv_blocks, block_size)
        reset_state()
        prompts = make_prompts(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        tracker = {}
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        def add(name):
            group, prompt_len, out_len = next((g,n,o) for nm,g,n,o in WORKLOAD if nm==name)
            t_arrive = time.perf_counter()
            llm.add_request(prompts[name],params(name))
            seq = sch.waiting[-1]
            tracker[name] = dict(seq=seq, group=group, arrival=t_arrive,
                expected=out_len, n_seen=0, first=None, end=None)
            state['group_by_id'][seq.seq_id] = name

        def step_and_observe():
            llm.step()
            torch.cuda.synchronize()
            now = time.perf_counter()
            for item in tracker.values():
                seq = item['seq']
                n = seq.num_completion_tokens
                if n > item['n_seen']:
                    if item['first'] is None:
                        item['first'] = now
                    item['n_seen'] = n
                if seq.is_finished and item['end'] is None:
                    item['end'] = now

        add('owner_a')
        add('owner_b')
        step_and_observe()  # only incumbent prefill, then append first sample
        owners = [tracker[n]['seq'] for n in ('owner_a','owner_b')]
        if not all(s in sch.running for s in owners) or sch.waiting:
            raise RuntimeError('Expected both owners RUNNING after one initial Prefill step')
        if [len(s.block_table) for s in owners] != [2,2] or len(sch.block_manager.free_block_ids) != 3:
            raise RuntimeError('Expected owners use 4 of 7 blocks before injection')
        if not all(len(s) == 511 for s in owners):
            raise RuntimeError('Owner length not 511 after initial sample')
        state['injected'] = True
        for name in ('head','medium','short'):
            add(name)
        state['free_blocks_at_injection'] = len(sch.block_manager.free_block_ids)
        steps_after = 0
        while not llm.is_finished():
            step_and_observe()
            steps_after += 1
            if steps_after > 400:
                raise RuntimeError('E2 no progress / excessive steps (>400)')
        torch.cuda.synchronize()
        total_elapsed = time.perf_counter()-t_start
        output_tokens = 0
        measurements = {}
        for name, group, prompt_len, expected in WORKLOAD:
            item = tracker[name]
            seq = item['seq']
            if item['first'] is None or item['end'] is None or seq.num_completion_tokens != expected:
                raise RuntimeError(f'Incomplete or wrong output count for {name}: got={seq.num_completion_tokens}, expected={expected}')
            output_tokens += expected
            m = dict(run=run_number, request=name, group=group,
                ttft_ms=round((item['first']-item['arrival'])*1000,3),
                e2e_ms=round((item['end']-item['arrival'])*1000,3),
                tpot_est_ms=round((item['end']-item['first'])*1000/(expected-1),3),
                output_tokens=expected)
            measurements[name] = m
            if record:
                request_rows.append(m)
        values = dict(run=run_number, throughput_tok_s=round(output_tokens/total_elapsed,3),
            elapsed_sec=round(total_elapsed,6),
            schedule_cpu_ms=round(state['sched_ns']/1e6,3),
            schedule_calls=state['steps'], prefill_steps=state['prefill_steps'],
            decode_steps=state['decode_steps'],
            prefill_tokens=state['prefill_tokens'],
            prefill_token_overhead=state['prefill_tokens']-sum(x[2] for x in WORKLOAD),
            preemptions=state['preemptions'],
            owner_preemptions=state['preempt_by_group'].get('owner_a',0)+state['preempt_by_group'].get('owner_b',0),
            medium_preemptions=state['preempt_by_group'].get('medium',0),
            bypass_steps=state['bypass_steps'], bypassed_requests=state['bypassed_requests'],
            first_post_inject_kind=state['first_post_inject'][0],
            first_post_inject_batch=state['first_post_inject'][1],
            free_blocks_at_injection=state['free_blocks_at_injection'],
            owner_tpot_est_mean_ms=round(statistics.mean(measurements[n]['tpot_est_ms'] for n in ('owner_a','owner_b')),3),
            owner_e2e_mean_ms=round(statistics.mean(measurements[n]['e2e_ms'] for n in ('owner_a','owner_b')),3),
            head_ttft_ms=measurements['head']['ttft_ms'],
            medium_ttft_ms=measurements['medium']['ttft_ms'],
            short_ttft_ms=measurements['short']['ttft_ms'],
            short_e2e_ms=measurements['short']['e2e_ms'])
        if record:
            print(f"[E2 {run_number}/{args.repeats}] throughput={values['throughput_tok_s']:.2f} tok/s "
                  f"preemptions={values['preemptions']} (medium={values['medium_preemptions']}) "
                  f"prefill={values['prefill_steps']} first_after_inject={values['first_post_inject_kind']}:"
                  f"{values['first_post_inject_batch']} short_ttft={values['short_ttft_ms']:.2f}ms",flush=True)
        return values

    try:
        sch.schedule = wrapped_schedule
        sch.preempt = wrapped_preempt
        # One unrecorded warmup on the EXACT SAME shape and pressure scenario;
        # fresh token IDs ensure no cached prefix hits between trials.
        print('Warmup...',flush=True)
        once(args.seed + 900_000_003, 0, False)
        for i in range(1,args.repeats+1):
            rows.append(once(args.seed + i * 1_000_003,i,True))
    finally:
        sch.schedule = original_schedule
        sch.preempt = original_preempt
    numeric = ['throughput_tok_s','elapsed_sec','schedule_cpu_ms','schedule_calls',
        'prefill_steps','decode_steps','prefill_tokens','prefill_token_overhead',
        'preemptions','owner_preemptions','medium_preemptions','bypass_steps',
        'bypassed_requests','owner_tpot_est_mean_ms','owner_e2e_mean_ms',
        'head_ttft_ms','medium_ttft_ms','short_ttft_ms','short_e2e_ms']
    report = dict(config=dict(model=args.model, repeats=args.repeats, seed=args.seed,
        kv_blocks=args.kv_blocks, block_size=block_size, real_kv_blocks=real_kv_blocks,
        workload=WORKLOAD, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,cuda=torch.version.cuda,
        git_branch=branch,git_revision=revision,scheduler_sha256=sha),
        metrics_median={k:round(statistics.median(r[k] for r in rows),3) for k in numeric},
        trials=rows)
    (output_dir/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    with (output_dir/'trials.csv').open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader()
        for row in rows:
            r=row.copy(); r['first_post_inject_batch']=json.dumps(r['first_post_inject_batch']); w.writerow(r)
    with (output_dir/'requests.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=request_rows[0].keys()); w.writeheader(); w.writerows(request_rows)
    print('Median throughput:',report['metrics_median']['throughput_tok_s'],'tok/s',flush=True)
    print('Saved:',output_dir/'summary.json',flush=True)

if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Compare A-only normal and optional profile runs from diagnostic kit."""
import argparse
import json
import statistics
from pathlib import Path


def load(path):
    p = Path(path)
    if p.is_dir():
        p = p / 'summary.json'
    return json.loads(p.read_text(encoding='utf-8'))


def med(values):
    return statistics.median(values)


def verify(a, b, expected_profile):
    for item in (a, b):
        assert item['scenario'] == 'A'
        assert item['profile'] is expected_profile, f"Expected profile={expected_profile}, got {item['profile']}"
    for key in ('seed', 'input_tokens_per_run', 'output_tokens_per_run', 'benchmark_source_sha256', 'model'):
        assert a[key] == b[key], f'Workload mismatch: {key}'
    assert a['scheduler_sha256'] != b['scheduler_sha256'], 'Both scheduler source hashes match! Are branches different?'


def print_pair(label, a, b, profile):
    verify(a, b, profile)
    aa = a['median_throughput_tok_s']
    bb = b['median_throughput_tok_s']
    delta = (bb / aa - 1.0) * 100.0
    print(f'\n## {label}')
    print(f'Baseline: {aa:,.2f} tok/s ({a["source_commit"]})')
    print(f'Bypass:   {bb:,.2f} tok/s ({b["source_commit"]})')
    print(f'Delta:    {delta:+.3f}%')
    if profile:
        keys = [
            'schedule_calls', 'scheduler_cpu_time_ms', 'scheduler_cpu_mean_ms_per_call',
            'prefill_steps', 'decode_steps', 'prefill_tokens_scheduled',
            'mean_prefill_batch_size', 'mean_decode_batch_size',
            'mean_prefill_token_budget_utilization_percent',
            'preempt_calls', 'can_allocate_calls', 'can_allocate_failures',
            'steps_with_bypass', 'requests_selected_after_bypass',
        ]
        print('\nMetric | baseline per-run median | bypass per-run median')
        print('---|---:|---:')
        for key in keys:
            a_vals = [r['scheduler'].get(key, 0) for r in a['rows']]
            b_vals = [r['scheduler'].get(key, 0) for r in b['rows']]
            print(f'{key} | {med(a_vals):.3f} | {med(b_vals):.3f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', required=True, help='main (unprofiled) result dir or summary.json')
    parser.add_argument('--bypass', required=True, help='bypass (unprofiled) result dir or summary.json')
    parser.add_argument('--main-profile', help='main profiled result dir')
    parser.add_argument('--bypass-profile', help='bypass profiled result dir')
    args = parser.parse_args()
    print_pair('Uninstrumented A-only comparison', load(args.main), load(args.bypass), False)
    if args.main_profile and args.bypass_profile:
        print_pair('Instrumented A-only diagnosis', load(args.main_profile), load(args.bypass_profile), True)
    elif args.main_profile or args.bypass_profile:
        raise ValueError('Provide both profile paths, or neither')


if __name__ == '__main__':
    main()

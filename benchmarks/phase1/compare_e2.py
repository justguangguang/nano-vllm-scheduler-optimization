#!/usr/bin/env python3
"""Three-way comparison for the deliberately constrained E2 KV-cache experiment."""
import argparse
import json
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--main',required=True)
ap.add_argument('--v2',required=True)
ap.add_argument('--v3',required=True)
a=ap.parse_args()
paths={'main':Path(a.main),'v2':Path(a.v2),'v3':Path(a.v3)}
r={k:json.loads((p/'summary.json').read_text()) for k,p in paths.items()}
keys=['model','repeats','seed','kv_blocks','block_size','real_kv_blocks','workload','gpu','torch','cuda']
for key in keys:
    values=[r[v]['config'][key] for v in ('main','v2','v3')]
    if values[0]!=values[1] or values[0]!=values[2]:
        raise SystemExit(f'Configuration mismatch on {key}: {values}')
sha=[r[v]['config']['scheduler_sha256'] for v in ('main','v2','v3')]
if len(set(sha))!=3:
    raise SystemExit('Scheduler hashes are not 3 distinct versions')

for name in ('main','v2','v3'):
    v=r[name]
    print(f"{name}: git={v['config']['git_revision']} ({v['config']['git_branch']}) scheduler_sha={v['config']['scheduler_sha256'][:12]}")

ks=['throughput_tok_s','elapsed_sec','prefill_steps','decode_steps','prefill_tokens',
    'prefill_token_overhead','preemptions','owner_preemptions','medium_preemptions',
    'schedule_cpu_ms','bypass_steps','bypassed_requests','owner_tpot_est_mean_ms',
    'owner_e2e_mean_ms','head_ttft_ms','medium_ttft_ms','short_ttft_ms','short_e2e_ms']
print('\nE2 metric (median) | main | v2 | v3 | v3 vs v2')
print('---|---:|---:|---:|---:')
for k in ks:
    x=[r[name]['metrics_median'][k] for name in ('main','v2','v3')]
    delta='-' if x[1]==0 else f'{(x[2]/x[1]-1)*100:+.2f}%'
    print(f'{k} | {x[0]:.3f} | {x[1]:.3f} | {x[2]:.3f} | {delta}')

print('\nFirst scheduling step after late requests arrive:')
for name in ('main','v2','v3'):
    x=r[name]['trials'][0]
    print(f"{name}: {x['first_post_inject_kind']} {x['first_post_inject_batch']}; preemptions={x['preemptions']}")
print('\nInterpretation guardrails:')
if r['v2']['metrics_median']['preemptions']>r['v3']['metrics_median']['preemptions']:
    print('SUPPORTED: v3 reduced preemptions relative to v2 in this controlled workload.')
else:
    print('NOT DEMONSTRATED: v3 did not reduce preemptions vs v2; inspect raw trials.')
if r['v3']['trials'][0]['first_post_inject_batch'] == ['short']:
    print('EXPECTED: v3 admitted the one-block short request first.')
else:
    print('WARNING: v3 first admission was not the expected short-only prefill.')
print('This is deliberately constrained KV capacity (metadata allocator only); not default-capacity serving.')

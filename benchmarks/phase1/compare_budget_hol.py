#!/usr/bin/env python3
"""Compare identical E1 CPU-instrumented real-GPU workloads."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--main', required=True)
p.add_argument('--v3', required=True)
a = p.parse_args()
base = json.loads((Path(a.main)/'summary.json').read_text())
v3 = json.loads((Path(a.v3)/'summary.json').read_text())
keys = ['model','repeats','seed','triplets','budget','max_model_len','output_tokens','gpu', 'torch','cuda']
for k in keys:
    if base['config'][k] != v3['config'][k]:
        raise SystemExit(f'Incomparable: config {k} differs: {base["config"][k]} vs {v3["config"][k]}')
print('Baseline:',base['config']['git_revision'],base['config']['scheduler_sha256'][:12])
print('V3:',v3['config']['git_revision'],v3['config']['scheduler_sha256'][:12],
      '(uncommitted scheduler)' if v3['config']['scheduler_modified_uncommitted'] else '')
print('Metric | Baseline | V3 | Delta % (positive = V3 bigger)')
print('---|---:|---:|---:')
for k in base['metrics_median']:
    old, new = base['metrics_median'][k], v3['metrics_median'][k]
    pct = (new/old-1)*100 if old else None
    delta = ('-' if pct is None else f'{pct:+.2f}%')
    print(f'{k} | {old:.3f} | {new:.3f} | {delta}')
print('\nFirst trial first-prefill batch:')
print('main:',base['trials'][0]['first_prefill_selected'][:8])
print('v3:  ',v3['trials'][0]['first_prefill_selected'][:8])

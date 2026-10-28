#!/usr/bin/env bash
# Full v2 benchmark: seed 42 (+ v1-design controls), then the other four seeds.
cd "$(dirname "$0")/../.."
PY=fraud_env/Scripts/python.exe
$PY -m experiments.v2.run_v2 --seed 42 --v1-controls
for s in 123 456 789 1024; do
  $PY -m experiments.v2.run_v2 --seed $s
done

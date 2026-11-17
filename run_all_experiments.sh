#!/bin/sh
# Chay toan bo lo thi nghiem: main -> ablation -> sensitivity -> large (co the ngat va chay lai: --resume)
cd "$(dirname "$0")"
python3 run_batch.py --exp main --out results/raw_main.csv --resume
python3 run_batch.py --exp ablation --out results/raw_ablation.csv --resume
python3 run_batch.py --exp large --out results/raw_large.csv --resume
python3 run_batch.py --exp sensitivity --out results/raw_sensitivity.csv --resume
python3 run_batch.py --exp sensitivity_large --out results/raw_sensitivity_large.csv --resume

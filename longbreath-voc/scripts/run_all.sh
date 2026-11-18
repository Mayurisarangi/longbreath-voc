#!/usr/bin/env bash
# Linux/macOS: reproduce all tables and figures (CPU, several hours). Run from the repo root.
set -e
mkdir -p results figs
python scripts/run.py --scenario ideal --out ideal
python scripts/run.py --scenario real --effect 0.5 --out real
python scripts/run.py --scenario real --effect 0.5 --models HTAN-noIIPE,HTAN-noBiLSTM,HTAN-noCross --out abl_real
python scripts/run.py --scenario ideal --models HTAN-noIIPE,HTAN-noBiLSTM,HTAN-noCross --out abl_ideal
for m in linear lmm mask; do
  python scripts/run.py --scenario real --effect 0.5 --models HTAN-small --impute $m --out imp_real_$m
  python scripts/run.py --scenario ideal --models HTAN-small --impute $m --out imp_ideal_$m
done
python scripts/run.py --scenario real --effect 0.5 --models HTAN-small,LMM --global_conf --out leak_real
python scripts/run.py --scenario ideal --models HTAN-small,LMM --global_conf --out leak_ideal
for e in 1.0 0.35 0.25 0.15; do python scripts/run.py --scenario real --effect $e --models LMM,FPCA,HTAN-small --out sweep_$e; done
python src/analyze.py ideal real abl_real abl_ideal leak_real leak_ideal imp_real_linear imp_real_lmm imp_real_mask imp_ideal_linear imp_ideal_lmm imp_ideal_mask sweep_1.0 sweep_0.35 sweep_0.25 sweep_0.15
python scripts/misc.py
python scripts/figs_large.py
python scripts/fig7L.py

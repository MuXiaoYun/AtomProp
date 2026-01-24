#!/bin/bash
PYTHONUNBUFFERED=1 nohup python finetune.py --dataset tox21 --gpu 0 > logs/finetune_nopre_tox21.log 2>&1 &
PYTHONUNBUFFERED=1 nohup python finetune.py --dataset toxcast --gpu 1 > logs/finetune_nopre_toxcast.log 2>&1 &
PYTHONUNBUFFERED=1 nohup python finetune.py --dataset bbbp --gpu 2 > logs/finetune_nopre_bbbp.log 2>&1 &
PYTHONUNBUFFERED=1 nohup python finetune.py --dataset sider --gpu 3 > logs/finetune_nopre_sider.log 2>&1 &
PYTHONUNBUFFERED=1 nohup python finetune.py --dataset clintox --gpu 4 > logs/finetune_nopre_clintox.log 2>&1 &
PYTHONUNBUFFERED=1 nohup python finetune_regression.py --dataset esol --gpu 5 > logs/finetune_nopre_esol.log 2>&1 &
PYTHONUNBUFFERED=1 nohup python finetune_regression.py --dataset freesolv --gpu 6 --bs 31 > logs/finetune_nopre_freesolv.log 2>&1 &
PYTHONUNBUFFERED=1 nohup python finetune_regression.py --dataset lipo --gpu 7 > logs/finetune_nopre_lipo.log 2>&1 &
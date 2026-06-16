#!/bin/bash
runname="mypre"
mkdir -p outputs/logs
PYTHONUNBUFFERED=1 nohup python finetune.py --dataset tox21 --gpu 0 > outputs/logs/finetune_tox21_$runname.log 2>&1 &
PYTHONUNBUFFERED=1 nohup python finetune.py --dataset toxcast --gpu 1 > outputs/logs/finetune_toxcast_$runname.log 2>&1 &
PYTHONUNBUFFERED=1 nohup python finetune.py --dataset bbbp --gpu 2 > outputs/logs/finetune_bbbp_$runname.log 2>&1 &
PYTHONUNBUFFERED=1 nohup python finetune.py --dataset sider --gpu 3 > outputs/logs/finetune_sider_$runname.log 2>&1 &
PYTHONUNBUFFERED=1 nohup python finetune.py --dataset clintox --gpu 4 > outputs/logs/finetune_clintox_$runname.log 2>&1 &
# PYTHONUNBUFFERED=1 nohup python finetune.py --dataset muv --gpu 5 > outputs/logs/finetune_muv_$runname.log 2>&1 &
PYTHONUNBUFFERED=1 nohup python finetune.py --dataset bace --gpu 6 > outputs/logs/finetune_bace_$runname.log 2>&1 &
# PYTHONUNBUFFERED=1 nohup python finetune.py --dataset hiv --gpu 7 > outputs/logs/finetune_hiv_$runname.log 2>&1 &

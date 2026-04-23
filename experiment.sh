#!/bin/bash
runname="mypre"
PYTHONUNBUFFERED=1 nohup python finetune.py --dataset tox21 --gpu 0 > logs/finetune_tox21_$runname.log 2>&1 &
PYTHONUNBUFFERED=1 nohup python finetune.py --dataset toxcast --gpu 1 > logs/finetune_toxcast_$runname.log 2>&1 &
PYTHONUNBUFFERED=1 nohup python finetune.py --dataset bbbp --gpu 2 > logs/finetune_bbbp_$runname.log 2>&1 &
PYTHONUNBUFFERED=1 nohup python finetune.py --dataset sider --gpu 3 > logs/finetune_sider_$runname.log 2>&1 &
PYTHONUNBUFFERED=1 nohup python finetune.py --dataset clintox --gpu 4 > logs/finetune_clintox_$runname.log 2>&1 &
# PYTHONUNBUFFERED=1 nohup python finetune.py --dataset muv --gpu 5 > logs/finetune_muv_$runname.log 2>&1 &
PYTHONUNBUFFERED=1 nohup python finetune.py --dataset bace --gpu 6 > logs/finetune_bace_$runname.log 2>&1 &
# PYTHONUNBUFFERED=1 nohup python finetune.py --dataset hiv --gpu 7 > logs/finetune_hiv_$runname.log 2>&1 &
#!/usr/bin/env bash
script_path=$(cd `dirname $0`; pwd)
cd $script_path
max_epoch=10
CUDA_VISIBLE_DEVICES=1 python main.py --phase 'train' --name 'demo' --max_epoch ${max_epoch}



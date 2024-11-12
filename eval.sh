#!/usr/bin/env bash
script_path=$(cd `dirname $0`; pwd)
cd $script_path

in_data_dir=/home/tsmc/teamip/lyz/data/PU1K/test/input_2048/input_2048
gt_data_dir=/home/tsmc/teamip/lyz/data/PU1K/test/input_2048/gt_8192

num_shape_point=2048

Model="./model/release/model.pth"

cd ./evaluation_code
bash eval_pu1k.sh
cd $script_path

CUDA_VISIBLE_DEVICES=1 python main.py --phase eval --ckpt ${Model} --num_shape_point ${num_shape_point} --test_data  ${in_data_dir} --gt_path  ${gt_data_dir}




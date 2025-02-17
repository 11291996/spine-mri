#!/bin/bash
data_dir=/data/datasets/spine/gtu/train
#this is the default value
original_modal=t1
target_modal=t2
model_dir=/data/model/cdm/spine/gtu/model_t1_t2.pth
#must be sure about the datasets directory in depending on the server
datasets_dir=/data/datasets

#these commands must be run in the order at the first time
/home/jaewan/spine-diff/spine-venv/bin/python3 -m src.cdm.cdm_mrm --data_dir $data_dir --original_modal $original_modal --target_modal $target_modal
/home/jaewan/spine-diff/spine-venv/bin/python3 -m src.cdm.cdm_mdn --data_dir $data_dir
/home/jaewan/spine-diff/spine-venv/bin/python3 -m src.cdm.cdm_cunet --data_dir $data_dir
/home/jaewan/spine-diff/spine-venv/bin/python3 -m src.cdm.cdm_score --model_dir $model_dir --datasets_dir $datasets_dir
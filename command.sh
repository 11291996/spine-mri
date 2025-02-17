#!/bin/bash

input_model_dir=/data/model/cdm/brain/brat2024/model_t1_t2.pth
target_model_dir=/data/model/cdm/spine/gtu/model_t1_t2.pth

/home/jaewan/spine-diff/spine-venv/bin/python3 -m subject_free_cdm --input_model_dir $input_model_dir --target_model_dir $target_model_dir
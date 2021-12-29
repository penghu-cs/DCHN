#!/usr/bin/env bash
available_num=100
output_shape=64
gama=1
# mirflickr25k nus_wide_tc21 MSCOCO_doc2vec XMedia nus_wide_tc10 IAPR-TC12
datasets=mirflickr25k
alpha=0.02
gama=1
seed=0
epochs=100

# train SHAM
python trainSHAM.py --datasets $datasets --output_shape $output_shape --gama $gama --available_num $available_num

# train image
python main_DCHN.py --seed $seed --epochs $epochs --view 0 --datasets $datasets --output_shape $output_shape --alpha $alpha --gama $gama --available_num $available_num --gpu_id 0 &

# train Text
python main_DCHN.py --seed $seed --epochs $epochs --view 1 --datasets $datasets --output_shape $output_shape --alpha $alpha --gama $gama --available_num $available_num --gpu_id 1 &

# python main_DCHN.py --view -1 --datasets $datasets --output_shape $output_shape --alpha $alpha --gama $gama --available_num $available_num --gpu_id -1 --multiprocessing

wait
# eval
python main_DCHN.py --epochs $epochs --view -1 --datasets $datasets --output_shape $output_shape --alpha $alpha --gama $gama --available_num $available_num --gpu_id 0 --mode eval --num_workers 0

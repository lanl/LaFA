#!/bin/bash

# Implicit with batching L2
python main.py --dataset MNIST --rank 10 \
                --base_iter 10000 \
                --nmf_iter 1000 \
                --implicit \
                --iterative \
                --no_iter 40 --taylor 25 --norm L2 \
                --eps_min 0.00 --eps_max 0.02 --alpha 0.01 --no_eps 10 \
                --seed 2711 \
                --batch 1000 \
                --no_batch 10

# Implicit with batching Linf
python main.py --dataset MNIST --rank 10 \
                --base_iter 10000 \
                --nmf_iter 1000 \
                --implicit \
                --iterative \
                --no_iter 40 --taylor 25 --norm Linf \
                --eps_min 0.00 --eps_max 0.01 --alpha 0.01 --no_eps 10 \
                --seed 2711 \
                --batch 1000 \
                --no_batch 10

# Backprop with no batching L2
python main.py --dataset MNIST --rank 10 \
                --base_iter 10000 \
                --nmf_iter 100 \
                --iterative \
                --no_iter 40 --taylor 25 --norm L2 \
                --eps_min 0.00 --eps_max 0.02 --alpha 0.01 --no_eps 10 \
                --seed 2711 \
                --batch 1000 \
                --no_batch 1

# Backprop with no batching Linf
python main.py --dataset MNIST --rank 10 \
                --base_iter 10000 \
                --nmf_iter 100 \
                --iterative \
                --no_iter 40 --taylor 25 --norm Linf \
                --eps_min 0.00 --eps_max 0.01--alpha 0.01 --no_eps 10 \
                --seed 2711 \
                --batch 1000 \
                --no_batch 1

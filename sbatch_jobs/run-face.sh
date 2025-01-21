#!/bin/bash

python main.py --dataset Face --rank 5 \
                --base_iter 10000 --nmf_iter 2000 \
                --implicit \
                --iterative \
                --no_iter 40 --taylor 100 --norm L2 \
                --eps_min 0.0 --eps_max 0.02 --alpha 0.01 --no_eps 20 \
                --seed 2711

python main.py --dataset Face --rank 5 \
                --base_iter 10000 --nmf_iter 2000 \
                --iterative \
                --no_iter 40 --taylor 100 --norm L2 \
                --eps_min 0.0 --eps_max 0.02 --alpha 0.01 --no_eps 20 \
                --seed 2711

python main.py --dataset Face --rank 5 \
                --base_iter 10000 --nmf_iter 2000 \
                --implicit \
                --iterative \
                --no_iter 40 --taylor 100 --norm Linf \
                --eps_min 0.0 --eps_max 0.01 --alpha 0.01 --no_eps 20 \
                --seed 2711

python main.py --dataset Face --rank 5 \
                --base_iter 10000 --nmf_iter 2000 \
                --iterative \
                --no_iter 40 --taylor 100 --norm Linf \
                --eps_min 0.0 --eps_max 0.01 --alpha 0.01 --no_eps 20 \
                --seed 2711
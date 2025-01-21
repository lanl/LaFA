#!/bin/bash

python main.py --dataset WTSI --rank 4 \
                --base_iter 5000 --nmf_iter 1600 \
                --implicit --iterative \
                --no_iter 40 --taylor 80 --norm Linf \
                --eps_min 0.0 --eps_max 0.001 --alpha 0.01 --no_eps 20 \
                --seed 2711

python main.py --dataset WTSI --rank 4 \
                --base_iter 5000 --nmf_iter 1600 \
                --iterative \
                --no_iter 40 --taylor 80 --norm Linf \
                --eps_min 0.0 --eps_max 0.001 --alpha 0.01 --no_eps 20 \
                --seed 2711

python main.py --dataset WTSI --rank 4 \
                --base_iter 5000 --nmf_iter 1600 \
                --implicit --iterative \
                --no_iter 40 --taylor 80 --norm L2 \
                --eps_min 0.0 --eps_max 0.02 --alpha 0.01 --no_eps 10 \
                --seed 2711

python main.py --dataset WTSI --rank 4 \
                --base_iter 5000 --nmf_iter 1600 \
                --iterative \
                --no_iter 40 --taylor 80 --norm L2 \
                --eps_min 0.0 --eps_max 0.02 --alpha 0.01 --no_eps 20 \
                --seed 2711
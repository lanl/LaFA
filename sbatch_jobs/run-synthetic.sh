#!/bin/bash

# PGD attacks for Linf/l2 of Implicit and backpropagate methods
# The script will write on 4 files. To have different initial states, you can change the seed
# Recommend to run about 20 times

# python main.py --dataset Synthetic --rank 3 \
#                 --implicit --iterative \
#                 --no_iter 40 --taylor 100 --norm Linf \
#                 --eps_min 0.0 --eps_max 0.04 --alpha 0.01 --no_eps 20 \
#                 --seed 2711

# python main.py --dataset Synthetic --rank 3 \
#                 --iterative \
#                 --no_iter 40 --taylor 100 --norm Linf \
#                 --eps_min 0.0 --eps_max 0.04 --alpha 0.01 --no_eps 20 \
#                 --seed 2711

# python main.py --dataset Synthetic --rank 3 \
#                 --implicit --iterative \
#                 --no_iter 40 --taylor 100 --norm L2 \
#                 --eps_min 0.0 --eps_max 0.05 --alpha 0.01 --no_eps 20 \
#                 --seed 2711

# python main.py --dataset Synthetic --rank 3 \
#                 --iterative \
#                 --no_iter 40 --taylor 100 --norm L2 \
#                 --eps_min 0.0 --eps_max 0.05 --alpha 0.01 --no_eps 20 \
#                 --seed 2711

python main.py --dataset Synthetic --rank 3 \
                --base_iter 2000 --nmf_iter 500 \
                --implicit --iterative \
                --no_iter 40 --taylor 80 --norm Linf \
                --eps_min 0.0 --eps_max 0.05 --alpha 0.01 --no_eps 20 \
                --seed 11

python main.py --dataset Synthetic --rank 3 \
                --base_iter 2000 --nmf_iter 200 \
                --iterative \
                --no_iter 40 --taylor 80 --norm Linf \
                --eps_min 0.0 --eps_max 0.05 --alpha 0.01 --no_eps 20 \
                --seed 11

python main.py --dataset Synthetic --rank 3 \
                --base_iter 2000 --nmf_iter 500 \
                --implicit --iterative \
                --no_iter 40 --taylor 80 --norm L2 \
                --eps_min 0.0 --eps_max 0.05 --alpha 0.01 --no_eps 20 \
                --seed 11

python main.py --dataset Synthetic --rank 3 \
                --base_iter 2000 --nmf_iter 200 \
                --iterative \
                --no_iter 40 --taylor 80 --norm L2 \
                --eps_min 0.0 --eps_max 0.05 --alpha 0.01 --no_eps 20 \
                --seed 11

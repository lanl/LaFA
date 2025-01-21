#!/bin/bash
for seed in {2711..2731}; do
    python main.py --dataset WTSIPCA --rank 2 \
                    --base_iter 5000 --nmf_iter 1600 \
                    --iterative \
                    --no_iter 40 --taylor 80 --norm L2 \
                    --eps_min 0.0 --eps_max 0.02 --alpha 0.01 --no_eps 50 \
                    --seed $seed
done
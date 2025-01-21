#!/bin/bash

# Directory to store the job scripts
mkdir -p job_scripts
cd job_scripts

# Parameters for the experiments
methods=("" "--implicit")
norms=("Linf" "L2")
eps_max_for_Linf="0.01"
eps_max_for_L2="0.02"
no_eps="10"
alpha="0.01"
no_iter="40"
taylor="25"
dataset="MNIST"
rank="10"
base_iter="10000"
nmf_iter_implicit="1000"
nmf_iter_no_implicit="100"
batch_size="1000"
no_batch_implicit="10"
no_batch_no_implicit="1"

# Generate job scripts for each seed
for seed in {2711..2720}; do
    job_file="job_${seed}.sh"
    echo "#!/bin/bash" > $job_file
    echo "#SBATCH --job-name=experiment_${seed}" >> $job_file
    echo "#SBATCH --output=experiment_${seed}_%j.out" >> $job_file
    echo "#SBATCH --error=experiment_${seed}_%j.err" >> $job_file
    echo "#SBATCH --partition=gpu" >> $job_file
    echo "#SBATCH --nodes=1" >> $job_file
    echo "#SBATCH --time=16:00:00" >> $job_file
    echo "#SBATCH -A w24_unsupgan_g" >> $job_file
    echo "source activate /usr/projects/slic/manish/llm-env" >> $job_file
    echo "cd /usr/projects/unsupgan/mvu_experiments/NMF_attacks/" >> $job_file

    # Configure different experiments for each GPU
    gpu_id=0
    for method in "${methods[@]}"; do
        nmf_iter="$nmf_iter_no_implicit"
        no_batch="$no_batch_no_implicit"
        if [[ "$method" == "--implicit" ]]; then
            nmf_iter="$nmf_iter_implicit"
            no_batch="$no_batch_implicit"
        fi

        for norm in "${norms[@]}"; do
            eps_max="${eps_max_for_Linf}"
            if [[ "$norm" == "L2" ]]; then
                eps_max="${eps_max_for_L2}"
            fi

            echo "CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset $dataset --rank $rank $method --iterative --base_iter $base_iter --nmf_iter $nmf_iter --no_iter $no_iter --taylor $taylor --norm $norm --eps_min 0.0 --eps_max $eps_max --alpha $alpha --no_eps $no_eps --seed $seed --batch $batch_size --no_batch $no_batch &" >> $job_file
            ((gpu_id++))
        done
    done

    echo "wait" >> $job_file
    echo "echo 'All processes done.'" >> $job_file
    chmod +x $job_file
    sbatch $job_file
done


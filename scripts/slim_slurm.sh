#!/bin/bash

#SBATCH --job-name=test-g1
#SBATCH --time=2-00:00:00

# Slurm parameters for Nexus cluster
# SBATCH --mem=16G # RAM
# SBATCH --mincpus=1
# SBATCH --output=/home/users/afernandes/Repositories/wsr_tmp/job_out_%A.out

# SBATCH --gres=shard:0 # GB of gpu VRAM
# SBATCH --exclude=nexus1,nexus2,nexus4 # we can exclude machines if we want

# Slurm parameters for INCD cluster
# SBATCH --partition=gpu
# SBATCH --output=/users5/vlabist/afonsofernandes/Repositories/wsr_tmp/job_out_%A.out
# SBATCH --gres=gpu:a100

# Slurm parameters for HLT Ada cluster
#SBATCH --mem=32G # RAM
#SBATCH --mincpus=10
#SBATCH --output=/home/u024151/Repositories/wsr_tmp/job_out_%A.out

# Activate the Conda environment
eval "$(conda shell.bash hook)"
#conda activate /home/users/afernandes/anaconda3/envs/wsr
#conda activate /users5/vlabist/afonsofernandes/anaconda3/envs/wsr
conda activate /home/u024151/anaconda3/envs/wsr

# To execute this slurm script: sbatch slim_slurm.sh -c 'test_sim'|'train'|'gen_data'
NUMBER_OF_CORES=32

# Read arguments and execute program
while getopts c: flag
do
    case "${flag}" in
        c) script=${OPTARG};;
    esac
done

if [[ -z $script ]]; then
    script="test"
fi

if [[ "$script" == "train" ]]; then
    bash scripts/train.sh
elif [[ "$script" == "hyperparam_optim" ]]; then
    bash scripts/hyperparam_optim.sh
elif [[ "$script" == "test" ]]; then
    bash scripts/test_sim.sh -nc "$NUMBER_OF_CORES"
elif [[ "$script" == "gen_data" ]]; then
    bash scripts/gen_data.sh
else
    echo "Unknown bash script: $script"
fi

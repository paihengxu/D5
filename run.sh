#!/usr/bin/env bash

#SBATCH --time=8:00:00                                         # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --qos=default                                           # set QOS, this will determine what resources can be requested
#SBATCH --account=clip
#SBATCH --partition=clip
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtxa6000:1

source /nfshomes/paiheng/.bashrc
conda activate opend5

command="${1}"

# Execute the command passed as an argument
echo "Executing: ${command}"
eval ${command}


conda deactivate

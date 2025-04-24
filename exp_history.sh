#!/usr/bin/env bash

# Dec 17, 2024

#python run_edu_problems.py --find_representative --output_path simse_test_3B.pkl 2>&1 | tee log_dir/simse_test_3B.log
#
#python run_edu_problems.py --output_path simse_test.pkl 2>&1 | tee log_dir/simse_test.log


# Jan 1, 2025

#for var in 'Objective' 'Unpacking' 'Self-Instruction' 'Self-Regulation' 'Ending'
#do
#    sbatch --job-name simse_${var} --output log_dir/simse_${var}.log run.sh ${var}
#done

# Jan 2, 2025 - rerun the treatment experiment with metacognitive strategies included in the prompt
#var="Treatment"
#sbatch --job-name simse_${var} --output log_dir/simse_${var}.log run.sh ${var}

# Jan 3, 2025 - training group only contain high quality samples
#var="Treatment_high"
#python run_edu_problems.py --output_path results/simse_${var}.pkl --simse_var ${var}

# Mar 5, 2025 - run experiments with the gcl dataset
#setup="gcl"
#python run_gcl_problem.py --output_path results/gcl.pkl 2>&1 --setup ${setup} | tee log_dir/gcl.log

#srun --job-name gcl_diff --output log_dir/gcl_diff.log --partition=clip --account=clip --gres=gpu:rtxa6000:1 --mem=32g --time=5:00:00 bash -c "source /nfshomes/paiheng/.bashrc && conda activate opend5 && python run_gcl_problem.py --output_path results/gcl_diff.pkl --setup diff"

# Mar 12, 2025 - rerun the treatment experiment with treatment vs. non-experimental
#var="Treatment_non_experimental"
#sbatch --job-name simse_${var} --output log_dir/simse_${var}.log run.sh ${var}

# Apr 17, 2025 - rerun gcl experiments
srun --job-name gcl_diff --output log_dir/gcl_diff.log --partition=clip --account=clip --gres=gpu:rtxa6000:1 --mem=32g --time=5:00:00 bash -c "source /nfshomes/paiheng/.bashrc && conda activate opend5 && python run_gcl_problem.py --output_path results/gcl_diff.pkl --setup diff"
srun --job-name gcl --output log_dir/gcl.log --partition=clip --account=clip --gres=gpu:rtxa6000:1 --mem=32g --time=5:00:00 bash -c "source /nfshomes/paiheng/.bashrc && conda activate opend5 && python run_gcl_problem.py --output_path results/gcl.pkl --setup gcl"
srun --job-name non_gcl --output log_dir/non_gcl.log --partition=clip --account=clip --gres=gpu:rtxa6000:1 --mem=32g --time=5:00:00 bash -c "source /nfshomes/paiheng/.bashrc && conda activate opend5 && python run_gcl_problem.py --output_path results/non_gcl.pkl --setup non_gcl"

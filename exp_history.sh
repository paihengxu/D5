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
var="Treatment_high"
python run_edu_problems.py --output_path results/simse_${var}.pkl --simse_var ${var}
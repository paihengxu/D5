#!/usr/bin/env bash

# Create results directory if it doesn't exist
mkdir -p results

# Define the setups, percentiles, and target options
SETUPS=("gcl" "diff" "non_gcl" "gcl_cf" "diff_cf" "non_gcl_cf")
PERCENTILES=(10 20)
TARGETS=("original" "new")

# Loop through all combinations and submit jobs
for setup in "${SETUPS[@]}"; do
    for percentile in "${PERCENTILES[@]}"; do
        for target in "${TARGETS[@]}"; do
            # Define output file name based on parameters
            if [ "$target" == "new" ]; then
                target_flag="--new_target"
                output_name="${setup}_p${percentile}_new_target"
            else
                target_flag=""
                output_name="${setup}_p${percentile}"
            fi

            # Create output directory for this run
            output_dir="results/${output_name}"
            mkdir -p "$output_dir"

            # Define output files
            output_pkl="${output_dir}/${output_name}.pkl"
            output_csv="${output_dir}/${output_name}.csv"
            log_file="${output_dir}/${output_name}.log"

            # Submit the job
            echo "Submitting job for setup=$setup, percentile=$percentile, target=$target"
            sbatch --job-name="${output_name}" \
                   --output="${log_file}" \
                   run.sh "python run_gcl_problem_v2.py --setup ${setup} --percentile ${percentile} ${target_flag} --output_path ${output_pkl}"

            # Wait a bit to avoid overwhelming the scheduler
            sleep 1
        done
    done
done

echo "All jobs submitted. Check the status with 'squeue -u $USER'"

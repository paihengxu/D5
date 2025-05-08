#!/usr/bin/env bash

# This script runs all experimental setups sequentially on a single node
# instead of submitting multiple SLURM jobs in parallel.

# Usage:
#   ./run_sequential.sh [--setup SETUP] [--percentile PERCENTILE] [--target TARGET] [--subsample N]
#
# Options:
#   --setup SETUP       Only run the specified setup (e.g., gcl, diff, non_gcl, gcl_cf, diff_cf, non_gcl_cf)
#   --percentile PERC   Only run the specified percentile (10 or 20)
#   --target TARGET     Only run the specified target (original or new)
#   --subsample N       Add --subsample N to all commands (to reduce sample size)
#
# Examples:
#   ./run_sequential.sh                     # Run all combinations
#   ./run_sequential.sh --setup gcl         # Run only gcl setup with all percentiles and targets
#   ./run_sequential.sh --subsample 100     # Run all combinations with --subsample 100

# Create results directory if it doesn't exist
mkdir -p results_gcl

# Parse command line arguments
FILTER_SETUP=""
FILTER_PERCENTILE=""
FILTER_TARGET=""
SUBSAMPLE_ARG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --setup)
      FILTER_SETUP="$2"
      shift 2
      ;;
    --percentile)
      FILTER_PERCENTILE="$2"
      shift 2
      ;;
    --target)
      FILTER_TARGET="$2"
      shift 2
      ;;
    --subsample)
      SUBSAMPLE_ARG="--subsample $2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set up environment
source /nfshomes/paiheng/.bashrc
conda activate opend5

# Define the setups, percentiles, and target options
if [ -z "$FILTER_SETUP" ]; then
  SETUPS=("gcl" "diff" "non_gcl" "gcl_cf" "diff_cf" "non_gcl_cf")
else
  SETUPS=("$FILTER_SETUP")
fi

if [ -z "$FILTER_PERCENTILE" ]; then
  PERCENTILES=(20)
else
  PERCENTILES=("$FILTER_PERCENTILE")
fi

if [ -z "$FILTER_TARGET" ]; then
  TARGETS=("original" "new")
else
  TARGETS=("$FILTER_TARGET")
fi

# Count total number of runs
TOTAL_RUNS=$((${#SETUPS[@]} * ${#PERCENTILES[@]} * ${#TARGETS[@]}))
CURRENT_RUN=0

echo "Starting sequential execution of $TOTAL_RUNS experimental setups"
echo "========================================================"

# Loop through all combinations and run sequentially
for setup in "${SETUPS[@]}"; do
    for percentile in "${PERCENTILES[@]}"; do
        for target in "${TARGETS[@]}"; do
            # Update counter
            CURRENT_RUN=$((CURRENT_RUN + 1))

            # Define output file name based on parameters
            if [ "$target" == "new" ]; then
                target_flag="--new_target"
                output_name="${setup}_p${percentile}_new_target"
            else
                target_flag=""
                output_name="${setup}_p${percentile}"
            fi

            # Create output directory for this run
            output_dir="results_gcl/${output_name}"
            mkdir -p "$output_dir"

            # Define output files
            output_pkl="${output_dir}/${output_name}.pkl"
            output_csv="${output_dir}/${output_name}.csv"
            log_file="${output_dir}/${output_name}.log"

            # Log start time
            echo "[$CURRENT_RUN/$TOTAL_RUNS] Starting run for setup=$setup, percentile=$percentile, target=$target"
            echo "Output will be saved to: $output_pkl"
            start_time=$(date +"%Y-%m-%d %H:%M:%S")
            echo "Start time: $start_time"

            # Check if output file already exists
            if [ -f "$output_pkl" ]; then
                echo "[$CURRENT_RUN/$TOTAL_RUNS] Skipping run for setup=$setup, percentile=$percentile, target=$target"
                echo "Output file already exists: $output_pkl"
                echo "========================================================"
                continue
            fi

            # Execute the command and redirect output to log file
            echo "=======================================================" > "$log_file"
            echo "Running setup=$setup, percentile=$percentile, target=$target" >> "$log_file"
            echo "Start time: $start_time" >> "$log_file"
            echo "=======================================================" >> "$log_file"

            # Run the command
            python run_gcl_problem_v2.py --setup ${setup} --percentile ${percentile} ${target_flag} ${SUBSAMPLE_ARG} --output_path ${output_pkl} 2>&1 | tee -a "$log_file"

            # Log end time
            end_time=$(date +"%Y-%m-%d %H:%M:%S")
            echo "End time: $end_time" | tee -a "$log_file"
            echo "=======================================================" | tee -a "$log_file"
            echo "" | tee -a "$log_file"

            # Calculate and display progress
            progress=$((CURRENT_RUN * 100 / TOTAL_RUNS))
            echo "Progress: $progress% ($CURRENT_RUN/$TOTAL_RUNS completed)"
            echo "========================================================"
        done
    done
done

# Deactivate conda environment
conda deactivate

echo "All runs completed successfully!"
echo "Results are stored in the 'results_gcl' directory."

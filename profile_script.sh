#!/bin/bash

# Script for running performance counter measurements on prediction.py for different scenarios (benign and backdoor)
# across a range of image classes and indices.

# Function: run_perf_measurements
# Purpose: Run performance counter measurements for a given scenario type
# Parameters:
#   type - A string indicating the scenario type (either 'benign' or 'backdoor')
#   venv_name - Virtual environment name

run_perf_measurements() {
    local type=$1
    local venv_name=$2
    local log_file="perf_${type}.log"

    # Loop over image classes and indices
    for image_class in {0..9}; do
        for index in {0..999}; do
            # Display current scenario details
            echo "${type}" "class" "${image_class}" "index" "${index}"

            # Run performance measurements using 'perf' tool
            # '-C 0' specifies CPU 0 for the measurements
            # '-e' lists the events to measure (branches, cache references, etc.)
            # '-r 10' repeats the measurement 10 times for averaging
            # 'taskset -c 0' runs the command on a specific CPU (CPU 0)
            sudo perf stat -C 0 -e branches,branch-misses,cache-references,cache-misses,instructions -r 10 \
                -o "${log_file}" --append taskset -c 0 "${venv_name}"/bin/python prediction.py --type="${type}" \
                --image_class="${image_class}" --index="${index}"
        done
    done
}

# Run measurements for both benign and backdoor scenarios
run_perf_measurements benign "$1"
run_perf_measurements backdoor "$1"

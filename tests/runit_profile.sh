#!/bin/bash
source ../setup_env.sh
#python -m cProfile myscript.py
#python -m cProfile [-o output_file] [-s sort_order] myscript.py
#nvprof --analysis-metrics -o prof_150k_10t.nvvp python3 generate_x_samples.py 2>&1 | tee /tmp/log.txt #smcloc_150kp_10step_nvprofile.out
#--print-gpu-summary
vprof --analysis-metrics -o prof_150k_2t.nvvp python3 generate_x_samples.py 2>&1 | tee /tmp/log_150k_2t.txt #smcloc_150kp_10step_nvprofile.out
#nvprof --analysis-metrics -o smcloc.nvprof python3 generate_x_samples.py 2>&1 | tee /tmp/log.txt
#python -m cProfile generate_x_samples.py | tee smcloc_10000p_50step_profile.out
#python3 -m cProfile -o smcloc_profile.prof generate_x_samples.py 2>&1 | tee smcloc_10000p_50step_profile.out

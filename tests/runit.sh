#!/bin/bash

source ./setup_env.sh
ulimit -s unlimited
ulimit -c unlimited

#export GPUGB_PER_PROC=$(echo print 0.9/${NP}. | python)
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=6
export TF_CPP_MIN_VLOG_LEVEL=0
#export OMP_SCHEDULE=dynamic #guided,static,auto,dynami
#export OMP_PLACES=threads
export TF_FP16_CONV_USE_FP32_COMPUTE=0
export TF_FP16_MATMUL_USE_FP32_COMPUTE=0
#export HOROVOD_TIMELINE=${SCRATCH_PAD}/timeline.json
#export HOROVOD_FUSION_THRESHOLD=256000000
#export HOROVOD_FUSION_THRESHOLD=0

python3 generate_x_samples.py 2>&1 | tee /tmp/log.txt
mv /tmp/log.txt .

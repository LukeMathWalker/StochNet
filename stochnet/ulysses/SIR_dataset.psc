#!/bin/bash

#PBS -l nodes=1:ppn=20
#PBS -l walltime=01:00:00
#PBS -q regular
#PBS -N SIR_dataset_generator
#PBS -k eo
#PBS -V

source activate tf_1.0_cpu
cd /home/lpalmier/workspace/StochNet/stochnet/dataset/

python SIR_dataset.py

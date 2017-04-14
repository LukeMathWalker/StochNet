#!/bin/bash

#PBS -l nodes=1:ppn=20
#PBS -l walltime=10:30:00
#PBS -q regular
#PBS -N New_SIR_model_tuning_Deep_MNC
#PBS -k eo
#PBS -V

source activate tf_1.0_cpu
cd /home/lpalmier/workspace/StochNet/stochnet/applicative/

python SIR_model_tuning_Deep_MNC.py

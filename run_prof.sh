#!/bin/sh
#SBATCH -p gpuK80
#SBATCH --gres=gpu:1
#logs stuff

make targ=$1 && srun -p gpuK80 --gres=gpu:1 nvprof ./$1 

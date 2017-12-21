#!/bin/sh
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#logs stuff

make targ=$1 && srun -p gpu --gres=gpu:1 ./$1 

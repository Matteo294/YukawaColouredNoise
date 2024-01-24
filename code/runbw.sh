#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=1000mb
#SBATCH -J test
#SBATCH --partition=gpu_8		# dev_gpu_4, gpu_4, or gpu_8
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR2@60		# send SIGNAL to the code to wrap up 60 seconds before killing it

module load compiler/gnu/10.2
module load devel/cuda/11.4
module load lib/hdf5/1.12.0-gnu-8.3

#without 'exec' here, the signal is not passed to our executable /=
exec ./out input.toml

#!/bin/bash

#SBATCH -N1
#SBATCH -n2
#SBATCH --account=gx219
#SBATCH --gres=gpu:1
#SBATCH -p aquila
#SBATCH -t 1-00:00:00
#SBATCH --mem=16000

/scratch/ry649/miniconda3/bin/python gen_rhythm.py

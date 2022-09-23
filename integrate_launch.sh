#!/bin/bash
#SBATCH -p shared    # partition (queue)
#SBATCH -n 12         # 8 cores
#SBATCH --mem 96G    # memory pool for all cores
#SBATCH -t 1-0:00   # time (D-HH:MM)
#SBATCH -o integrate_%j.log              # Standard output
#SBATCH -e integrate_%j.err              # Standard error

cctbx.python integrate.py $1
#!/bin/bash
#SBATCH -p shared    # partition (queue)
#SBATCH -n 12         # 8 cores
#SBATCH --mem 96G    # memory pool for all cores
#SBATCH -t 1-0:00   # time (D-HH:MM)
#SBATCH -o logs/integrate_%j.log              # Standard output
#SBATCH -e logs/integrate_%j.err              # Standard error

DIALS_PATH=${2:-'/n/hekstra_lab/people/brookner/dials-v3-11-1/dials_env.sh'}

source $DIALS_PATH

cctbx.python integrate.py $1
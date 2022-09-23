#!/bin/bash
#SBATCH -p shared    # partition (queue)
#SBATCH -n 12         # 8 cores
#SBATCH --mem 96G    # memory pool for all cores
#SBATCH -t 1-0:00   # time (D-HH:MM)
#SBATCH -o int_test_%j.log              # Standard output
#SBATCH -e int_test_%j.err              # Standard error

cctbx.python int_test.py $1
cctbx.python stills2mtz.py $1
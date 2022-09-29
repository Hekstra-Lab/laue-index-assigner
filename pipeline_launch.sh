#!/bin/bash
#SBATCH -p shared    # partition (queue)
#SBATCH -n 12         # 8 cores
#SBATCH --mem 96G    # memory pool for all cores
#SBATCH -t 1-0:00   # time (D-HH:MM)
#SBATCH -o logs/launch_ons_%j.log              # Standard output
#SBATCH -e logs/launch_ons_%j.err              # Standard error

bash pipeline_with_variables.sh e011f_640ns_ dials_files_640ns_f 4
bash pipeline_with_variables.sh e011g_640ns_ dials_files_640ns_g 2
bash pipeline_with_variables.sh e011h_640ns_ dials_files_640ns_h 1
bash pipeline_with_variables.sh e011f_200ns_ dials_files_200ns_f 4
bash pipeline_with_variables.sh e011g_200ns_ dials_files_200ns_g 2
bash pipeline_with_variables.sh e011h_200ns_ dials_files_200ns_h 1
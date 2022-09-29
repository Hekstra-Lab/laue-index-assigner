#!/bin/bash
#SBATCH --job-name=gpu_careless_integ
#SBATCH -p gpu_requeue,seas_gpu_requeue # partition (queue)
#SBATCH --mem 32G # memory pool for all cores
#SBATCH -t 0-8:00 # time (D-HH:MM)
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH -o logs/test_careless_integrate_off_%j.out        # Standard output
#SBATCH -e logs/test_careless_integrate_off_%j.err        # Standard error

module load cuda/11.1.0-fasrc01 cudnn/8.1.0.77_cuda11.2-fasrc01
source /n/hekstra_lab/people/brookner/miniconda3/etc/profile.d/conda.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/hekstra_lab/people/brookner/miniconda3/lib/

conda activate carelessminimal

out=careless_off_all_integrate
mkdir $out

BASE_ARGS=(
  --iterations=10_000
  --refine-uncertainties 
  --wavelength-key='wavelength' 
  --studentt-likelihood-dof=32
  # "xobs,yobs,wavelength,dHKL,BATCH"  # int_test version
  "Hobs,Kobs,Lobs,xcal,ycal,wavelength,dHKL,BATCH" # integrate version

)

careless poly \
  ${BASE_ARGS[@]} \
  dials_files_off_*/integrated_from_integrate.mtz \
  $out/merged_off_hsDHFR_

DURATION=$SECONDS
MESSAGE="Job $SLURM_JOB_ID:careless finished on $HOSTNAME in $(($DURATION / 60)) minutes."
echo $MESSAGE

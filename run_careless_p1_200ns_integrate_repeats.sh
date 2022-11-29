#!/bin/bash
#SBATCH --job-name=gpu_p1_careless
#SBATCH -p gpu_requeue,seas_gpu_requeue # partition (queue)
#SBATCH --mem 32G # memory pool for all cores
#SBATCH -t 0-12:00 # time (D-HH:MM)
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH -o logs/careless_p1_%j.out        # Standard output
#SBATCH -e logs/careless_p1_%j.err        # Standard error

# Do this interactively, I think??
# module load cuda/11.1.0-fasrc01 cudnn/8.1.0.77_cuda11.2-fasrc01
# source /n/hekstra_lab/people/brookner/miniconda3/etc/profile.d/conda.sh
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/hekstra_lab/people/brookner/miniconda3/lib/

# conda activate cl

out1=careless_200ns_p1_integrate_repeats
mkdir $out1

BASE_ARGS=(
  --iterations=10_000 
  --wavelength-key='wavelength' 
  --studentt-likelihood-dof=32 
  --merge-half-datasets 
  --half-dataset-repeats=5
  # "Hobs,Kobs,Lobs,xobs,yobs,wavelength,dHKL,BATCH"  # int_test version
  "Hobs,Kobs,Lobs,xcal,ycal,wavelength,dHKL,BATCH" # integrate version
)

careless poly \
  ${BASE_ARGS[@]} \
  concat_careless_inputs/concat_200ns_p1_integrate.mtz \
  $out1/merged_p1_200ns_hsDHFR_repeats

DURATION=$SECONDS
MESSAGE="Job $SLURM_JOB_ID:careless finished on $HOSTNAME in $(($DURATION / 60)) minutes."
echo $MESSAGE
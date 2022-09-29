#!/bin/bash
############################################################################
# User-defined global parameters
DIALS_PATH='/n/hekstra_lab/people/brookner/dials-v3-11-1/dials_env.sh'
INSTALL_RS=0

DIFF_IMG_DIR='/n/holyscratch01/hekstra_lab/brookner/lauescr/hsDHFR/images/'

SPACE_GROUP_NUMBER=20
CELL='"87.897,94.559,96.265,90.000,90.000,90.000"'

PIXEL_MASK=0

USE_SLURM_FOR_INTEGRATION=1
# End user-defined global parameters
############################################################################
# User-defined variables to change for each run
IMG_PREFIX=$1 # e.g. 'e011e_off_'
OUT_DIR=$2 # e.g. 'dials_files_off_e', or whatever naming convention you prefer
OSCILLATION=$3 # The angle between consecutive images, in degrees
# End user-defined variables to change for each run
############################################################################

source $DIALS_PATH

if [ $INSTALL_RS -eq 1 ]
then
  cctbx.python -m pip install reciprocalspaceship
fi

mkdir $OUT_DIR

bash scan_varying_indexer.sh ${OUT_DIR} ${DIFF_IMG_DIR} ${IMG_PREFIX} ${OSCILLATION} ${SPACE_GROUP_NUMBER} ${CELL} ${PIXEL_MASK}
cctbx.python sequence_to_stills_no_sb.py output.experiments=${OUT_DIR}/stills_no_sb.expt output.reflections=${OUT_DIR}/stills_no_sb.refl ${OUT_DIR}/refined_varying.*
cctbx.python dials_assign.py ${OUT_DIR}
cctbx.python assign_beams_to_stills.py ${OUT_DIR}/optimized.expt ${OUT_DIR}/optimized.refl ${OUT_DIR}/multi
bash refine_initial.sh multi.expt multi.refl ultra_refined ${OUT_DIR}
bash refine_remove_outliers.sh ultra_refined.expt ultra_refined.refl mega_ultra_refined ${OUT_DIR}
cctbx.python poly_predictor.py ${OUT_DIR}

if [ $USE_SLURM_FOR_INTEGRATION -eq 1 ]
then
  sbatch integrate_launch.sh ${OUT_DIR} $DIALS_PATH
  sbatch int_test_launch.sh ${OUT_DIR} $DIALS_PATH
else
  cctbx.python integrate.py ${OUT_DIR}
  cctbx.python int_test.py ${OUT_DIR}
  cctbx.python stills2mtz.py ${OUT_DIR}
fi

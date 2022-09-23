DIFF_IMG_DIR='/n/holyscratch01/hekstra_lab/brookner/lauescr/hsDHFR/images/'
IMG_PREFIX='e011e_off_'
OUT_DIR='dials_files_182_362'

OSCILLATION=4
SPACE_GROUP_NUMBER=20
CELL='"87.897,94.559,96.265,90.000,90.000,90.000"'

mkdir $OUT_DIR

bash scan_varying_indexer.sh ${OUT_DIR} ${DIFF_IMG_DIR} ${IMG_PREFIX} ${OSCILLATION} ${SPACE_GROUP_NUMBER} ${CELL}
cctbx.python sequence_to_stills_no_sb.py ${OUT_DIR}/refined_varying.*
cctbx.python dials_assign.py ${OUT_DIR}
cctbx.python assign_beams_to_stills.py ${OUT_DIR}/optimized.expt ${OUT_DIR}/optimized.refl ${OUT_DIR}/multi
bash refine_initial.sh multi.expt multi.refl ultra_refined ${OUT_DIR}
bash refine_remove_outliers.sh ultra_refined.expt ultra_refined.refl mega_ultra_refined ${OUT_DIR}
cctbx.python poly_predictor.py ${OUT_DIR}
cctbx.python integrate.py ${OUT_DIR}
cctbx.python int_test.py ${OUT_DIR}
cctbx.python stills2mtz.py ${OUT_DIR}
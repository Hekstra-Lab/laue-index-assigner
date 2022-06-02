source config_params.txt
bash scan_varying_indexer.sh
cctbx.python sequence_to_stills_no_sb.py dials_temp_files/refined_varying.*
cctbx.python dials_assign.py
cctbx.python assign_beams_to_stills.py dials_temp_files/optimized.expt dials_temp_files/optimized.refl dials_temp_files/multi
bash refine_initial.sh multi.expt multi.refl ultra_refined
bash refine_remove_outliers.sh ultra_refined.expt ultra_refined.refl mega_ultra_refined
cctbx.python poly_predictor.py
libtbx.python int_test.py

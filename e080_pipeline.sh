bash scan_varying_indexer.sh
cctbx.python sequence_to_stills_no_sb.py dials_temp_files/refined_varying.*
cctbx.python dials_assign.py
cctbx.python assign_beams_to_stills.py dials_temp_files/optimized.expt dials_temp_files/optimized.refl dials_temp_files/multi
bash refine_stills_no_sb.sh multi.expt multi.refl ultra_refined
cctbx.python gen_lam_func.py

bash scan_varying_indexer.sh
cctbx.python dials_assign.py
cctbx.python sequence_to_stills_no_sb.py dials_temp_files/refined_varying.expt dials_temp_files/optimized.refl
cctbx.python assign_beams_to_stills.py

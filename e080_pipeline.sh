bash scan_varying_indexer.sh
cctbx.python sequence_to_stills_no_sb.py dials_temp_files/refined_varying.*
cctbx.python dials_assign.py
cctbx.python assign_beams_to_stills.py

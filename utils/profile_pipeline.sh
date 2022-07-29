bash scan_varying_indexer.sh
cctbx.python -m cProfile -s cumtime sequence_to_stills_no_sb.py dials_temp_files/refined_varying.* | tee monochromatic.prf
cctbx.python -m cProfile -s cumtime dials_assign.py | tee optimized.prf
cctbx.python -m cProfile -s cumtime assign_beams_to_stills.py dials_temp_files/optimized.expt dials_temp_files/optimized.refl dials_temp_files/multi | tee multi.prf
bash refine_initial.sh multi.expt multi.refl ultra_refined
bash refine_remove_outliers.sh ultra_refined.expt ultra_refined.refl mega_ultra_refined
cctbx.python -m cProfile -s cumtime poly_predictor.py | tee predicted.prf
cctbx.python -m cProfile -s cumtime integrate.py | tee integrated.prf

FILE_INPUT_TEMPLATE="/n/home04/rahewitt/laue_indexer/laue-index-assigner/dials_temp_files/multi_expts/"
FILE_OUTPUT_TEMPLATE="/n/home04/rahewitt/laue_indexer/laue-index-assigner/dials_temp_files/refined_expts/"
TUKEY_MULTIPLIER=0.
# 1 unit cell axis is fixed in precognition (b-axis) (maybe tolerance for detector distance -- look into this)
# det. dist./wavelength/unit cell axes (fix 2/3)
# tentatively fix wavelength + b-axis
# turn this into stills before refinement

dials.refine ${FILE_INPUT_TEMPLATE}${1} \
  refinement.reflections.weighting_strategy.override='stills' \
  refinement.parameterisation.beam.fix='in_spindle_plane' `# Consider fixing Distance` \
  refinement.parameterisation.detector.fix='distance' `# Consider fixing Distance` \
  refinement.parameterisation.crystal.unit_cell.fix_list='real_space_a' \
  refinement.parameterisation.beam.constraints.parameter='Mu2' \
  refinement.reflections.outlier.algorithm='null' \
  output.log="${FILE_OUTPUT_TEMPLATE}dials.${2}.log" \
  output.experiments="${FILE_OUTPUT_TEMPLATE}${2}.expt" \
  output.reflections="${FILE_OUTPUT_TEMPLATE}${2}.refl" 
#  refinement.parameterisation.beam.fix=all \

#  refinement.reflections.outlier.algorithm='mcd' \
#  refinement.reflections.outlier.algorithm='tukey' \
#  refinement.reflections.outlier.tukey.iqr_multiplier=0. \


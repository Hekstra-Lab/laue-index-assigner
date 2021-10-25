FILE_OUTPUT_TEMPLATE="/n/home04/rahewitt/laue_indexer/laue-index-assigner/dials_temp_files/"
TUKEY_MULTIPLIER=0.
# 1 unit cell axis is fixed in precognition (b-axis) (maybe tolerance for detector distance -- look into this)
# det. dist./wavelength/unit cell axes (fix 2/3)
# tentatively fix wavelength + b-axis
# turn this into stills before refinement

dials.refine ${FILE_OUTPUT_TEMPLATE}multi_stills.* \
  refinement.reflections.weighting_strategy.override='stills' \
  refinement.reflections.outlier.algorithm='tukey' \
  refinement.reflections.outlier.tukey.iqr_multiplier=0. \
  refinement.parameterisation.beam.fix=all \
  refinement.parameterisation.detector.fix=orientation `# Consider fixing Distance` \
  output.log="${FILE_OUTPUT_TEMPLATE}dials.ultra_refined.log" \
  output.experiments="${FILE_OUTPUT_TEMPLATE}ultra_refined.expt" \
  output.reflections="${FILE_OUTPUT_TEMPLATE}ultra_refined.refl" 


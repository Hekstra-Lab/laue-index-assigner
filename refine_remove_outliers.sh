SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
FILE_INPUT_TEMPLATE="${SCRIPT_DIR}/dials_temp_files/"
FILE_OUTPUT_TEMPLATE="${SCRIPT_DIR}/dials_temp_files/"
TUKEY_MULTIPLIER=0.

dials.refine -v -v  ${FILE_INPUT_TEMPLATE}${1} ${FILE_INPUT_TEMPLATE}${2} \
  refinement.reflections.weighting_strategy.override='stills' \
  refinement.parameterisation.beam.fix="*in_spindle_plane *out_spindle_plane" \
  refinement.parameterisation.detector.fix='distance' \
  refinement.parameterisation.crystal.unit_cell.fix_list='real_space_a' \
  refinery.engine='SparseLevMar' \
  refinement.reflections.outlier.minimum_number_of_reflections=1 \
  refinement.reflections.outlier.algorithm='mcd' \
  refinement.reflections.outlier.separate_images='True' \
  refinement.parameterisation.auto_reduction.action='remove' \
  parameterisation.auto_reduction.min_nref_per_parameter=1 \
  output.log="${FILE_OUTPUT_TEMPLATE}dials.${3}.log" \
  output.experiments="${FILE_OUTPUT_TEMPLATE}${3}.expt" \
  output.reflections="${FILE_OUTPUT_TEMPLATE}${3}.refl" 
#  refinement.parameterisation.beam.fix=all \

#  refinement.reflections.outlier.algorithm='tukey' \
#  refinement.reflections.outlier.tukey.iqr_multiplier=0. \
#  refinement.parameterisation.beam.constraints.parameter='Mu2' \
#  refinement.reflections.outlier.mcd.threshold_probability=0.99 \
#  refinement.reflections.outlier.algorithm='null' \

cctbx.python store_wavelengths.py

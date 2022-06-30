#!/bin/bash

# Relative path of repo directory
REL_PATH="$PWD"

# Parameters
FILE_INPUT_TEMPLATE="${DIFF_IMG_DIR}/e080_###.mccd" # Env variable set by config.sh
FILE_OUTPUT_TEMPLATE="${REL_PATH}/dials_temp_files/"
# EXPECTED_WAVELENGTH=1.070490100011937
EXPECTED_WAVELENGTH=1.04
OSCILLATION=1
SPACE_GROUP_NUMBER=19
CELL='"34,45,99,90,90,90"'
PIXEL_SIZE="(0.08854,0.08854)"
TUKEY_MULTIPLIER=0.
MASK=0
if test -f "${REL_PATH}/dials_temp_files/pixels.mask"; then
    MASK=1
fi

dials.import geometry.scan.oscillation=0,$OSCILLATION \
    geometry.goniometer.axes=0,1,0 \
    geometry.beam.wavelength=$EXPECTED_WAVELENGTH \
    geometry.detector.panel.pixel_size=$PIXEL_SIZE \
    input.template=$FILE_INPUT_TEMPLATE \
    output.experiments="${FILE_OUTPUT_TEMPLATE}imported.expt" \
    output.log="${FILE_OUTPUT_TEMPLATE}dials.import.log"

if [ $MASK == 1 ]; then
    dials.find_spots "${FILE_OUTPUT_TEMPLATE}imported.expt" \
        nproc=12 \
        spotfinder.lookup.mask="${REL_PATH}/dials_temp_files/pixels.mask" \
        spotfinder.threshold.dispersion.gain=0.10 \
        spotfinder.force_2d=True \
        max_separation=10 \
        output.shoeboxes=False \
        output.reflections="${FILE_OUTPUT_TEMPLATE}strong.refl" \
        output.log="${FILE_OUTPUT_TEMPLATE}dials.find_spots.log"
        #spotfinder.filter.d_min=0 \
else
    dials.find_spots "${FILE_OUTPUT_TEMPLATE}imported.expt" \
    nproc=12 \
    spotfinder.threshold.dispersion.gain=0.10 \
    spotfinder.force_2d=True \
    max_separation=10 \
    output.shoeboxes=False \
    output.reflections="${FILE_OUTPUT_TEMPLATE}strong.refl" \
    output.log="${FILE_OUTPUT_TEMPLATE}dials.find_spots.log"
    #spotfinder.filter.d_min=0 \
fi

dials.index "${FILE_OUTPUT_TEMPLATE}imported.expt" "${FILE_OUTPUT_TEMPLATE}strong.refl" \
    space_group=$SPACE_GROUP_NUMBER \
    unit_cell=$CELL \
    indexing.refinement_protocol.n_macro_cycles=10 \
    refinement.parameterisation.goniometer.fix=None \
    refinement.parameterisation.beam.fix=all \
    refinement.parameterisation.detector.fix=orientation \
    refinement.parameterisation.scan_varying=True \
    refinement.reflections.outlier.algorithm=tukey \
    refinement.reflections.outlier.tukey.iqr_multiplier=$TUKEY_MULTIPLIER \
    refinement.reflections.outlier.minimum_number_of_reflections=1 \
    output.log="${FILE_OUTPUT_TEMPLATE}dials.expected_index.log" \
    output.experiments="${FILE_OUTPUT_TEMPLATE}expected_index.expt" \
    output.reflections="${FILE_OUTPUT_TEMPLATE}expected_index.refl"

dials.refine "${FILE_OUTPUT_TEMPLATE}expected_index.expt" "${FILE_OUTPUT_TEMPLATE}expected_index.refl" \
    refinement.parameterisation.goniometer.fix=None `# Consider fixing entirely as proxy for detector.orientation` \
    refinement.parameterisation.beam.fix=all \
    refinement.parameterisation.crystal.fix=cell \
    refinement.parameterisation.detector.fix=orientation `# Consider fixing Distance` \
    refinement.parameterisation.scan_varying=True \
    refinement.reflections.outlier.algorithm=tukey \
    refinement.reflections.outlier.tukey.iqr_multiplier=$TUKEY_MULTIPLIER \
    refinement.reflections.outlier.minimum_number_of_reflections=1 \
    output.log="${FILE_OUTPUT_TEMPLATE}dials.refined_varying.log" \
    output.experiments="${FILE_OUTPUT_TEMPLATE}refined_varying.expt" \
    output.reflections="${FILE_OUTPUT_TEMPLATE}refined_varying.refl" 

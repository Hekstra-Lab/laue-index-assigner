from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory
from dxtbx.model import ExperimentList
from dials.algorithms.refinement.prediction.managed_predictors import ExperimentsPredictorFactory
from copy import deepcopy
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import reciprocalspaceship as rs
import argparse
import sys

# Get I/O options from user
parser = argparse.ArgumentParser()
parser.add_argument('in_expt', help='Input experiment file.')
parser.add_argument('in_refl', help='Input reflection file.')
parser.add_argument('output', help='Template for output file. {output}.expt, {output}.refl will be produced.')
args = parser.parse_args()

# Set parameters
expt_filename = args.in_expt
refl_filename = args.in_refl
new_expt_filename = args.output + '.expt'
new_refl_filename = args.output + '.refl'

# Get experiments
expts = ExperimentListFactory.from_json_file(expt_filename)
new_expts = ExperimentList()

# Initialize flex tables for refl files
refl_input = flex.reflection_table().from_file(refl_filename)
refl_output = refl_input.copy()
refl_output["id"] = flex.int([-1]*len(refl_output))

# Initialize data frame
dials_df = rs.DataSet({
    'Wavelength' : refl_input['Wavelength'],
    'ID' : refl_input['id'],
    'new_ID' : [-1]*len(refl_input)
})#.infer_mtz_dtypes()
 
# Generate beams per reflection
print(f'Number of rows: {len(dials_df)}')
exp_id = -1
for i, refl in tqdm(dials_df.iterrows()):
    try:
        # New beam per reflection
        expt = expts[refl['ID'][i]]
        temp = expt.beam.get_s0()
        new_expt = expt
        new_expt.beam = deepcopy(expt.beam)
        new_expt.beam.set_wavelength(refl['Wavelength'][i])
        s0 = (expt.beam.get_s0() / np.linalg.norm(expt.beam.get_s0())) / new_expt.beam.get_wavelength()
        new_expt.beam.set_s0(s0)
        exp_id = exp_id + 1 # Increment experiment ID
        new_expt.identifier = str(exp_id)
        new_expts.append(new_expt)

        # Write new beam identifiers to reflections
        dials_df.at[i,'new_ID'] = exp_id
    except:
        print('Warning: Unindexed strong spot has wavelength 0.')
        dials_df.at[i,'new_ID'] = -1
        continue

print('finished loop')
# Replace reflection IDs with new IDs
idx = flex.int(dials_df['new_ID'])
refl_output["id"] = idx
print('assigned IDs')

# Repredict centroids
print('Repredicting centroids')
ref_predictor = ExperimentsPredictorFactory.from_experiments(new_expts)
ref_predictor(refl_output)

print('writing experiments')
# Write experiment file with multiple beams
new_expts.as_file(new_expt_filename)
print('experiments written')

print('writing refls')
# Write refl file
refl_output.as_file(new_refl_filename)
print('refls written')


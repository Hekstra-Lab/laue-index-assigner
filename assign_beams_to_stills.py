from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory
from dxtbx.model import ExperimentList
from copy import deepcopy
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import reciprocalspaceship as rs

# Set parameters
expt_filename = "./dials_temp_files/stills_no_sb.expt"
refl_filename = "./dials_temp_files/stills_no_sb.refl"
new_expt_filename = "./dials_temp_files/stills_no_sb_multi.expt"
new_refl_filename = "./dials_temp_files/stills_no_sb_multi.refl"

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
for i, refl in tqdm(dials_df.iterrows()):
    # New beam per reflection
    expt = expts[refl['ID'][i]]
    new_expt = expt
    new_expt.beam = deepcopy(expt.beam)
    new_expt.beam.set_wavelength(refl['Wavelength'][i])
    new_expt.identifier = str(i)
    new_expts.append(new_expt)

    # Write new beam identifiers to reflections
    dials_df.at[i,'new_ID'] = i

print('finished loop')
# Replace reflection IDs with new IDs
idx = flex.int(dials_df['new_ID'])
refl_output["id"] = idx
print('assigned IDs')

# Write new wavelength-dependent data (should we do this or keep optimized values?)
#for i in np.arange(len(refl_output)): # Normalize each s1, divide by appropriate wavelength
#    s1 = refl_output["s1"][i]
#    s1 /= np.linalg.norm(s1)
#    s1 = s1/new_expts[refl_output["id"][i]].beam.get_wavelength()
#    refl_output["s1"][i] = s1
#refl_output.map_centroids_to_reciprocal_space(new_expts)
#hkl_predictor = AssignIndicesLocal(nearest_neighbours=8)
##hkl_predictor = AssignIndicesGlobal(tolerance=0.1)
#hkl_predictor(refl_output, new_expts)

print('writing experiments')
# Write experiment file with multiple beams
new_expts.as_file(new_expt_filename)
print('experiments written')

print('writing refls')
# Write refl file
refl_output.as_file(new_refl_filename)
print('refls written')


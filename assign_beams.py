from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory
from dxtbx.model import ExperimentList
from copy import deepcopy
import numpy as np
import pandas as pd
import reciprocalspaceship as rs

# TODO: Check these filter functions for correctness -- these were grabbed from another file

def filter_experiments(exptList, refl_table):
    'Filters the list of experiments to remove those with no assigned reflections'
    new_expt_list = deepcopy(exptList) # Check that this method doesn't copy reference but value
    new_refl_table = refl_table.copy() # Same here
    expts_to_remove = []
    for i in range(len(new_expt_list)):
        idx = np.asarray(refl_output["id"] == i)
        # Remove the experiment and decrement experiment IDs of all subsequent experiments
        # Also adjust reflections' assignments to experiments accordingly
        if(idx.sum() == 0):
            expts_to_remove.append(i)
            new_refl_table = decrement_refl_ids(new_refl_table, i)
    # Remove the empty experiments here
    new_expt_list = remove_expts(new_expt_list, expts_to_remove)
    return new_expt_list, new_refl_table

def decrement_refl_ids(refl_table, index):
    new_refl_table = refl_table.copy()
    to_be_decremented = refl_table["id"] >= index
    for i in range(len(new_refl_table)):
        if(to_be_decremented[i]):
            new_refl_table["id"][i] = new_refl_table["id"][i] - 1
    return new_refl_table

def remove_expts(exptList, expts_to_remove):
    new_expt_list = ExperimentList()
    num_removed = 0
    for i in range(len(exptList)):
        if(i not in expts_to_remove):
            exptList[i].identifier = str(int(exptList[i].identifier) - num_removed)
            new_expt_list.append(exptList[i])
        else:
            num_removed = num_removed + 1
    return new_expt_list

def decrement_expt_ids(exptList, index):
    'Decrements the identifier for every experiment at index to the end of the experiment list'
    new_expt_list = Copy.copy(exptList)
    for i in range(index, len(exptList)):
        new_expt_list[i].identifier = str(int(new_expt_list[i].identifier)-1)
    return new_expt_list

# Set parameters
expt_filename = "/n/home04/rahewitt/laue_indexer/laue-index-assigner/dials_temp_files/refined_varying.expt"
refl_filename = "/n/home04/rahewitt/laue_indexer/laue-index-assigner/optimized.refl"
new_expt_filename = "./dials_temp_files/multi.expt"
new_refl_filename = "./dials_temp_files/multi.refl"
min_wavelength = 1.0 # Maybe there should be a smart tool that determines these beam params, with optional override
max_wavelength = 1.2
n_beams = 41
lams = np.linspace(min_wavelength, max_wavelength, num=n_beams)
lam_step = lams[1]-lams[0]

# Get experiments
expts = ExperimentListFactory.from_json_file(expt_filename)
expt = expts[0]
new_expts = ExperimentList()

# Generate all beam objects
for i,lam in enumerate(lams):
    new_expt = expt
    new_expt.beam = deepcopy(expt.beam)
    new_expt.beam.set_wavelength(lam)
    new_expt.identifier = str(i)
    new_expts.append(new_expt)

# Initialize flex tables for refl files
refl_input = flex.reflection_table().from_file(refl_filename)
refl_output = refl_input.copy()
refl_output["id"] = flex.int([-1]*len(refl_output))

# Initialize data frame
dials_df = rs.DataSet({
    'Wavelength' : refl_input['Wavelength'].as_numpy_array(),
    'ID' : refl_input['xyzobs.px.value'].parts()[2].as_numpy_array() - 0.5,
}).infer_mtz_dtypes()

# Bin data frame IDs by closest beam wavelength
bins = np.asarray(np.append(lams - lam_step/2, lams[-1]+lam_step/2)) # This leaves out everything above/below a lam_step of the max/min beams, should we exclude large/small wavelengths, throw a warning, or simply add those reflections to their closest beams?
labels = np.arange(len(lams))
dials_df['new_ID'] = pd.cut(dials_df['Wavelength'], bins=bins, labels=labels)
dials_df['new_ID'][dials_df['Wavelength'] < bins[0]] = 0
if(np.sum(pd.to_numeric(dials_df['new_ID'])) != 0):
    print("Warning: Wavelengths outside of range detected. Pairing them with closest matching beams.")
dials_df['new_ID'][dials_df['Wavelength'] > bins[-1]] = labels[-1]
dials_df['new_ID'] = pd.to_numeric(dials_df['new_ID'])

# Replace reflection IDs with new IDs
idx = flex.int(dials_df['new_ID'])
refl_output["id"] = idx

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

# Filter out any experiments with no reflections TODO: MAKE SURE THIS STILL WORKS FOR SURE
new_expts, refl_output = filter_experiments(new_expts, refl_output)
if len(new_expts) == 0:
    print("Error: All experiments have no reflections.")

# Write experiment file with multiple beams
new_expts.as_file(new_expt_filename)

# Write refl file
refl_output.as_file(new_refl_filename)
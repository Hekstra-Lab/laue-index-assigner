from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
from dials.array_family import flex
import numpy as np
import argparse

# Get I/O options from user
print('Parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument('in_expt', help='Input experiment file.')
parser.add_argument('in_refl', help='Input reflection file.')
parser.add_argument('res_cutoff', help='Resolution cutoff.', type=float)
args = parser.parse_args()

# Get experiment+refl table names
print('Loading data')
expt_file = args.in_expt
refl_file = args.in_refl

# Load experiment list and reflection table
elist = ExperimentListFactory.from_json_file(expt_file, check_format=True)
refls = reflection_table.from_file(refl_file)

# Remove reflections below cutoff
print('Removing reflections')
cutoff = args.res_cutoff
resolutions = refls.compute_d(elist).as_numpy_array()
selected = resolutions >= cutoff

# Trim reflection table based on cutoff
print('Generating new experiment list and reflection table')
new_refls = refls.select(flex.bool(selected))

# Construct new experiment list for these reflections
idx = new_refls['id'].as_numpy_array().astype(str).tolist()
elist.select_on_experiment_identifiers(idx)

# Overwrite data
print('Overwriting data')
elist.as_file(expt_file)
new_refls.as_file(refl_file)

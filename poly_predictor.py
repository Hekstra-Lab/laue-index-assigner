from diffgeolib import LauePredictor
import numpy as np
import gemmi
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table

# Hyperparameters for predictor
lam_min = 0.95
lam_max = 1.15
d_min = 1.4

# Load DIALS files
expt_file = "dials_temp_files/ultra_refined.expt"
refl_file = "dials_temp_files/ultra_refined.refl"

# Get data
elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)
refls = reflection_table.from_file(refl_file)

# Get beam vector
s0 = np.array(elist[0].beam.get_s0())

# Get experiment data from experiment objects
experiment = elist[0]
cryst = experiment.crystal
spacegroup = gemmi.SpaceGroup(cryst.get_space_group().type().universal_hermann_mauguin_symbol())

# Get unit cell params
cell_params = cryst.get_unit_cell().parameters()
cell = gemmi.UnitCell(*cell_params)

# Get U matrix
U = np.asarray(cryst.get_U()).reshape(3,3)

la = LauePredictor(s0, cell, U, lam_min, lam_max, d_min, spacegroup)
temp = la.predict_s1()
print(temp.shape)

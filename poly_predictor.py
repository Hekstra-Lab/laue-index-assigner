from diffgeolib import LauePredictor
import numpy as np
import gemmi
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table


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

# Wavelengths per spot
lams = refls['Wavelength'].as_numpy_array()

# Hyperparameters for predictor
lam_min = np.min(lams)
lam_max = np.max(lams)
d_min = 1.4 # TODO: What's a good value or calculation for this?

# Get s1 vectprs
la = LauePredictor(s0, cell, U, lam_min, lam_max, d_min, spacegroup)
s1 = la.predict_s1()
print(s1.shape)

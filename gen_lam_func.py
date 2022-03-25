"""
This script generates the lambda-curve for the experiment.
"""
import numpy as np
import reciprocalspaceship as rs
from dxtbx.model import ExperimentList
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
from dials.array_family import flex
from matplotlib import pyplot as plt
import scipy
import seaborn as sns
from IPython import embed

# TODO: Write function 

# Load files
expt_file = "dials_temp_files/mega_ultra_refined.expt"
refl_file = "dials_temp_files/mega_ultra_refined.refl"
elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)
refls = reflection_table.from_file(refl_file)

# Remove outliers                                                     
idx = refls.get_flags(refls.flags.used_in_refinement).as_numpy_array()
idy = np.arange(len(elist))[idx].tolist()                             
elist = ExperimentList([elist[i] for i in idy])                       
refls = refls.select(flex.bool(idx))                                  

# Get rlps and normalize
refls.map_centroids_to_reciprocal_space(elist)
rlps = refls['rlp'].as_numpy_array()
norms = np.linalg.norm(rlps, axis=1)

# Get angle between rlp and beam vector
z = elist[0].beam.get_unit_s0()
thetas = rs.utils.math.angle_between(rlps, z) # in radians

# Get wavelength of rlp
lams = refls['Wavelength'].as_numpy_array()

# Relevant plot
x = norms*np.cos(np.deg2rad(thetas))
y = norms*np.sin(np.deg2rad(thetas))
plt.plot(x, y, '.k', alpha=0.1)
plt.xlim(-1,0.1)
plt.ylim(-0.1,1)
plt.show()

# Fit with kernel density estimator
train_data = np.asarray([x, y])
kde = scipy.stats.gaussian_kde(train_data)

# Above is a 2D Cartesian estimator...can we just use a 1D radial estimator?

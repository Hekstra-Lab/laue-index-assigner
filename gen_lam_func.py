"""
This script generates the lambda-curve for the experiment.
"""
import numpy as np
import reciprocalspaceship as rs
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
from dials.array_family import flex
from matplotlib import pyplot as plt
from IPython import embed

# Load files
expt_file = "dials_temp_files/ultra_refined.expt"
refl_file = "dials_temp_files/ultra_refined.refl"
elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)
refls = reflection_table.from_file(refl_file)

# Get rlps and normalize
refls.map_centroids_to_reciprocal_space(elist)
rlps = refls['rlp'].as_numpy_array()
normed_rlps = rlps / np.linalg.norm(rlps, axis=1)[:, None]

# Get absolute angle between rlp and beam vector
z = elist[0].beam.get_unit_s0()
thetas = rs.utils.math.angle_between(rlps, z) # in radians

# Get wavelength of rlp
# TODO: Check if these wavelengths still correspond to their beam wavelengths after refinement
# ANSWER: They don't --- fix this
lams = refls['Wavelength'].as_numpy_array()

# Plot data
plt.hist(np.rad2deg(thetas), bins=100)
plt.xlabel('Angle with beam direction (degs)')
plt.ylabel('Number of reflections')
plt.show()

plt.hist(lams, bins=100)
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Number of reflections')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(normed_rlps[:,0], normed_rlps[:,1], normed_rlps[:,2], color='green')
plt.title('Normalized RLPs')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(rlps[:,0], rlps[:,1], rlps[:,2], color='green')
plt.title('RLPs')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

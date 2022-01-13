"""
This script generates the lambda-curve for the experiment.
"""
import numpy as np
import reciprocalspaceship as rs
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
from dials.array_family import flex
from matplotlib import pyplot as plt
import scipy
import seaborn as sns

# TODO: Write function 

# Load files
expt_file = "dials_temp_files/ultra_refined.expt"
refl_file = "dials_temp_files/ultra_refined.refl"
elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)
refls = reflection_table.from_file(refl_file)

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
# TODO: Add circles
x = norms*np.cos(np.deg2rad(thetas))
y = norms*np.sin(np.deg2rad(thetas))
plt.plot(x, y, '.k', alpha=0.1)
plt.xlim(-1,0.1)
plt.ylim(-0.1,1)
plt.show()

# Fit with kernel density estimator
train_data = np.asarray([x, y])
kde = scipy.stats.gaussian_kde(train_data)

# Heatmap results of kde
n = 20
x_lims = np.linspace(-1, 0.1, n)
y_lims = np.linspace(-0.1, 1, n)
dx = (x_lims[-1] - x_lims[0]) / n
dy = (y_lims[-1] - y_lims[0]) / n
results = np.zeros(shape=(len(x_lims), len(y_lims)))
for i in range(len(x_lims)):
    for j in range(len(y_lims)):
        results[i,j] = kde.integrate_box((x_lims[i]-dx/2, y_lims[j]-dy/2), (x_lims[i]+dx/2, y_lims[j]+dy/2))
ax = sns.heatmap(results, annot=True, xticklabels=x_lims, yticklabels=y_lims, vmin=np.min(results), vmax=np.max(results), cbar=True)
ax.invert_yaxis()
plt.show()

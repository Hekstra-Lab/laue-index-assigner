from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
import argparse

# Get I/O options from user
parser = argparse.ArgumentParser()
parser.add_argument('in_expt', help='Input experiment file.')
parser.add_argument('in_refl', help='Input reflection file.')
args = parser.parse_args()

# Get files
expt_filename = args.in_expt
refl_filename = args.in_refl

#centroid distance cutoff in pixels
centroid_max_distance = 10.

print('reading DIALS files')
elist = ExperimentListFactory.from_json_file(expt_filename)
refls = reflection_table.from_file(refl_filename)
refls_in = refls.select(refls.get_flags(refls.flags.used_in_refinement)) # Inliers
refls_out = refls.select(~refls.get_flags(refls.flags.used_in_refinement)) # Outliers

# Outlier data
dials_df_out = pd.DataFrame(data = 
np.asarray([refls_out['xyzobs.px.value'].parts()[0].as_numpy_array(),
    refls_out['xyzobs.px.value'].parts()[1].as_numpy_array(),
    refls_out['xyzcal.px'].parts()[0].as_numpy_array(),
    refls_out['xyzcal.px'].parts()[1].as_numpy_array(),
    refls_out['miller_index'].as_vec3_double().parts()[0].as_numpy_array(),
    refls_out['miller_index'].as_vec3_double().parts()[1].as_numpy_array(),
    refls_out['miller_index'].as_vec3_double().parts()[2].as_numpy_array(),
    refls_out['Wavelength'].as_numpy_array()]).T,
    columns = ['X','Y','Xpred','Ypred','H','K','L','Wavelength'])

# Inlier data
dials_df = pd.DataFrame(data = 
np.asarray([refls_in['xyzobs.px.value'].parts()[0].as_numpy_array(),
    refls_in['xyzobs.px.value'].parts()[1].as_numpy_array(),
    refls_in['xyzcal.px'].parts()[0].as_numpy_array(),
    refls_in['xyzcal.px'].parts()[1].as_numpy_array(),
    refls_in['miller_index'].as_vec3_double().parts()[0].as_numpy_array(),
    refls_in['miller_index'].as_vec3_double().parts()[1].as_numpy_array(),
    refls_in['miller_index'].as_vec3_double().parts()[2].as_numpy_array(),
    refls_in['Wavelength'].as_numpy_array()]).T,
    columns = ['X','Y','Xpred','Ypred','H','K','L','Wavelength'])


xy  = dials_df[['X', 'Y']].to_numpy(float)
xy_pred  = dials_df[['Xpred', 'Ypred']].to_numpy(float)
xy_out  = dials_df_out[['X', 'Y']].to_numpy(float)
xy_pred_out  = dials_df_out[['Xpred', 'Ypred']].to_numpy(float)

x_resids = xy_pred[:,0] - xy[:,0]
y_resids = xy_pred[:,1] - xy[:,1]
x_resids_out = xy_pred_out[:,0] - xy_out[:,0]
y_resids_out = xy_pred_out[:,1] - xy_out[:,1]

# Plot centroids
# For inliers
xobs = xy[:,0]
yobs = xy[:,1]
xcalc = xy_pred[:,0]
ycalc = xy_pred[:,1]

plt.plot(np.vstack((xobs, xcalc)), np.vstack((yobs,ycalc)), '-k')
plt.plot(xobs, yobs, 'o', color='#a6611a', label='Observed inliers')
plt.plot(xcalc, ycalc, 'o', color='#dfc27d', label='Predicted inliers')

# For outliers
xobs = xy_out[:,0]
yobs = xy_out[:,1]
xcalc = xy_pred_out[:,0]
ycalc = xy_pred_out[:,1]

plt.plot(np.vstack((xobs, xcalc)), np.vstack((yobs,ycalc)), '-k')
plt.plot(xobs, yobs, 'o', color='#018571', label='Observed outliers')
plt.plot(xcalc, ycalc, 'o', color='#80cdc1', label='Predicted outliers')
plt.title('Centroid Residuals')
plt.xlabel('X (px)')
plt.ylabel('Y (px)')
plt.legend()
plt.show()

# Residuals
plt.scatter(xy[:,0], xy[:,1], c='g', alpha=0.3)
plt.scatter(xy_pred[:,0], xy_pred[:,1], c='r', alpha=0.3)
plt.xlabel('X (px)')
plt.ylabel('Y (px)')
plt.title('Observed vs Calculated Centroids of Inliers After Refinement')
plt.legend(labels=['Observed', 'Predicted'])
plt.show()

plt.hist(x_resids, bins=50, range=(-2,2))
plt.xlabel('X residual (px)')
plt.ylabel('Bin Count')
plt.title('X Residuals Post-Refinement')
plt.show()

plt.hist(y_resids, bins=50, range=(-2,2))
plt.xlabel('Y residual (px)')
plt.ylabel('Bin Count')
plt.title('Y Residuals Post-Refinement')
plt.show()

plt.hist(dials_df['Wavelength'], bins=50)
plt.xlabel('Wavelength')
plt.ylabel('Bin Count')
plt.title('Wavelengths')
plt.show()

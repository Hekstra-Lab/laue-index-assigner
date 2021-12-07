from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table

expt_file = "" # <-- Fill this in 
refl_file = "" # <-- Fill this in 

#centroid distance cutoff in pixels
centroid_max_distance = 10.

print('reading DIALS files')
elist = ExperimentListFactory.from_json_file(expt_file)
refls = reflection_table.from_file(refl_file)

dials_df = pd.DataFrame(data = 
np.asarray([refls['xyzobs.px.value'].parts()[0].as_numpy_array(),
    refls['xyzobs.px.value'].parts()[1].as_numpy_array(),
    refls['xyzcal.px'].parts()[0].as_numpy_array(),
    refls['xyzcal.px'].parts()[1].as_numpy_array(),
    refls['miller_index'].as_vec3_double().parts()[0].as_numpy_array(),
    refls['miller_index'].as_vec3_double().parts()[1].as_numpy_array(),
    refls['miller_index'].as_vec3_double().parts()[2].as_numpy_array(),
    refls['Wavelength'].as_numpy_array()]).T,
    columns = ['X','Y','Xpred','Ypred','H','K','L','Wavelength'])


xy  = dials_df[['X', 'Y']].to_numpy(float)
xy_pred  = dials_df[['Xpred', 'Ypred']].to_numpy(float)

x_resids = xy_pred[:,0] - xy[:,0]
y_resids = xy_pred[:,1] - xy[:,1]

#plt.scatter(xy[:,0], xy[:,1], c='g', alpha=0.3)
#plt.scatter(xy_pred[:,0], xy_pred[:,1], c='r', alpha=0.3)
#plt.xlabel('X (px)')
#plt.ylabel('Y (px)')
#plt.title('Observed vs Calculated Centroids After Refinement')
#plt.legend(labels=['Observed', 'Predicted'])
#plt.show()

plt.hist(x_resids, bins=50)
plt.xlabel('X residual (px)')
plt.ylabel('Bin Count')
plt.title('X Residuals Post-Refinement')
plt.show()

plt.hist(y_resids, bins=50)
plt.xlabel('Y residual (px)')
plt.ylabel('Bin Count')
plt.title('Y Residuals Post-Refinement')
plt.show()

plt.hist(dials_df['Wavelength'], bins=50)
plt.xlabel('Wavelength')
plt.ylabel('Bin Count')
plt.title('Wavelengths')
plt.show()

import pandas as pd
import numpy as np
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
from dials.array_family import flex

# Filenames
expt_file = 'dials_temp_files/ultra_refined.expt'
refl_file = 'dials_temp_files/ultra_refined.refl'
re_file = '~/Downloads/e002c_off_002.mccd.re.spt'

# Make dataframe + refl table
precog_df = pd.read_csv(re_file, sep=' +', header=None)
elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)
refls = reflection_table.from_file(refl_file).copy()

# Get centroids
centroids = np.zeros(shape=(len(precog_df), 3))
centroids[:,0] = precog_df.iloc[:, 3].to_numpy()
centroids[:,1] = precog_df.iloc[:, 4].to_numpy()

# Replace refl centroids with dataframe centroids
refls['xyzobs.px.value'] = flex.vec3_double(centroids)

# Write files
elist.as_file('precog_integrate.expt')
refls.as_file('precog_integrate.refl')

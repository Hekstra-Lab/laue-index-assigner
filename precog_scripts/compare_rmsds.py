from diffgeolib import *
from tqdm import trange,tqdm
from matplotlib import pyplot as plt
from scitbx import matrix
from IPython import embed
from glob import glob
from utils.precognition import parse_ii_inp_file_pairs
import numpy as np
import reciprocalspaceship as rs
import gemmi
import pandas as pd

expt_file = "dials_temp_files/ultra_refined.expt"
refl_file = "dials_temp_files/ultra_refined.refl"
precog_filename = "data/precog_rmsds"

#centroid distance cutoff in pixels
centroid_max_distance = 10.

print('parsing precog files')
precog_rmsds = pd.read_csv(precog_filename, header=None, delimiter=' ')
precog_rmsds = precog_rmsds[precog_rmsds.columns[-2]].to_numpy()

print('reading DIALS files')
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
elist = ExperimentListFactory.from_json_file(expt_file)
cryst = elist.crystals()[0]
unit_cell = cryst.get_unit_cell()
spacegroup = gemmi.SpaceGroup(cryst.get_space_group().type().universal_hermann_mauguin_symbol())
cell = gemmi.UnitCell(*unit_cell.parameters())
refls = reflection_table.from_file(refl_file)
refls = refls.select(refls.get_flags(refls.flags.used_in_refinement))

print('generating DIALS dataframe')
dials_df = rs.DataSet({
    'X' : refls['xyzobs.px.value'].parts()[0].as_numpy_array(),
    'Y' : refls['xyzobs.px.value'].parts()[1].as_numpy_array(),
    'Xcal' : refls['xyzcal.px'].parts()[0].as_numpy_array(),
    'Ycal' : refls['xyzcal.px'].parts()[1].as_numpy_array(),
    'BATCH' : refls['imageset_id'].as_numpy_array(),
}, cell = cell, spacegroup=spacegroup).infer_mtz_dtypes()

print('initializing metrics')
dials_rmsds = np.zeros(len(precog_rmsds))
rmsd_diff = np.zeros(len(precog_rmsds))

# Iterate by frame and match HKLs, seeing what percentage are correct
for i in trange(len(precog_rmsds)):
    im_dia = dials_df[dials_df['BATCH'] == i]

    # Get XY positions for refls
    xy = im_dia[['X', 'Y']].to_numpy(float)
    xycal = im_dia[['Xcal', 'Ycal']].to_numpy(float)

    # Get DIALS RMSD for this image
    x_diffs = xy[:,0] - xycal[:,0]
    y_diffs = xy[:,1] - xycal[:,1]
    resid_mags = x_diffs**2 + y_diffs**2 # Square happens as first step of RMSD already
    dials_rmsds[i] = np.sqrt(np.mean(resid_mags))

    rmsd_diff[i] = dials_rmsds[i] - precog_rmsds[i]

plt.figure()
plt.plot(range(len(precog_rmsds)), precog_rmsds)
plt.xlabel('Image Number')
plt.ylabel('Precognition RMSDs (px)')
plt.show()

plt.figure()
plt.plot(range(len(precog_rmsds)), dials_rmsds)
plt.xlabel('Image Number')
plt.ylabel('DIALS RMSDs (px)')
plt.show()

plt.figure()
plt.plot(range(len(precog_rmsds)), rmsd_diff)
plt.xlabel('Image Number')
plt.ylabel('RMSD Difference Between DIALS/Precognition (px)')
plt.show()

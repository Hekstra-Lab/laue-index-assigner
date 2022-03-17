from diffgeolib import *
from tqdm import trange,tqdm
from matplotlib import pyplot as plt
from scitbx import matrix
from IPython import embed
from glob import glob
import numpy as np
import reciprocalspaceship as rs
import gemmi
import pandas as pd

expt_file1 = "dials_temp_files/optimized.expt"
refl_file1 = "dials_temp_files/optimized.refl"
expt_file2 = "dials_temp_files/ultra_refined.expt"
refl_file2 = "dials_temp_files/ultra_refined.refl"

#centroid distance cutoff in pixels
centroid_max_distance = 10.

print('reading DIALS files')
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
elist1 = ExperimentListFactory.from_json_file(expt_file1)
cryst1 = elist1.crystals()[0]
unit_cell1 = cryst1.get_unit_cell()
refls1 = reflection_table.from_file(refl_file1)
elist2 = ExperimentListFactory.from_json_file(expt_file2)
cryst2 = elist2.crystals()[0]
unit_cell2 = cryst2.get_unit_cell()
refls2 = reflection_table.from_file(refl_file2)

# Remove reflections not used in refinement
refls2 = refls2.select(refls2.get_flags(refls2.flags.used_in_refinement))

print('generating DIALS dataframes')
dials_df1 = rs.DataSet({
    'X' : refls1['xyzobs.px.value'].parts()[0].as_numpy_array(),
    'Y' : refls1['xyzobs.px.value'].parts()[1].as_numpy_array(),
    'Wavelength' : refls1['Wavelength'].as_numpy_array(),
    'BATCH' : refls1['imageset_id'].as_numpy_array(),
}, cell = gemmi.UnitCell(*unit_cell1.parameters()), spacegroup=gemmi.SpaceGroup(cryst1.get_space_group().type().universal_hermann_mauguin_symbol())).infer_mtz_dtypes()
dials_df2 = rs.DataSet({
    'X' : refls2['xyzobs.px.value'].parts()[0].as_numpy_array(),
    'Y' : refls2['xyzobs.px.value'].parts()[1].as_numpy_array(),
    'Wavelength' : refls2['Wavelength'].as_numpy_array(),
    'BATCH' : refls2['imageset_id'].as_numpy_array(),
}, cell = gemmi.UnitCell(*unit_cell2.parameters()), spacegroup=gemmi.SpaceGroup(cryst2.get_space_group().type().universal_hermann_mauguin_symbol())).infer_mtz_dtypes()

print('initializing metrics')
nspots = np.zeros(len(elist2))
nmatch = np.zeros(len(elist2))

both_df = None

# Iterate by frame and match HKLs, seeing what percentage are correct
for i in trange(np.max(dials_df2['BATCH'])):
    # Get reflection indices from each batch
    im1 = dials_df1[dials_df1['BATCH'] == i]
    im2 = dials_df2[dials_df2['BATCH'] == i]
    nspots[i] = len(im2)
    if len(im2) == 0: # Empty image
        continue

    dmat = np.linalg.norm(
        im2[['X', 'Y']].to_numpy(float)[:,None,:] - \
        im1[['X', 'Y']].to_numpy(float)[None,:,:],
        axis = -1
    )

    # This prevents duplicated matches
    idx1,idx2 = np.where((dmat == dmat.min(0)) & (dmat == dmat.min(1)[:,None]) & (dmat <= centroid_max_distance))
    im2 = im2.iloc[idx1]
    im1 = im1.iloc[idx2]

    # Get XY positions for refls
    xy1 = im1[['X', 'Y']].to_numpy(float)
    xy2  = im2[['X', 'Y']].to_numpy(float)

    nmatch[i] = len(im2)

    # Add this image to `both_df`
    _both_df = im2.reset_index().join(im1.reset_index(), rsuffix='_pre', lsuffix='_post')
    both_df = pd.concat((both_df, _both_df))

plt.figure()
plt.plot(
    both_df['Wavelength_pre'].to_numpy(),
    both_df['Wavelength_post'].to_numpy(),
    'k.',
    alpha=0.1,
)
plt.xlabel('$\lambda$ (Pre-Refinement)')
plt.ylabel('$\lambda$ (Post-Refinement)')
plt.show()

lam_diffs = both_df['Wavelength_post'] - both_df['Wavelength_pre']
plt.figure()
plt.hist(lam_diffs, bins=100)
plt.xlabel('Wavelength Shift (Angstroms)')
plt.ylabel('Number of Spots')
plt.show()

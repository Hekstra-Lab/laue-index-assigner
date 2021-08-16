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

#expt_file = "/home/rahewitt/laue_indexer/laue-index-assigner/dials_temp_files/updated.expt"
expt_file = "dials_temp_files/refined_varying.expt"
#expt_file = "/home/rahewitt/Downloads/peak_refined.expt"
refl_file = "optimized.refl"

#centroid distance cutoff in pixels
centroid_max_distance = 10.

#ds = rs.read_precognition(ii_file).reset_index()
print('parsing precog files')
precog_df = parse_ii_inp_file_pairs(
    sorted(glob('data/e080_???.mccd.ii')),
    sorted(glob('data/e080_???.mccd.inp')),
).reset_index()

print('reading DIALS files')
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
elist = ExperimentListFactory.from_json_file(expt_file)
cryst = elist.crystals()[0]
unit_cell = cryst.get_unit_cell()
precog_df.spacegroup = gemmi.SpaceGroup(cryst.get_space_group().type().universal_hermann_mauguin_symbol())
precog_df.cell = gemmi.UnitCell(*unit_cell.parameters())
refls = reflection_table.from_file(refl_file)

print('generating DIALS dataframe')
dials_df = pd.DataFrame(index=np.zeros(len(refls)))
dials_df['X'] = refls['xyzobs.px.value'].parts()[0].as_numpy_array()
dials_df['Y'] = refls['xyzobs.px.value'].parts()[1].as_numpy_array()
dials_df['H'] = refls['miller_index'].as_vec3_double().parts()[0].as_numpy_array()
dials_df['K'] = refls['miller_index'].as_vec3_double().parts()[1].as_numpy_array()
dials_df['L'] = refls['miller_index'].as_vec3_double().parts()[2].as_numpy_array()
dials_df['Wavelength'] = refls['Wavelength'].as_numpy_array()
#dials_df['BATCH'] = refls['shoebox'].bounding_boxes().parts()[4]
dials_df['BATCH'] = refls['xyzobs.px.value'].parts()[2].as_numpy_array() - 0.5


print('initializing metrics')
percent_correct = np.zeros(elist[0].imageset.size())
percent_outliers = np.zeros(elist[0].imageset.size())
nspots = np.zeros(elist[0].imageset.size())
nmatch = np.zeros(elist[0].imageset.size())


# Iterate by frame and match HKLs, seeing what percentage are correct
for i in trange(elist[0].imageset.size()):
    # Get reflection indices from each batch
    im_pre = precog_df[precog_df['BATCH'] == i]
    im_dia = dials_df[dials_df['BATCH'] == i]
    nspots[i] = len(im_dia)

    #Removing the unindexed refls
    outliers = np.all(im_dia[['H', 'K', 'L']] == 0., axis=1)
    percent_outliers[i] = 100.*outliers.sum() / len(outliers)
    xy_outliers = im_dia.loc[outliers, ['X', 'Y']].to_numpy(float)
    im_dia = im_dia[~outliers]
    if len(im_dia) == 0:
        continue

    dmat = np.linalg.norm(
        im_dia[['X', 'Y']].to_numpy(float)[:,None,:] - \
        im_pre[['X', 'Y']].to_numpy(float)[None,:,:],
        axis = -1
    )

    # This prevents duplicated matches
    idx1,idx2 = np.where((dmat == dmat.min(0)) & (dmat == dmat.min(1)[:,None]) & (dmat <= centroid_max_distance))
    im_dia = im_dia.iloc[idx1]
    im_pre = im_pre.iloc[idx2]

    if len(im_dia) == 0:
        continue

    precog_hkl = im_pre[['H', 'K', 'L']].to_numpy(float)
    dials_hkl  = im_dia[['H', 'K', 'L']].to_numpy(float)

    # Get XY positions for refls
    precog_xy = im_pre[['X', 'Y']].to_numpy(float)
    dials_xy  = im_dia[['X', 'Y']].to_numpy(float)

    # Align precog to DIALS hkls
    aligned_hkls = align_hkls(dials_hkl, precog_hkl, precog_df.spacegroup)

    # Check correctness of matching
    correct = is_ray_equivalent(aligned_hkls, dials_hkl)
    if len(correct) > 0:
        percent_correct[i] = 100.*sum(correct)/len(correct)

    nmatch[i] = len(im_dia)

    if i == 0:
        filename = elist[0].imageset.get_image_identifier(i)
        pixels = plt.imread(filename)

        pixels[pixels==0] = 1.
        plt.matshow(np.log(pixels), cmap='Greys_r')

        plt.plot(*precog_xy.T, 'yo', mfc='none', ms=11, label='Precog')
        plt.plot( *dials_xy[correct].T, 'ko', mfc='none', ms=9, label='Dials (correct)')
        plt.plot(*dials_xy[~correct].T, 'ro', mfc='none', ms=9, label='Dials (incorrect)')
        plt.plot(*xy_outliers.T, 'bo', mfc='none', ms=9, label='Dials (outliers)')

        x = np.column_stack((dials_xy[:,0], precog_xy[:,0]))
        y = np.column_stack((dials_xy[:,1], precog_xy[:,1]))
        idx = np.linalg.norm(dials_xy - precog_xy, axis=-1) > 5.
        x,y = x[idx].T,y[idx].T
        plt.plot(x, y, '-k')
        plt.legend()
        embed(colors='linux')
        plt.show()

plt.figure()
plt.plot(np.arange(len(percent_correct)), percent_correct, label='Correct Inliers')
plt.plot(np.arange(len(percent_correct)), percent_outliers, label='Outliers')
plt.xlabel('Image Number')
plt.ylabel('Percent')
plt.title('Fraction Reflections Correctly Indexed by Image')
plt.legend()


plt.figure()
plt.plot(np.arange(len(nspots)), nspots, label='Strong Spots')
plt.plot(np.arange(len(nmatch)), nmatch, label='Matched to Precog')
plt.xlabel('Image Number')
plt.ylabel('Count')
plt.title('Spots per Image')
plt.legend()
plt.show()


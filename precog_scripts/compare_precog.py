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

expt_file = "dials_temp_files/optimized.expt"
refl_file = "dials_temp_files/optimized.refl"

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
elist = ExperimentListFactory.from_json_file(expt_file, False)
cryst = elist.crystals()[0]
unit_cell = cryst.get_unit_cell()
precog_df.spacegroup = gemmi.SpaceGroup(cryst.get_space_group().type().universal_hermann_mauguin_symbol())
precog_df.cell = gemmi.UnitCell(*unit_cell.parameters())
refls = reflection_table.from_file(refl_file)

print('generating DIALS dataframe')
dials_df = rs.DataSet({
    'X' : refls['xyzobs.px.value'].parts()[0].as_numpy_array(),
    'Y' : refls['xyzobs.px.value'].parts()[1].as_numpy_array(),
    'H' : refls['miller_index'].as_vec3_double().parts()[0].as_numpy_array(),
    'K' : refls['miller_index'].as_vec3_double().parts()[1].as_numpy_array(),
    'L' : refls['miller_index'].as_vec3_double().parts()[2].as_numpy_array(),
    'Wavelength' : refls['Wavelength'].as_numpy_array(),
    'BATCH' : refls['imageset_id'].as_numpy_array(),
}, cell = precog_df.cell, spacegroup=precog_df.spacegroup).infer_mtz_dtypes()


print('initializing metrics')
percent_correct = np.zeros(len(elist))
percent_outliers = np.zeros(len(elist))
percent_misindexed = np.zeros(len(elist))
nspots = np.zeros(len(elist))
nmatch = np.zeros(len(elist))


both_df = None

# Iterate by frame and match HKLs, seeing what percentage are correct
for i in trange(len(elist)):
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
        percent_misindexed[i] = 100.*sum(~correct)/len(correct)
        percent_correct[i] = 100.*sum(correct)/len(correct)

    nmatch[i] = len(im_dia)

    # Add this image to `both_df`
    im_pre.loc[:,['H', 'K', 'L']] = aligned_hkls
    im_pre.infer_mtz_dtypes(inplace=True)
    _both_df = im_dia.reset_index().join(im_pre.reset_index(), rsuffix='_pre', lsuffix='_dia')
    #_both_df = im_dia.join(im_pre.set_index(['H', 'K', 'L']), on=['H', 'K', 'L'], lsuffix='_dia', rsuffix='_pre')
    _both_df['correct'] = correct
    both_df = pd.concat((both_df, _both_df))

    if i == 0:
        x_diff = precog_xy[:,0] - dials_xy[:,0]
        y_diff = precog_xy[:,1] - dials_xy[:,1]
        
        plt.figure()
        plt.scatter(dials_xy[:,0][correct], dials_xy[:,1][correct], c='b', alpha=1, label='Correct')
        plt.scatter(dials_xy[:,0][~correct], dials_xy[:,1][~correct], c='r', alpha=1, label='Incorrect')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.title('DIALS vs Precog Spot Centroids (Single Image)')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(
            _both_df.loc[_both_df.correct, 'Wavelength_pre'].to_numpy(),
            _both_df.loc[_both_df.correct, 'Wavelength_dia'].to_numpy(),
            'k.',
            alpha=0.5,
        )
        plt.xlabel('$\lambda$ (Precognition)')
        plt.ylabel('$\lambda$ (DIALS)')
        plt.show()

plt.figure()
plt.plot(np.arange(len(percent_correct)), percent_correct)
plt.xlabel('Image Number')
plt.ylabel('Percent')
plt.title('Fraction Reflections Correctly Indexed by Image')
plt.show()

plt.figure()
plt.plot(np.arange(len(percent_correct)), percent_misindexed)
plt.xlabel('Image Number')
plt.ylabel('Percent')
plt.title('Fraction Reflections Misindexed by Image')
plt.show()
#plt.figure()
#plt.plot(np.arange(len(nspots)), nspots, label='Strong Spots')
#plt.plot(np.arange(len(nmatch)), nmatch, label='Matched to Precog')
#plt.xlabel('Image Number')
#plt.ylabel('Count')
#plt.title('Spots per Image')
#plt.legend()
#plt.show()

plt.figure()
plt.plot(
    both_df.loc[both_df.correct, 'Wavelength_pre'].to_numpy(),
    both_df.loc[both_df.correct, 'Wavelength_dia'].to_numpy(),
    'k.',
    alpha=0.1,
)
plt.xlabel('$\lambda$ (Precognition)')
plt.ylabel('$\lambda$ (DIALS)')
plt.show()

plt.figure()
c1,c2,c3 = "#1b9e77", "#d95f02", "#7570b3"
alpha = 0.1
cor = both_df.correct
plt.plot(both_df.loc[cor, 'H_pre'].to_numpy(), both_df.loc[cor, 'H_dia'].to_numpy(), '.', color=c1, label='H (correct)',  alpha=alpha)
plt.plot(both_df.loc[cor, 'K_pre'].to_numpy(), both_df.loc[cor, 'K_dia'].to_numpy(), '.', color=c2, label='K (correct)', alpha=alpha)
plt.plot(both_df.loc[cor, 'L_pre'].to_numpy(), both_df.loc[cor, 'L_dia'].to_numpy(), '.', color=c3, label='L (correct)', alpha=alpha)
plt.xlabel("Precognition")
plt.ylabel("DIALS")
plt.legend()
plt.show()

lam_diffs = both_df.loc[cor, 'Wavelength_dia'] - both_df.loc[cor, 'Wavelength_pre']
plt.figure()
plt.hist(lam_diffs, bins=100)
plt.xlabel('Wavelength Error (Angstroms)')
plt.ylabel('Number of Spots')
plt.show()

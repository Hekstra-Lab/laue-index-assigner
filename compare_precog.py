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

expt_file = "/home/rahewitt/dhfr_data/dials_temp_files/expected_index.expt"
refl_file = "optimized.refl"

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
dials_df['H'] = refls['miller_index'].parts()[0].as_numpy_array()
dials_df['K'] = refls['miller_index'].parts()[1].as_numpy_array()
dials_df['L'] = refls['miller_index'].parts()[2].as_numpy_array()
dials_df['Wavelength'] = refls['Wavelength'].as_numpy_array()
dials_df['BATCH'] = refls['shoebox'].bounding_boxes().parts()[4]

print('getting HKL matrices')
dials_hkl = dials_df[['H','K','L']].to_numpy().astype(int).T
precog_hkl = precog_df[['H','K','L']].to_numpy().astype(int).T

print('initializing metrics')
percent_correct = np.zeros(elist[0].imageset.size())

# Iterate by frame and match HKLs, seeing what percentage are correct
for i in tqdm(np.arange(elist[0].imageset.size())):
    # Get reflection indices from each batch
    precog_idx = (precog_df['BATCH'] == i).astype(bool)
    dials_idx = (dials_df['BATCH'] == i).astype(bool)

    # Get XY positions for refls
    dials_xy = dials_df[['X', 'Y']].to_numpy().astype(float)[dials_idx]
    precog_xy = precog_df[['X', 'Y']].to_numpy().astype(float)[precog_idx]

    # Filter out extraneous precog refls
    precog_filter = np.full(len(precog_xy), np.inf)
    for j in np.arange(len(dials_xy)):                                                  
        precog_filter[j] = np.argmin(np.linalg.norm(dials_xy[j,:] - precog_xy, axis=-1))
    precog_img_hkl = np.zeros((3,len(dials_hkl[:, dials_idx].T)))
    for j in np.arange(len(dials_xy[:, 0])):                                              
        precog_img_hkl[:,j] = precog_hkl.T[precog_idx, :][precog_filter[j].astype(int), :]

    # Align precog to DIALS hkls
    aligned_hkls = align_hkls(dials_hkl[:, dials_idx].T, precog_img_hkl.T, precog_df.spacegroup)

    # Check correctness of matching
    correct = is_ray_equivalent(aligned_hkls, dials_hkl[:,dials_idx].T)
    percent_correct[i] = sum(correct)/len(correct)

plt.plot(np.arange(len(percent_correct)), percent_correct)
plt.xlabel('Image Number')
plt.ylabel('Percentage of Reflections Indexed Correctly')
plt.title('Percent Reflections Correctly Indexed by Image')
plt.show()

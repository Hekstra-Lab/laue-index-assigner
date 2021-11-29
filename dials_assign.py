from diffgeolib import *
from scitbx import matrix
from IPython import embed
from tqdm import tqdm,trange
import numpy as np
import pandas as pd
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
from dials.array_family import flex

# Hyperparameters for indexer
macro_cycles = 3
lam_min = 0.95
lam_max = 1.15
d_min = 1.4
rlp_radius = 0.002
n_steps = 10
new_expt_filename = 'dials_temp_files/optimized.expt'
new_refl_filename = 'dials_temp_files/optimized.refl'

# Load DIALS files
expt_file = "dials_temp_files/stills_no_sb.expt"
refl_file = "dials_temp_files/stills_no_sb.refl"

elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)
refls = reflection_table.from_file(refl_file)

#This will populate refls['s1'] & refls['rlp']
refls.centroid_px_to_mm(elist)
refls.map_centroids_to_reciprocal_space(elist)

s0 = np.array(elist[0].beam.get_s0())

# Write to reflection file
refls['Wavelength'] = flex.double(len(refls))
refls['miller_index'] = flex.miller_index(len(refls))

# Loop over images and index by frame
for i in trange(elist[0].imageset.size()):
    # Get experiment data from experiment objects
    experiment = elist[i]
    cryst = elist.crystals()[0]
    spacegroup = gemmi.SpaceGroup(cryst.get_space_group().type().universal_hermann_mauguin_symbol())

    idx = refls['xyzobs.px.value'].parts()[2] - 0.5 == i
    subrefls = refls.select(idx)

    # Get crystal orientation
    A_mat = np.asarray(cryst.get_A()).reshape(3,3)
    cell_params = cryst.get_unit_cell().parameters()
    cell = gemmi.UnitCell(*cell_params)

    # Generate s vectors
    s1 = subrefls['s1'].as_numpy_array()

    # Generate assigner object
    la = LaueAssigner(s0, s1, cell, A_mat, lam_min, lam_max, d_min, spacegroup)

    la.assign()
    for j in range(macro_cycles):
        la.update_rotation()
        la.assign()
        la.reject_outliers()
        la.assign()
    la.reset_inliers()
    la.assign()

    refls['miller_index'].set_selected(
        idx, 
        flex.miller_index(la._H.astype('int').tolist()),
    )
    refls['Wavelength'].set_selected(
        idx, 
        flex.double(la._wav.tolist()),
    )


# Write out reflection file
refls.as_file(new_refl_filename)

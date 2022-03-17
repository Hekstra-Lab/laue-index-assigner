from diffgeolib import *
from scitbx import matrix
from tqdm import tqdm,trange
import numpy as np
import pandas as pd
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
from dials.array_family import flex
from cctbx.sgtbx import space_group
from cctbx.uctbx import unit_cell

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
print('Loading DIALS files')
expt_file = "dials_temp_files/stills_no_sb.expt"
refl_file = "dials_temp_files/stills_no_sb.refl"

elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)
refls = reflection_table.from_file(refl_file)

# This will populate refls['s1'] & refls['rlp']
refls.centroid_px_to_mm(elist)
refls.map_centroids_to_reciprocal_space(elist)

s0 = np.array(elist[0].beam.get_s0())

# Write to reflection file
refls['Wavelength'] = flex.double(len(refls))
refls['miller_index'] = flex.miller_index(len(refls))

# Loop over images and index by frame
print('Reindexing images')
for i in trange(len(elist.imagesets())):
    # Get experiment data from experiment objects
    experiment = elist[i]
    cryst = experiment.crystal
    spacegroup = gemmi.SpaceGroup(cryst.get_space_group().type().universal_hermann_mauguin_symbol())

    # Get reflections on this image
    idx = refls['id'] == i
    subrefls = refls.select(idx)

    # Get unit cell params
    cell_params = cryst.get_unit_cell().parameters()
    cell = gemmi.UnitCell(*cell_params)

    # Generate s vectors
    s1 = subrefls['s1'].as_numpy_array()

    # Get U matrix
    U = np.asarray(cryst.get_U()).reshape(3,3)

    # Generate assigner object
    la = LaueAssigner(s0, s1, cell, U, lam_min, lam_max, d_min, spacegroup)

    # Optimize Miller indices
    la.assign()
    for j in range(macro_cycles):
        la.update_rotation()
        la.assign()
        la.reject_outliers()
        la.assign()
    la.reset_inliers()
    la.assign()

    # Recalculate s1 based on new wavelengths
    s1 = la.s1 / la.wav[:,None]

    # Reset crystal parameters based on new geometry
    cryst.set_U(la.R.flatten())
    cryst.set_A(la.RB.flatten())
    cryst.set_B(la.B.flatten())
    cryst.set_space_group(space_group(la.spacegroup.hall))
    cryst.set_unit_cell(unit_cell(la.cell.parameters))

    # Write data to reflections
    refls['s1'].set_selected(
        idx,
        flex.vec3_double(s1)
    )
    refls['miller_index'].set_selected(
        idx, 
        flex.miller_index(la._H.astype('int').tolist()),
    )
    refls['Wavelength'].set_selected(
        idx, 
        flex.double(la._wav.tolist()),
    )

# Write out experiment file
print('Writing experiment data.')
elist.as_file(new_expt_filename)

# Write out reflection file
print('Writing reflection data.')
refls.as_file(new_refl_filename)

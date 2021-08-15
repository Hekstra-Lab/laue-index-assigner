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
lam_min = 0.95
lam_max = 1.3
d_min = 1.4
rlp_radius = 0.002
n_steps = 10
new_refl_filename = 'optimized.refl'

# Load DIALS files
expt_file = "dials_temp_files/refined_varying.expt" #<-- Fill this in
refl_file = "dials_temp_files/strong_1.04.refl" #<-- Fill this in

elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)
refls = reflection_table.from_file(refl_file)

# Get experiment data from experiment objects
experiment = elist[0]
cryst = elist.crystals()[0]
spacegroup = gemmi.SpaceGroup(cryst.get_space_group().type().universal_hermann_mauguin_symbol())
unit_cell = cryst.get_unit_cell()
detector = next(Detector.from_expt_file(expt_file))

#This will populate refls['s1'] & refls['rlp']
refls.centroid_px_to_mm(elist)
refls.map_centroids_to_reciprocal_space(elist)

s0 = np.array(elist[0].beam.get_s0())



# Write to reflection file
refls['Wavelength'] = flex.double(len(refls))
#refl_output['s1'] = flex.vec3_double(df[['s1_0','s1_1','s1_2']].to_numpy().tolist())
refls['miller_index'] = flex.miller_index(len(refls))

# Loop over images and index by frame
for i in trange(elist[0].imageset.size()):
    idx = refls['xyzobs.px.value'].parts()[2] - 0.5 == i
    subrefls = refls.select(idx)

    ##########################################################
    # This block computes the RUB matrices as defined int he dials conventions
    # We actually only need the rotation matrix R@U and the cell params for the assigner class
    # I am going to leave this here for now in case we need these patterns later
    ##########################################################
    # Get appropriate orientation matrix
    step = experiment.scan.get_oscillation()[1]
    gonio_setting_matrix = matrix.sqr(experiment.goniometer.get_setting_rotation())
    gonio_axis = matrix.col(experiment.goniometer.get_rotation_axis())

    #These are sorta defined here: https://dials.github.io/documentation/conventions.html
    R_mat = gonio_axis.axis_and_angle_as_r3_rotation_matrix(\
            angle=experiment.scan.get_angle_from_array_index(int(i)) + (step / 2),\
            deg=True,) * gonio_setting_matrix 
    B_mat = matrix.sqr(cryst.get_B_at_scan_point(int(i)))
    U_mat = matrix.sqr(cryst.get_U_at_scan_point(int(i)))

    #This is the trustworthy implementation from Aaron's code
    A_mat = gonio_axis.axis_and_angle_as_r3_rotation_matrix(\
            angle=experiment.scan.get_angle_from_array_index(int(i)) + (step / 2),\
            deg=True,) * gonio_setting_matrix * matrix.sqr(cryst.get_A_at_scan_point(int(i)))

    R = np.array(R_mat).reshape((3, 3)) @ np.array(U_mat).reshape((3, 3))
    B = np.array(B_mat).reshape((3, 3))
    RB = R@B

    RB_true = np.array(A_mat).reshape((3,3))

    cell_params = cryst.get_unit_cell_at_scan_point(int(i)).parameters()
    cell = gemmi.UnitCell(*cell_params)
    assert np.allclose(RB, RB_true)

    # Generate s vectors
    s1 = subrefls['s1'].as_numpy_array()

    #xy = subrefls['xyzobs.px.value'].as_numpy_array()[:,:2]
    #s1 = normalize(detector.pix2lab(xy))

    # Generate assigner object
    la = LaueAssigner(s0, s1, cell, R, lam_min, lam_max, d_min, spacegroup)

    la.assign()
    for _ in range(3):
        la.update_rotation()
        la.assign()
        la.reject_outliers()
        la.assign()

    refls['miller_index'].set_selected(
        idx, 
        flex.miller_index(la._H.astype('int').tolist()),
    )
    refls['Wavelength'].set_selected(
        idx, 
        flex.double(la._wav.tolist()),
    )

#This is obviously wrong? why are we doing this?
    #df.loc[['s1_0', 's1_1', 's1_2']] = la._H


# Write out reflection file
refls.as_file(new_refl_filename)

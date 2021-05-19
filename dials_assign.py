from diffgeolib import *
from scitbx import matrix
from IPython import embed
from tqdm import tqdm
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
#expt_file = "/home/rahewitt/dhfr_data/dials_temp_files/1ims/expected_index.expt"
#refl_file = "/home/rahewitt/dhfr_data/dials_temp_files/1ims/strong_1.070490100011937.refl"
#refl_file = "/home/rahewitt/dhfr_data/dials_temp_files/strong_1.070490100011937.refl"
#refl_file = "/home/rahewitt/laue_indexer/laue-index-assigner/dials_temp_files/strong_1.04.refl"
#refl_file = "data/strong.refl"
#expt_file = "/home/rahewitt/laue_indexer/laue-index-assigner/dials_temp_files/expected_index.expt"
expt_file = "/home/rahewitt/laue_indexer/laue-index-assigner/dials_temp_files/refined_varying.expt"
refl_file = "/home/rahewitt/laue_indexer/laue-index-assigner/dials_temp_files/strong_1.04.refl"
#expt_file = "/home/rahewitt/laue_indexer/laue-index-assigner/dials_temp_files/updated.expt"
#refl_file = "/home/rahewitt/laue_indexer/laue-index-assigner/dials_temp_files/refined_varying.refl"
#expt_file = "/home/rahewitt/Downloads/peak_refined.expt"
#refl_file = "/home/rahewitt/Downloads/peak_refined.refl"
#refl_file = "/home/rahewitt/dhfr_data/dials_temp_files/strong.refl"
elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)
refls = reflection_table.from_file(refl_file)
refls.split_partials_with_shoebox()

# Get experiment data from experiment objects
experiment = elist[0]
cryst = elist.crystals()[0]
unit_cell = cryst.get_unit_cell()
detector = next(Detector.from_expt_file(expt_file))

# Initialize dataframe to overwrite refl file data
df = pd.DataFrame(index=np.zeros(len(refls)), columns=['H', 'K', 'L', 's1_0', 's1_1', 's1_2','Wavelength'])

# Loop over images and index by frame
for i in tqdm(np.arange(elist[0].imageset.size())):
    # Get reflections for this image
#    idx = (refls['shoebox'].bounding_boxes().parts()[4] == i)
    embed()
    idx = ((refls['shoebox'].centroid_all().position_frame() - 0.5).round() == i)

    # Get pixel positions
    pix_position = refls['xyzobs.px.value'].as_numpy_array()[:, 0:2][idx]

    # Get appropriate orientation matrix
    step = experiment.scan.get_oscillation()[1]
    gonio_setting_matrix = matrix.sqr(experiment.goniometer.get_setting_rotation())
    gonio_axis = matrix.col(experiment.goniometer.get_rotation_axis())
    A_mat = gonio_axis.axis_and_angle_as_r3_rotation_matrix(\
            angle=experiment.scan.get_angle_from_array_index(int(i))+ (step / 2),\
            deg=True,)* gonio_setting_matrix* matrix.sqr(cryst.get_A_at_scan_point(int(i)))
    RB = np.array(A_mat).reshape((3,3))

    # Generate unit cell vectors
    O = np.array(unit_cell.orthogonalization_matrix()).reshape((3,3))
    abc = np.linalg.norm(O, axis=0)
    hmax = np.floor(abc/d_min)

    # Generate s vectors
    s0 = np.array([0., 0., -1.])
    s1 = detector.pix2lab(pix_position)
    s1 = s1/np.linalg.norm(s1, axis=1)[:,None]

    # Generate assigner object
    la = LaueAssigner(s0, s1, lam_min, lam_max, hmax, RB)
    
    # Optimize bases
    la.optimize_bases(rlp_radius, n_steps)

    # Update dataframe
    df['H'][idx] = la.H[:,0]
    df['K'][idx] = la.H[:,1]
    df['L'][idx] = la.H[:,2]
    df['s1_0'][idx] = la.H[:,0]
    df['s1_1'][idx] = la.H[:,1]
    df['s1_2'][idx] = la.H[:,2]
    df['Wavelength'][idx] = la.wavelengths

# Cast types in dataframe
df.H = pd.to_numeric(df.H)
df.K = pd.to_numeric(df.K)
df.L = pd.to_numeric(df.L)
df.s1_0 = pd.to_numeric(df.s1_0)
df.s1_1 = pd.to_numeric(df.s1_1)
df.s1_2 = pd.to_numeric(df.s1_2)
df.Wavelength = pd.to_numeric(df.Wavelength)

# Write to reflection file
refl_output = refls.copy()
refl_output['Wavelength'] = flex.double(df['Wavelength'])
refl_output['s1'] = flex.vec3_double(df[['s1_0','s1_1','s1_2']].to_numpy())
refl_output['miller_index'] = flex.vec3_double(df[['H','K','L']].to_numpy())

# Write out reflection file
refl_output.as_file(new_refl_filename)

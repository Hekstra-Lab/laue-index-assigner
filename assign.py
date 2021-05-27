from diffgeolib import *
from tqdm import trange,tqdm
from matplotlib import pyplot as plt
from scitbx import matrix
from IPython import embed
import numpy as np
import reciprocalspaceship as rs
import gemmi

print('started')

# These are the hyperparameters of the indexer =>
lam_min,lam_max = 0.95, 1.3
dmin = 1.4
rlp_radius = 0.002
nsteps = 10
# <= These are the hyperparameters of the indexer

ii_file  = "data/e080_003.mccd.ii"
inp_file = "data/e080_003.mccd.inp"

#expt_file = "data/split_000.expt" #<= output from the scan varying model we shipped to derek. 
expt_file = "/home/rahewitt/dhfr_data/dials_temp_files/1ims/expected_index.expt" #<= output from the scan varying model we shipped to derek. 
# ^^using goniometer maths, this could be replaced with the output of dials.index.
refl_file = "/home/rahewitt/dhfr_data/dials_temp_files/1ims/strong_1.070490100011937.refl"
#refl_file = "data/strong.refl" #<= `dials.find_spots {image_path}/*_001.mccd gain=0.1`
#refl_file = "/home/rahewitt/dhfr_data/dials_temp_files/1ims/expected_index.refl" #<= `dials.find_spots {image_path}/*_001.mccd gain=0.1`

ds = rs.read_precognition(ii_file).reset_index()

print('read precog .ii file')

from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
refls = reflection_table.from_file(refl_file)
train_pixpos = refls['xyzobs.px.value'].as_numpy_array()[:,:2]

elist = ExperimentListFactory.from_json_file(expt_file)
c = elist.crystals()[0]
uc = c.get_unit_cell()
ds.spacegroup = gemmi.SpaceGroup(c.get_space_group().type().universal_hermann_mauguin_symbol())
ds.cell = gemmi.UnitCell(*uc.parameters())

print('read DIALS files')

# Just sort and filter the integrated spots from precognition to match `refls`
# Note that this isn't perfect and sometimes it might happen that the rows in `refls` get assigned to the same entry in `ds`
pixpos = ds[['X', 'Y']].to_numpy(np.float32)
idx = np.argmin(np.linalg.norm(train_pixpos[:,None] - pixpos, axis=-1), axis=-1)
ds = ds.iloc[idx]
pixpos = ds[['X', 'Y']].to_numpy(np.float32)
hkl = ds[['H', 'K', 'L']].to_numpy(np.float32).T

print('sorted spots')

"""
RB = np.array(c.get_A()).reshape((3,3)) returns the reciprocal basis vectors

Reciprocal space basis vectors
------------------------
astar123 = [0]
bstar123 = [1]
cstar123 = [2]
"""
experiment = elist[0]
step = experiment.scan.get_oscillation()[1]
frame = 3
gonio_setting_matrix = matrix.sqr(experiment.goniometer.get_setting_rotation())
gonio_axis = matrix.col(experiment.goniometer.get_rotation_axis())
A_mat = gonio_axis.axis_and_angle_as_r3_rotation_matrix(angle=experiment.scan.get_angle_from_array_index(frame)- (step / 2), deg=True,)* gonio_setting_matrix* matrix.sqr(c.get_A())
RB = np.array(A_mat).reshape((3,3))
O = np.array(uc.orthogonalization_matrix()).reshape((3,3))
B = np.linalg.inv(O).T
R = RB@np.linalg.inv(B)

print('applied rotations')

# Detector.from_expt_file returns a generator of detector panels
detector = next(Detector.from_expt_file(expt_file))

s0 = np.array([0., 0., -1.])
s1 = detector.pix2lab(train_pixpos)
s1 = s1/np.linalg.norm(s1, axis=1)[:,None]
Q = (s1 - s0)

abc = np.linalg.norm(O, axis=0)
hmax = np.floor(abc/dmin)

print('beginning assignment')
la = LaueAssigner(s0, s1, lam_min, lam_max, hmax, RB)
print('finished assigning')

# Here we are going to compute the miller indices implied by dials's choice of 
# reciprocal bases. Then we are going to update the indexing solution from precog
# to match it.
lam = ds.Wavelength.to_numpy(np.float32)
href = (np.linalg.pinv(RB)@(Q.T/lam[None,:])).T
# ^^Technically you should just be able to use use np.linalg.inv but sometimes it gives me trouble so I just used the pseudo-inverse here for safety.
print('did this stuff')

#Put the precog output into the indexing solution chosen by dials
hkl = align_hkls(href, hkl.T, ds.spacegroup)
print('did hkls')
#assert np.allclose(href, hkl, atol=3.)

plt.figure()
plt.title("Millers Before Optimization")
correct = is_ray_equivalent(hkl, la.H)
plt.plot(train_pixpos[correct,0], train_pixpos[correct,1], 'k.', label='Correct')
plt.plot(train_pixpos[~correct,0], train_pixpos[~correct,1], 'r.', label='Incorrect')
plt.legend()

la.optimize_bases(rlp_radius, nsteps)

plt.figure()
correct = is_ray_equivalent(hkl, la.H)
plt.title("Millers After Optimization")
plt.plot(train_pixpos[correct,0], train_pixpos[correct,1], 'k.', label='Correct')
plt.plot(train_pixpos[~correct,0], train_pixpos[~correct,1], 'r.', label='Incorrect')
plt.legend()

plt.figure()
plt.plot(lam, la.wavelengths, 'k.')
plt.xlabel("Wavelength (Precognition)")
plt.ylabel("Wavelength")

plt.show()

from diffgeolib import *
from tqdm import trange,tqdm
from matplotlib import pyplot as plt
import numpy as np
import reciprocalspaceship as rs
import gemmi

# These are the hyperparameters of the indexer =>
lam_min,lam_max = 0.95, 1.3
dmin = 1.4
rlp_radius = 0.002
nsteps = 10
# <= These are the hyperparameters of the indexer

ii_file  = "data/e080_001.mccd.ii"
inp_file = "data/e080_001.mccd.inp"

expt_file = "data/split_000.expt" #<= output from the scan varying model we shipped to derek. 
# ^^using goniometer maths, this could be replaced with the output of dials.index.
refl_file = "data/strong.refl" #<= `dials.find_spots {image_path}/*_001.mccd gain=0.1`

ds = rs.read_precognition(ii_file).reset_index()


from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
refls = reflection_table.from_file(refl_file)
train_pixpos = refls['xyzobs.px.value'].as_numpy_array()[:,:2]

elist = ExperimentListFactory.from_json_file(expt_file)
c = elist.crystals()[0]
uc = c.get_unit_cell()
ds.spacegroup = gemmi.SpaceGroup(c.get_space_group().type().universal_hermann_mauguin_symbol())
ds.cell = gemmi.UnitCell(*uc.parameters())

# Just sort and filter the integrated spots from precognition to match `refls`
# Note that this isn't perfect and sometimes it might happen that the rows in `refls` get assigned to the same entry in `ds`
pixpos = ds[['X', 'Y']].to_numpy(np.float32)
idx = np.argmin(np.linalg.norm(train_pixpos[:,None] - pixpos, axis=-1), axis=-1)
ds = ds.iloc[idx]
pixpos = ds[['X', 'Y']].to_numpy(np.float32)
hkl = ds[['H', 'K', 'L']].to_numpy(np.float32).T

"""
RB = np.array(c.get_A()).reshape((3,3)) returns the reciprocal basis vectors

Reciprocal space basis vectors
------------------------
astar123 = [0]
bstar123 = [1]
cstar123 = [2]
"""
RB = np.array(c.get_A()).reshape((3,3))
O = np.array(uc.orthogonalization_matrix()).reshape((3,3))
B = np.linalg.inv(O).T
R = RB@np.linalg.inv(B)

# Detector.from_expt_file returns a generator of detector panels
detector = next(Detector.from_expt_file(expt_file))

s0 = np.array([0., 0., -1.])
s1 = detector.pix2lab(pixpos)
s1 = s1/np.linalg.norm(s1, axis=1)[:,None]
Q = (s1 - s0)

# Here we are going to compute the miller indices implied by dials's choice of 
# reciprocal bases. Then we are going to update the indexing solution from precog
# to match it.
lam = ds.Wavelength.to_numpy(np.float32)
href = (np.linalg.pinv(RB)@(Q.T/lam[None,:])).T
# ^^Technically you should just be able to use use np.linalg.inv but sometimes it gives me trouble so I just used the pseudo-inverse here for safety.

#Put the precog output into the indexing solution chosen by dials
hkl = align_hkls(href, hkl.T, ds.spacegroup)
assert np.allclose(href, hkl, atol=3.)

abc = np.linalg.norm(O, axis=0)
hmax = np.floor(abc/dmin)

la = LaueAssigner(s0, s1, lam_min, lam_max, hmax, RB)

plt.figure()
plt.title("Millers Before Optimization")
correct = is_ray_equivalent(hkl, la.H)
plt.plot(pixpos[correct,0], pixpos[correct,1], 'k.', label='Correct')
plt.plot(pixpos[~correct,0], pixpos[~correct,1], 'r.', label='Inorrect')
plt.legend()

la.optimize_bases(rlp_radius, nsteps)

plt.figure()
correct = is_ray_equivalent(hkl, la.H)
plt.title("Millers After Optimization")
plt.plot(pixpos[correct,0], pixpos[correct,1], 'k.', label='Correct')
plt.plot(pixpos[~correct,0], pixpos[~correct,1], 'r.', label='Inorrect')
plt.legend()

plt.figure()
plt.plot(lam, la.wavelengths, 'k.')
plt.xlabel("Wavelength (Precognition)")
plt.ylabel("Wavelength")

plt.show()

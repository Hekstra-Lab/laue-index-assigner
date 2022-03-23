from diffgeolib import LauePredictor
import numpy as np
import gemmi
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
from dials.array_family import flex
from dials.algorithms.spot_prediction import ray_intersection
from matplotlib import pyplot as plt
from IPython import embed

# Load DIALS files
expt_file = "dials_temp_files/mega_ultra_refined.expt"
refl_file = "dials_temp_files/mega_ultra_refined.refl"

# Get data
print('Loading DIALS files.')
elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)
refls = reflection_table.from_file(refl_file)

# Get experiment data from experiment objects
print('Getting experiment data.')
img_num = 0
i = 0
img = elist.imagesets()[img_num]
experiment = elist[0]
while(True): # Get first expt for this image
    experiment = elist[i]
    if(experiment.imageset == img):
        break
    i = i+1
cryst = experiment.crystal
spacegroup = gemmi.SpaceGroup(cryst.get_space_group().type().universal_hermann_mauguin_symbol())

# Get beam vector
s0 = np.array(experiment.beam.get_s0())

# Get unit cell params
cell_params = cryst.get_unit_cell().parameters()
cell = gemmi.UnitCell(*cell_params)

# Get U matrix
U = np.asarray(cryst.get_U()).reshape(3,3)

# Get observed centroids
sub_refls = refls.select(refls['imageset_id'] == img_num)
xobs = sub_refls['xyzobs.mm.value'].parts()[0].as_numpy_array() 
yobs = sub_refls['xyzobs.mm.value'].parts()[1].as_numpy_array() 

# Wavelengths per spot
lams = sub_refls['Wavelength'].as_numpy_array()

# Hyperparameters for predictor
lam_min = np.min(lams)
lam_max = np.max(lams)
d_min = 1.4 # TODO: What's a good value or calculation for this?

# Get s1 vectors
print('Predicting s1 vectors.')
la = LauePredictor(s0, cell, U, lam_min, lam_max, d_min, spacegroup)
s1 = la.predict_s1()

# Build new reflection table for predictions
preds = reflection_table.empty_standard(len(s1))

# Populate s1 and phi columns
preds['s1'] = flex.vec3_double(s1)
preds['phi'] = flex.double(np.zeros(len(s1))) # Data are stills

# Get which reflections intersect detector
print('Getting centroids.')
intersects = ray_intersection(experiment.detector, preds)
preds = preds.select(intersects)

# Get predicted centroids
x = preds['xyzcal.mm'].parts()[0].as_numpy_array()
y = preds['xyzcal.mm'].parts()[1].as_numpy_array()

# Plot image
print('Plotting data.')
plt.scatter(x, y, c='r', alpha=0.5)
plt.scatter(xobs, yobs, c='b', alpha=0.5)
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('Predicted vs observed centroids')
plt.legend(labels=['Predicted', 'Observed'])
plt.show()

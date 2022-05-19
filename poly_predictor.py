from diffgeolib import LauePredictor
import numpy as np
import gemmi
from dxtbx.model import ExperimentList
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
from dials.array_family import flex
from dials.algorithms.spot_prediction import ray_intersection
from matplotlib import pyplot as plt
import scipy
from gen_lam_func import gen_kde
from tqdm import tqdm, trange
from IPython import embed

# Load DIALS files
expt_file = "dials_temp_files/shrunk.expt"
refl_file = "dials_temp_files/shrunk.refl"
#expt_file = "dials_temp_files/mega_ultra_refined.expt"
#refl_file = "dials_temp_files/mega_ultra_refined.refl"

# Get data
print('Loading DIALS files.')
elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)
refls = reflection_table.from_file(refl_file)

# Remove outliers
print('Removing outliers')
idx = refls.get_flags(refls.flags.used_in_refinement).as_numpy_array()
idy = np.arange(len(elist))[idx].tolist()
elist = ExperimentList([elist[i] for i in idy])
refls = refls.select(flex.bool(idx))

# Set up reflection table to store valid predictions
final_preds = reflection_table()

# Get experiment data from experiment objects
print('Making predictions per image')
for img_num in trange(len(elist.imagesets())):
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
    lam_min = 0.95
    lam_max = 1.25
    d_min = 1.4 # TODO: What's a good value or calculation for this?
    
    # Get s1 vectors
    la = LauePredictor(s0, cell, U, lam_min, lam_max, d_min, spacegroup)
    s1, new_lams, q_vecs, millers = la.predict_s1()

    # Build new reflection table for predictions
    preds = reflection_table.empty_standard(len(s1))
    del preds['intensity.prf.value']
    del preds['intensity.prf.variance']
    del preds['intensity.sum.value']
    del preds['intensity.sum.variance']
    del preds['lp']
    del preds['profile_correlation']
    
    # Populate needed columns
    preds['id'] = flex.int([int(experiment.identifier)]*len(preds)) # As long as wavelength doesn't matter...
    preds['imageset_id'] = flex.int([img_num]*len(preds))
    preds['s1'] = flex.vec3_double(s1)
    preds['phi'] = flex.double(np.zeros(len(s1))) # Data are stills
    preds['wavelength'] = flex.double(new_lams)
    preds['rlp'] = flex.vec3_double(q_vecs)
    preds['miller_index'] = flex.miller_index(millers.astype('int').tolist()) 

    # Get which reflections intersect detector
    intersects = ray_intersection(experiment.detector, preds)
    preds = preds.select(intersects)
    new_lams = new_lams[intersects]
    
    # Generate a KDE
    _, _, kde = gen_kde(elist, refls)
    
    # Get predicted centroids
    x = preds['xyzcal.mm'].parts()[0].as_numpy_array()
    y = preds['xyzcal.mm'].parts()[1].as_numpy_array()
    
    # Get probability densities for predictions:
    rlps = preds['rlp'].as_numpy_array()
    norms = (np.linalg.norm(rlps, axis=1))**2
    pred_data = [norms, new_lams]
    probs = kde.pdf(pred_data)
    
    # Cut off using log probabilities
    cutoff_log = 0 # TODO Make this a hyperparameter
    sel = np.log(probs) >= cutoff_log
    x_sel = x[sel]
    y_sel = y[sel]
    probs_sel = probs[sel]
    preds = preds.select(flex.bool(sel))

    # Append image predictions to dataset
    final_preds.extend(preds)
    
    ## Plot image
    #print('Plotting data.')
    #cmap = plt.cm.get_cmap('viridis')
    #plt.scatter(xobs, yobs, c='k', marker='x', s=8, alpha=0.9)
    #plt.scatter(x, y, c=probs, cmap=cmap, s=2, alpha=1)
    #plt.colorbar()
    #plt.xlabel('x (mm)')
    #plt.ylabel('y (mm)')
    #plt.title('Predicted vs observed centroids')
    #plt.legend(labels=['Predicted', 'Observed'])
    #plt.show()
    #
    #cmap = plt.cm.get_cmap('viridis')
    #plt.scatter(xobs, yobs, c='k', marker='x', s=8, alpha=0.9)
    #plt.scatter(x_sel, y_sel, c=probs_sel, cmap=cmap, s=2, alpha=1)
    #plt.colorbar()
    #plt.xlabel('x (mm)')
    #plt.ylabel('y (mm)')
    #plt.title('Filtered Predicted vs observed centroids')
    #plt.legend(labels=['Predicted', 'Observed'])
    #plt.show()
    #
    #cutoffs = np.linspace(min(probs), max(probs), 100)
    #accepted = np.zeros(len(cutoffs))
    #for i,j in enumerate(cutoffs):   
    #    accepted[i] = sum(probs >= j)
    #plt.scatter(cutoffs, accepted, s=2, c='k')
    #plt.xlabel('cutoff prob. density')
    #plt.ylabel('# accepted spots')
    #plt.show()
    
    # Write data

final_preds.as_file('dials_temp_files/predicted.refl')

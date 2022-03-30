"""
This script generates the lambda-curve for the experiment.
"""
import numpy as np
from dxtbx.model import ExperimentList
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
from dials.array_family import flex
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import scipy
import itertools as it
from IPython import embed

def gen_kde(elist, refls):
    """ This function trains a Gaussian KDE on 1/d^2 and wavelengths of submitted strong spots"""
    # Firstly remove all harmonic reflections from consideration TODO

    # Get rlps and normalize
    print('Calculating rlps')
    refls.map_centroids_to_reciprocal_space(elist)
    rlps = refls['rlp'].as_numpy_array()
    norms = np.linalg.norm(rlps, axis=1)
    
    # Get wavelength of rlp
    print('Getting wavelengths')
    lams = refls['Wavelength'].as_numpy_array()
    
    # Fit with kernel density estimator
    print('Training KDE')
    normalized_resolution = norms**2
    train_data = np.asarray([normalized_resolution, lams])
    kde = scipy.stats.gaussian_kde(train_data)
    return normalized_resolution, lams, kde

def plot_kde():
    # Load files
    print('Loading DIALS files')
    expt_file = "dials_temp_files/mega_ultra_refined.expt"
    refl_file = "dials_temp_files/mega_ultra_refined.refl"
    elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)
    refls = reflection_table.from_file(refl_file)
    
    # Remove outliers                                                     
    print('Removing outliers')
    idx = refls.get_flags(refls.flags.used_in_refinement).as_numpy_array()
    idy = np.arange(len(elist))[idx].tolist()                             
    elist = ExperimentList([elist[i] for i in idy])                       
    refls = refls.select(flex.bool(idx))
    
    # Generate KDE
    normalized_resolution, lams, kde = gen_kde(elist, refls)
    
    # Make a mesh
    print('Making a meshgrid')
    N = 50
    x = np.linspace(min(normalized_resolution), max(normalized_resolution), N)
    y = np.linspace(min(lams), max(lams), N)
    z = np.zeros(shape=(len(x), len(y)))
    zeros = np.zeros(len(x))
    
    # Evaluate PDF on mesh
    for x0, y0 in it.product(x, y):
        i = np.where(x == x0)[0][0]
        j = np.where(y == y0)[0][0]
        z[i, j] = kde.pdf([x0, y0])
    
    # Plot a wireframe
    print('Plotting')
    norm = plt.Normalize(z.min(), z.max())
    colors = cm.viridis(norm(z))
    rcount, ccount, _ = colors.shape
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surface = ax.plot_surface(x[:, None] + zeros, y[None, :] + zeros, z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
    surface.set_facecolor((0,0,0,0))
    plt.xlabel('$1/d^2$ (A$^{-2}$)')
    plt.ylabel('$\lambda$ (A)')
    plt.title('PDF of Reflections')
    plt.show()

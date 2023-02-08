from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
from dials.algorithms.shoebox import MaskCode
import numpy as np
from matplotlib import pyplot as plt
from IPython import embed

# Load files
elist = ExperimentListFactory.from_json_file('dials_temp_files/integrated.expt', check_format=True)
refls = reflection_table.from_file('dials_temp_files/integrated.refl')

# Get MaskCode
foreground_code = int(MaskCode.Foreground | MaskCode.Valid)

# Turn off interactive mode
plt.ioff()

# Track median,mean intensities per image
means = np.zeros(len(elist.imagesets()))
medians = np.zeros(len(elist.imagesets()))

# Plot integration masks on each image
for i, img_set in enumerate(elist.imagesets()):
    # Get image intensity data
    img_refls = refls.select(refls['imageset_id'] == i)
    img_I = img_refls['intensity.sum.value'].as_numpy_array()
    means[i] = np.sum(img_I) / len(img_refls)
    medians[i] = np.median(img_I)

    # Get shoebox data
    img = img_set[0][0]
    img_refls = refls.select(refls['imageset_id'] == i)
    foreground_coords = []
    # Get shoebox data
    for sbox in img_refls['shoebox']:
        coords = sbox.coords().select(sbox.mask.as_1d() == foreground_code)
        coords = coords.as_numpy_array()
        foreground_coords.append(coords)
    x,y,_ = np.vstack(foreground_coords).T
    # Get pixel data
    pix_values = img.as_numpy_array()
    # Plot
    plt.plot(x, y, 'k.')
    plt.matshow(np.log1p(pix_values), vmin=2, vmax=4, fignum=0)
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    plt.title(f'Integration Masks on Image {i}')
    plt.colorbar()
    plt.show()

plt.plot(means, 'r', label='Means')
plt.plot(medians, 'b', label='Medians')
plt.xlabel('Imageset')
plt.ylabel('Intensity per Reflection')
plt.legend()
plt.show()

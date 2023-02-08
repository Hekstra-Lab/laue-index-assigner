"""
For each reflection, overwrites the 'Wavelength' column with the appropriate wavelength
given by the experiment beam.
"""

import numpy as np
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
from dials.array_family import flex
from tqdm import tqdm, trange
import argparse                                                             
                                                                            
# Get I/O options from user                                                 
parser = argparse.ArgumentParser()                                          
parser.add_argument('image_num', type=int, help='Image number to analyze.') 
args = parser.parse_args()                                                  
                                                                            
# Set parameters                                                            
image_analyzed = args.image_num                                             

# Parse arguments for filenames
expt_file = f'dials_temp_files/refinement/mega_ultra_refined{image_analyzed:06d}.expt'
refl_file = f'dials_temp_files/refinement/mega_ultra_refined{image_analyzed:06d}.refl'

# Load DIALS files
elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)
refls = reflection_table.from_file(refl_file)

# Get experiment ID per reflection
idx = refls['id'].as_numpy_array()

# Initialize wavelength array
lams = np.zeros(len(idx))

# For each beam, set all corresponding wavelengths in the reflection table to beam value
for i in trange(len(idx)):
    lams[i] = elist[idx[i]].beam.get_wavelength()

# Store lams back into refl file
refls['Wavelength'] = flex.double(lams)

# Overwrite refl file
refls.as_file(refl_file)

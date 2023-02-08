from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory
from dxtbx.model import ExperimentList
from dials.array_family.flex import reflection_table
import numpy as np
import argparse
from tqdm import tqdm, trange
from IPython import embed

# Get I/O options from user
print('Parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument('in_expt', help='Input experiment file.')
parser.add_argument('in_refl', help='Input reflection file.')
args = parser.parse_args()

# Get data
print('Loading data')
expt_filename = args.in_expt
refl_filename = args.in_refl
expts = ExperimentListFactory.from_json_file(expt_filename)
refls = flex.reflection_table().from_file(refl_filename)
begin = 0
end = len(expts.imagesets())

# Sanity checks
print('Running sanity checks')
if(end == 1):
    print('Dataset has only one image.')
    exit()

# Get new expts, refls for range of images
print('Generating files')
imgs = expts.imagesets()
img_nums = np.arange(end)
next_expt_idx = 0 # To prevent sampling already finished experiments
img_ids = refls['imageset_id'].as_numpy_array()
for img_num in trange(begin, end):
    idx = img_nums == img_num # Get only one image
    new_expts = ExperimentList()
    for i in range(next_expt_idx, len(expts)): # Skip previous images
        expt = expts[i]
        if(idx[imgs.index(expt.imageset)]):
            new_expts.append(expt)
            next_expt_idx = i + 1
        else: # Should occur once done with expts for this image
            break

    new_refls = refls.select(flex.bool(refls['imageset_id'] == img_num))
    
    # Write data
    print('Writing data')
    new_expts.as_file(f'dials_temp_files/split_images/split{img_num:06d}.expt')
    new_refls.as_file(f'dials_temp_files/split_images/split{img_num:06d}.refl')

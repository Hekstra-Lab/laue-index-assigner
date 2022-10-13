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
parser.add_argument('img1', help='First image for output block.', type=int)
parser.add_argument('img2', help='Final (inclusive) image for output block.', type=int)
parser.add_argument('in_expt', help='Input experiment file.')
parser.add_argument('in_refl', help='Input reflection file.')
args = parser.parse_args()

# Get data
print('Loading data')
expt_filename = args.in_expt
refl_filename = args.in_refl
expts = ExperimentListFactory.from_json_file(expt_filename)
refls = flex.reflection_table().from_file(refl_filename)
begin = min(args.img1, args.img2)
end = max(args.img1, args.img2)

# Sanity checks
print('Running sanity checks')
if(len(expts.imagesets()) < end):
    print('There are not that many images.')
    exit()
if(begin < 0):
    print('Images are 0-indexed. Please start with a valid image')
    exit()

# Get new expts, refls for range of images
img_nums = np.arange(len(expts.imagesets()))
idx = (img_nums >= begin) & (img_nums <= end)
print('Making new experiment list')
imgs = expts.imagesets()
new_expts = ExperimentList()
for i in trange(len(expts)):
    expt = expts[i]
    if(idx[imgs.index(expt.imageset)]):
        new_expts.append(expt)
print('Making new reflection table')
idz = [False]*len(refls)
for i in trange(len(refls)):
    if(idx[refls['imageset_id'].as_numpy_array()[i]]):
        idz[i] = True
new_refls = refls.select(flex.bool(idz))

# Write data
print('Writing data')
new_expts.as_file('dials_temp_files/shrunk.expt')
new_refls.as_file('dials_temp_files/shrunk.refl')

for i in trange(3048):
  expt_filename = f'split{i:06}.expt'
  refl_filename = f'split{i:06}.refl'
  expts = ExperimentListFactory.from_json_file(expt_filename)
  refls = flex.reflection_table().from_file(refl_filename)
  min_id = int(expts.identifiers()[0])
  min_img_id = int(np.min(refls['imageset_id'].as_numpy_array()))
  for j in range(len(expts)):
    expts[j].identifier = str(j)
  idx = refls['id'].as_numpy_array() - min_id
  refls['id'] = flex.int(idx)
  idy = refls['imageset_id'].as_numpy_array() - min_img_id
  refls['imageset_id'] = flex.int(idy)
  expts.as_file(f'/n/home04/rahewitt/laue_indexer/laue-index-assigner/dials_temp_files/split_images/split{i:06}.expt')
  refls.as_file(f'/n/home04/rahewitt/laue_indexer/laue-index-assigner/dials_temp_files/split_images/split{i:06}.refl')

import numpy as np
from dials.algorithms.shoebox import MaskCode
from dials.model.data import Shoebox
from dials.array_family import flex
import int_methods
from dxtbx.model import ExperimentList
from dxtbx.model.experiment_list import ExperimentListFactory
import logging
from tqdm import tqdm, trange
from utils.refl_table_utils import gen_experiment_identifiers
from IPython import embed

#logging.basicConfig(level=logging.DEBUG)
print("Load DIALS files")
elist = ExperimentListFactory.from_json_file("dials_temp_files/mega_ultra_refined.expt", check_format=True)
refls = flex.reflection_table.from_file("dials_temp_files/mega_ultra_refined.refl")
preds = flex.reflection_table.from_file("dials_temp_files/predicted.refl")
phil_file = "proc_sigb.phil"

# Get only refined reflections
idx = refls.get_flags(refls.flags.used_in_refinement)
refls = refls.select(idx)
refls.reset_ids()
preds.reset_ids()

print("Populating columns in predictions")
preds['flags'] = flex.size_t(len(preds), 1)
preds['entering'] = flex.bool(len(preds), False)
preds['delpsical.rad'] = flex.double(len(preds), 0)
preds["xyzobs.mm.value"] = preds["xyzcal.mm"]
if "phi" in list(preds.keys()):
    del preds['phi']
px = elist[0].detector[0].get_pixel_size()[0]
calpx = flex.vec3_double([ (x/px, y/px, 0.5) for x,y,_ in preds["xyzcal.mm"]])
preds["xyzcal.px"] = calpx
preds["xyzobs.px.value"] = calpx # TODO these aren't observed though?
if 'q' in list(preds.keys()):   # integrator looks for qvecs in rlp (not q)
	preds['rlp'] = preds['q']
if "shoebox" in list(preds.keys()):
    del preds["shoebox"]

print("Populating bounding boxes in strong reflections")
new_bb = flex.int6([(x1,x2,y1,y2,0,1) for x1,x2,y1,y2,_,_ in refls['bbox']])
refls["bbox"] = new_bb
if "shoebox" in list(refls.keys()):
    del refls["shoebox"]

print("Mapping reflections to experiment identifiers")
refls = gen_experiment_identifiers(refls, elist)
preds = gen_experiment_identifiers(preds, elist)

print("Integrating reflections")
new_elist, new_refls = int_methods.integrate(phil_file, elist, refls, preds)

print("Writing experiment data.")
new_elist.as_file('dials_temp_files/integrated.expt')

print("Writing reflection data.")
new_refls.as_file('dials_temp_files/integrated.refl')

print("Finished!")

from argparse import ArgumentParser

# LIBTBX_SET_DISPATCHER_NAME ricks.image_viewer 

parser  = ArgumentParser()
parser.add_argument("--refl", type=str, help="path to refl mega_ultra_file")
parser.add_argument("--expt", type=str, help="path to expt list")
parser.add_argument("--split", action="store_true", help="separate inliers/outliers")
args = parser.parse_args()

import os
from dxtbx.model import ExperimentList
from dials.array_family import flex

R = flex.reflection_table.from_file(args.refl)
R['id'] = flex.int(len(R),0)
eid = R.experiment_identifiers()
for k in eid.keys():
    del eid[k]

El = ExperimentList.from_file(args.expt, False)
expt = El[0]
wave = sum([b.get_wavelength() for b in El.beams()]) / len(El)
expt.beam.set_wavelength(wave)


El = ExperimentList()
El.append(expt)


if args.split:
    inliers = R.get_flags(R.flags.used_in_refinement)
    outliers = flex.bool([not fl for fl in inliers])

    Rin = R.select(inliers)
    Rout = R.select(outliers)

    Rin['id'] = flex.int(len(Rin), 1)
    Rout['id'] = flex.int(len(Rout), 2)
    R.extend(Rin)
    R.extend(Rout)
    # copy expt twice
    from copy import deepcopy
    expt_in = deepcopy(expt)
    expt_in.identifier='1'
    expt_out = deepcopy(expt)
    expt_out.identifier='2'
    El.append(expt_in)
    El.append(expt_out)


# write new
dummie = "_ricks_temps_for_image_viewer.expt"
dummie_refl = "_ricks_temps_for_image_viewer.refl"
El.as_file(dummie)
R.as_file(dummie_refl)

fnames = dummie, dummie_refl

opts ="show_max_pix=False show_all_pix=False show_shoebox=False show_ctr_mass=False brightness=100"
os.system("dials.image_viewer %s %s %s" % ( fnames+ (opts,)))

for f in fnames:
    os.remove(f)

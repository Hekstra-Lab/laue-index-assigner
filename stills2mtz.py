#!/usr/bin/env cctbx.python
from dials.array_family import flex
from cctbx import sgtbx
from dxtbx.model.experiment_list import ExperimentListFactory 
from glob import glob
from os.path import exists,abspath
import numpy as np
import argparse
import reciprocalspaceship as rs
import pandas as pd
import gemmi
import re
import sys

expt_files = [f"{sys.argv[1]}/integrated_from_int_test.expt",]
refl_files = [f"{sys.argv[1]}/integrated_from_int_test.refl",]

def make_annotated_mtz(exptFN, reflFN):
    """ Make an MTZ file from DIALS expt and refl files for scaling/merging."""
    # Get DIALS data
    table = flex.reflection_table().from_file(reflFN)
    elist = ExperimentListFactory.from_json_file(exptFN, check_format=False)

    # Get miller indices
    h = table["miller_index"].as_vec3_double()

    # Get a gemmi cell
    cell = np.zeros(6)
    for crystal in elist.crystals():
        cell += np.array(crystal.get_unit_cell().parameters())/len(elist.crystals())
    cell = gemmi.UnitCell(*cell)

    # Get a spacegroup
    sginfo = elist.crystals()[0].get_space_group().info()
    symbol = sgtbx.space_group_symbols(sginfo.symbol_and_number().split('(')[0]) #<--- this cannot be the 'correct' way to do this
    spacegroup = gemmi.SpaceGroup(symbol.universal_hermann_mauguin())

    data = rs.DataSet({
      'H' : h.as_numpy_array()[:,0].astype(np.int32), 
      'K' : h.as_numpy_array()[:,1].astype(np.int32), 
      'L' : h.as_numpy_array()[:,2].astype(np.int32), 
      'BATCH' : table['imageset_id'].as_numpy_array() + 1,
      'I' : table['intensity.sum.value'].as_numpy_array(),
      'SIGI' : table['intensity.sum.variance'].as_numpy_array()**0.5,
      'xobs' : table['xyzobs.px.value'].as_numpy_array()[:,0], 
      'yobs' : table['xyzobs.px.value'].as_numpy_array()[:,1], 
      'wavelength' : table['wavelength'].as_numpy_array(),
      'BG' : table['background.sum.value'].as_numpy_array(),
      'SIGBG' : table['background.sum.variance'].as_numpy_array()**0.5
    }, cell=cell, spacegroup=spacegroup).infer_mtz_dtypes()
    return data

data = None
cell = None
for i, (expt,refl) in enumerate(zip(expt_files, refl_files)):
    ds = make_annotated_mtz(expt, refl)
    if cell is None:
        cell = ds.cell
    else:
        ds.cell = cell
    #ds.copy().write_mtz(f'unmerged_{i}.mtz', skip_problem_mtztypes=True)
    if data is not None:
        ds['BATCH'] = ds['BATCH'] + data['BATCH'].max()
    data = rs.concat((ds, data), check_isomorphous=False)

data.write_mtz(f"{sys.argv[1]}/integrated_from_int_test.mtz", skip_problem_mtztypes=True)


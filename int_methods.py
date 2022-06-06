import numpy as np
from dials.array_family import flex
from simtbx.diffBragg import utils
from simtbx.modeling import predictions
import glob
import pandas
import os
import shutil
from scipy.interpolate import interp1d
from dxtbx.model import ExperimentList
from dials.algorithms.integration.stills_significance_filter import SignificanceFilter
from dials.algorithms.indexing.stills_indexer import calc_2D_rmsd_and_displacements
from dials.model.data import Shoebox
from dials.algorithms.shoebox import MaskCode
from tqdm import tqdm, trange
from copy import deepcopy
import sys

# Note: these imports and following 3 methods will eventually be in CCTBX/simtbx/diffBragg/utils
from dials.algorithms.spot_finding.factory import SpotFinderFactory
from dials.algorithms.spot_finding.factory import FilterRunner
from dials.model.data import PixelListLabeller, PixelList
from dials.algorithms.spot_finding.finder import pixel_list_to_reflection_table
from libtbx.phil import parse
from dials.command_line.stills_process import phil_scope

from dials.algorithms.integration.integrator import create_integrator
from dials.algorithms.profile_model.factory import ProfileModelFactory
from dials.algorithms.profile_model.gaussian_rs.model import Model
from dxtbx.model import ExperimentList
from dials.array_family import flex
from IPython import embed

def stills_process_params_from_file(phil_file):
    """
    :param phil_file: path to phil file for stills_process
    :return: phil params object
    """
    phil_file = open(phil_file, "r").read()
    user_phil = parse(phil_file)
    phil_sources = [user_phil]
    working_phil, unused = phil_scope.fetch(
        sources=phil_sources, track_unused_definitions=True)
    params = working_phil.extract()
    return params

def process_reference(reference):
    """Load the reference spots."""
    assert "miller_index" in reference
    assert "id" in reference
    refls = reference.get_flags(reference.flags.used_in_refinement)
    rubbish = reference.select(~refls)
    if refls.count(False) > 0:
        reference.del_selected(~refls)
    if len(reference) == 0:
        raise RuntimeError(
            """
    Invalid input for reference reflections.
    Expected > %d indexed spots, got %d
  """
            % (0, len(reference))
        )
    unindexed = reference["miller_index"] == (0, 0, 0)
    if unindexed.count(True) > 0:
        rubbish.extend(reference.select(unindexed))
        reference.del_selected(unindexed)
    invalid = reference["id"] < 0
    if invalid.count(True) > 0:
        raise RuntimeError(
            """
    Invalid input for reference reflections.
    %d reference spots have an invalid experiment id
  """
            % invalid.count(True)
        )
    return reference, rubbish

def integrate(phil_file, experiments, indexed, predicted):
    """
    integrate a single experiment at the locations specified by the predicted table
    The predicted table should have a column specifying strong reflections
    """
    params = stills_process_params_from_file(phil_file) 
    indexed,_ = process_reference(indexed)
    #experiments = ProfileModelFactory.create(params, experiments, indexed)

    new_reflections = flex.reflection_table()
    #for expt_id, expt in enumerate(experiments):
    #    if(expt.profile.sigma_b() < params.profile.gaussian_rs.parameters.sigma_b_cutoff):
    #        refls = indexed.select(indexed["id"] == expt_id+1) # 1-indexed
    #        if(len(refls) == 0):
    #            continue
    #        boxes = expt.profile.compute_bbox(refls, expt.crystal, expt.beam, expt.detector) # Generate bounding boxes
    #        refls['bbox'] = boxes
    #        new_reflections.extend(refls)

    #indexed = new_reflections
    #if len(experiments) == 0:
    #    raise RuntimeError("No experiments after filtering by sigma_b")
    #boxes = expt.profile.compute_bbox(refls, expt.crystal, expt.beam, expt.detector) # Generate bounding boxes
    #refls['bbox'] = boxes

    # Create shoeboxes
    shoeboxes = []
    curr_iset = indexed[0]['imageset_id']
    img_data = experiments.imagesets()[curr_iset].get_raw_data(0)
    img_data = np.array([data.as_numpy_array() for data in img_data])
    print("Creating shoeboxes for profile fitting")
    for i in trange(len(indexed)):
        x1, x2, y1, y2, z1, z2 = indexed[i]['bbox']
        sbox = Shoebox(indexed[i]['bbox']) 
        _curr_iset = indexed[i]['imageset_id']
        if _curr_iset != curr_iset:
            curr_iset = _curr_iset
            img_data = experiments.imagesets()[curr_iset].get_raw_data(0)
            img_data = np.array([data.as_numpy_array() for data in img_data])
        refl_data = img_data[0, y1:y2, x1:x2].reshape((1, y2-y1, x2-x1)) # Assumes single-panel detector
        sbox.data = flex.float(np.ascontiguousarray(refl_data))
        sbox.background = flex.float(np.zeros_like(refl_data))
        mask = np.full(shape = refl_data.shape, fill_value=MaskCode.Valid, dtype=np.int)
        sbox.mask = flex.int(np.ascontiguousarray(mask))
        shoeboxes.append(sbox)
    indexed['shoebox'] = flex.shoebox(shoeboxes)

    # Integrate by image
    print("Integrating reflections by image")
    predicted.match_with_reference(indexed)
    integrated = flex.reflection_table()
    new_expts = ExperimentList()
    sigma_bs = np.zeros(len(experiments))
    sigma_ms = np.zeros(len(experiments))
    for i in trange(len(experiments.imagesets())): # Loop over images
        img_predicted = predicted.select(predicted['imageset_id'] == i)
        if len(img_predicted) == 0:
            continue
        expt_ids = img_predicted.experiment_identifiers().values()
        img_experiments = deepcopy(experiments)
        img_experiments.select_on_experiment_identifiers(list([expt_ids[0]]))
        img_experiments[0].identifier = expt_ids[0]

        img_experiments[0].profile = Model.create_from_reflections(
            params.profile, 
            indexed.select(indexed['imageset_id'] == i), 
            img_experiments[0].crystal, 
            img_experiments[0].beam, 
            img_experiments[0].detector
        )
        sigma_bs[i] = img_experiments[0].profile.to_dict()['sigma_b']
        sigma_ms[i] = img_experiments[0].profile.to_dict()['sigma_m']
        img_predicted.reset_ids()
        img_integrator = create_integrator(params, img_experiments, img_predicted)
        img_integrated = img_integrator.integrate()
        if i > 0:
            img_integrated.experiment_identifiers()[0] = integrated.experiment_identifiers().values()[0]
        integrated.extend(img_integrated)
        new_expts.append(img_experiments[0])

    #if params.significance_filter.enable:

    #    sig_filter = SignificanceFilter(params)
    #    filtered_refls = sig_filter(experiments, integrated)
    #    accepted_expts = ExperimentList()
    #    accepted_refls = flex.reflection_table()
    #    for expt_id, expt in enumerate(experiments):
    #        refls = filtered_refls.select(filtered_refls["id"] == expt_id)
    #        if len(refls) > 0:
    #            accepted_expts.append(expt)
    #            refls["id"] = flex.int(len(refls), len(accepted_expts) - 1)
    #            accepted_refls.extend(refls)

    #    if len(accepted_refls) == 0:
    #        raise RuntimeError("No reflections left after applying significance filter")
    #    experiments = accepted_expts
    #    integrated = accepted_refls
#
#    # Delete the shoeboxes used for intermediate calculations, if requested
#    if params.integration.debug.delete_shoeboxes and "shoebox" in integrated:
#        del integrated["shoebox"]
#
#
    #rmsd_indexed, _ = calc_2D_rmsd_and_displacements(indexed)
    #log_str = "RMSD indexed (px): %f\n" % rmsd_indexed
    #for i in range(6):
    #    bright_integrated = integrated.select(
    #        (
    #                integrated["intensity.sum.value"]
    #                / flex.sqrt(integrated["intensity.sum.variance"])
    #        )
    #        >= i
    #    )
    #    if len(bright_integrated) > 0:
    #        rmsd_integrated, _ = calc_2D_rmsd_and_displacements(bright_integrated)
    #    else:
    #        rmsd_integrated = 0
    #    log_str += (
    #            "N reflections integrated at I/sigI >= %d: % 4d, RMSD (px): %f\n"
    #            % (i, len(bright_integrated), rmsd_integrated)
    #    )

    #for crystal_model in experiments.crystals():
    #    if hasattr(crystal_model, "get_domain_size_ang"):
    #        log_str += ". Final ML model: domain size angstroms: {:f}, half mosaicity degrees: {:f}".format(
    #            crystal_model.get_domain_size_ang(),
    #            crystal_model.get_half_mosaicity_deg(),
    #        )

    #print(log_str)
    np.savetxt('sigma_b.txt', sigma_bs)
    np.savetxt('sigma_m.txt', sigma_ms)
    return experiments, integrated

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
    experiments = ProfileModelFactory.create(params, experiments, indexed)

    new_experiments = ExperimentList()
    new_reflections = flex.reflection_table()
    for expt_id, expt in enumerate(experiments):
        if(expt.profile.sigma_b() < params.profile.gaussian_rs.parameters.sigma_b_cutoff):
            refls = indexed.select(indexed["id"] == expt_id+1) # 1-indexed
            if(len(refls) == 0):
                continue
            boxes = expt.profile.compute_bbox(refls, expt.crystal, expt.beam, expt.detector) # Generate bounding boxes
            refls['bbox'] = boxes
            new_reflections.extend(refls)
            new_experiments.append(expt)

    experiments = new_experiments
    indexed = new_reflections
    if len(experiments) == 0:
        raise RuntimeError("No experiments after filtering by sigma_b")

    predicted.match_with_reference(indexed)
    integrator = create_integrator(params, experiments, predicted)
    integrated = integrator.integrate()

    if params.significance_filter.enable:

        sig_filter = SignificanceFilter(params)
        filtered_refls = sig_filter(experiments, integrated)
        accepted_expts = ExperimentList()
        accepted_refls = flex.reflection_table()
        for expt_id, expt in enumerate(experiments):
            refls = filtered_refls.select(filtered_refls["id"] == expt_id)
            if len(refls) > 0:
                accepted_expts.append(expt)
                refls["id"] = flex.int(len(refls), len(accepted_expts) - 1)
                accepted_refls.extend(refls)

        if len(accepted_refls) == 0:
            raise RuntimeError("No reflections left after applying significance filter")
        experiments = accepted_expts
        integrated = accepted_refls

    # Delete the shoeboxes used for intermediate calculations, if requested
    if params.integration.debug.delete_shoeboxes and "shoebox" in integrated:
        del integrated["shoebox"]


    rmsd_indexed, _ = calc_2D_rmsd_and_displacements(indexed)
    log_str = "RMSD indexed (px): %f\n" % rmsd_indexed
    for i in range(6):
        bright_integrated = integrated.select(
            (
                    integrated["intensity.sum.value"]
                    / flex.sqrt(integrated["intensity.sum.variance"])
            )
            >= i
        )
        if len(bright_integrated) > 0:
            rmsd_integrated, _ = calc_2D_rmsd_and_displacements(bright_integrated)
        else:
            rmsd_integrated = 0
        log_str += (
                "N reflections integrated at I/sigI >= %d: % 4d, RMSD (px): %f\n"
                % (i, len(bright_integrated), rmsd_integrated)
        )

    for crystal_model in experiments.crystals():
        if hasattr(crystal_model, "get_domain_size_ang"):
            log_str += ". Final ML model: domain size angstroms: {:f}, half mosaicity degrees: {:f}".format(
                crystal_model.get_domain_size_ang(),
                crystal_model.get_half_mosaicity_deg(),
            )

    print(log_str)
    return experiments, integrated

"""
Split a sequence into a set of stills.

Example:

  dials.sequence_to_stills sequence.expt sequence.refl
"""


import logging
from IPython import embed

from dxtbx.model import MosaicCrystalSauter2014
from dxtbx.model.experiment_list import Experiment, ExperimentList
from libtbx.phil import parse
from scitbx import matrix

from dials.algorithms.refinement.prediction.managed_predictors import (
    ExperimentsPredictorFactory,
)
from dials.array_family import flex
from dials.model.data import Shoebox
from dials.util import show_mail_handle_errors
from dials.util.options import OptionParser, reflections_and_experiments_from_files

logger = logging.getLogger("dials.command_line.sequence_to_stills")

# The phil scope
phil_scope = parse(
    """
output {
  experiments = dials_temp_files/multi_stills.expt
    .type = str
    .help = "Filename for the experimental models that have been converted to stills"
  reflections = dials_temp_files/multi_stills.refl
    .type = str
    .help = "Filename for the reflection tables with split shoeboxes (3D to 2D)"
  domain_size_ang = None
    .type = float
    .help = "Override for domain size. If None, use the crystal's domain size, if"
            "available"
  half_mosaicity_deg = None
    .type = float
    .help = "Override for mosaic angle. If None, use the crystal's mosaic angle, if"
            "available"
}
max_scan_points = None
  .type = int
  .expert_level = 2
  .help = Limit number of scan points
"""
)


def sequence_to_stills(experiments, reflections, params):
    assert len(reflections) == 1
    reflections = reflections[0]

    new_experiments = ExperimentList()
    new_reflections = flex.reflection_table()

#    # This is the subset needed to integrate
#    for key in [
#        "id",
#        "imageset_id",
#        "shoebox",
#        "bbox",
#        "intensity.sum.value",
#        "intensity.sum.variance",
#        "entering",
#        "flags",
#        "miller_index",
#        "panel",
#        "xyzobs.px.value",
#        "xyzobs.px.variance",
#    ]:
    for key in reflections.keys():
        new_reflections[key] = type(reflections[key])()
    reflections["imageset_id"] = flex.int(len(reflections), 0)
    new_reflections["imageset_id"] = flex.int()
#        elif key == "entering":
#            reflections["entering"] = flex.bool(len(reflections), False)
#            new_reflections["entering"] = flex.bool()
#        else:
#            raise RuntimeError(f"Expected key not found in reflection table: {key}")

    first_loop = True
    crystals = []
    imagesets = []

    for expt_id, experiment in enumerate(experiments):
        # Generate crystals and imagesets for each scan point on first loop
        if(first_loop):
            for i_scan_point in range(*experiment.scan.get_array_range()):
                # Get the goniometer setting matrix
                goniometer_setting_matrix = matrix.sqr(
                    experiment.goniometer.get_setting_rotation()
                )
                goniometer_axis = matrix.col(experiment.goniometer.get_rotation_axis())
                step = experiment.scan.get_oscillation()[1]
    
                # The A matrix is the goniometer setting matrix for this scan point
                # times the scan varying A matrix at this scan point. Note, the
                # goniometer setting matrix for scan point zero will be the identity
                # matrix and represents the beginning of the oscillation.
                # For stills, the A matrix needs to be positioned in the midpoint of an
                # oscillation step. Hence, here the goniometer setting matrixis rotated
                # by a further half oscillation step.
                A = (
                    goniometer_axis.axis_and_angle_as_r3_rotation_matrix(
                        angle=experiment.scan.get_angle_from_array_index(i_scan_point)
                        + (step / 2),
                        deg=True,
                    )
                    * goniometer_setting_matrix
                    * matrix.sqr(experiment.crystal.get_A_at_scan_point(i_scan_point))
                )
                crystal = MosaicCrystalSauter2014(experiment.crystal)
                crystal.set_A(A)
    
                # Copy in mosaic parameters if available
                if params.output.domain_size_ang is None and hasattr(
                    experiment.crystal, "get_domain_size_ang"
                ):
                    crystal.set_domain_size_ang(experiment.crystal.get_domain_size_ang())
                elif params.output.domain_size_ang is not None:
                    crystal.set_domain_size_ang(params.output.domain_size_ang)
    
                if params.output.half_mosaicity_deg is None and hasattr(
                    experiment.crystal, "get_half_mosaicity_deg"
                ):
                    crystal.set_half_mosaicity_deg(
                        experiment.crystal.get_half_mosaicity_deg()
                    )
                elif params.output.half_mosaicity_deg is not None:
                    crystal.set_half_mosaicity_deg(params.output.half_mosaicity_deg)
    
                # Add to list of crystals
                crystals.append(crystal)
    
                # Add to list of imagesets
                imagesets.append(experiment.imageset.as_imageset()[i_scan_point : i_scan_point + 1])
    
                # Don't regenerate this list
                first_loop = False

        # Split experiment by scan points
        for i_scan_point in range(*experiment.scan.get_array_range()):
            new_experiment = Experiment(
                identifier = str(len(crystals)*expt_id + i_scan_point),
                detector=experiment.detector,
                beam=experiment.beam,
                crystal=crystals[i_scan_point],
                imageset=imagesets[i_scan_point],
            )
            new_experiments.append(new_experiment)

# ----------------EXPERIMENTS CREATED---------------------------------

    for i_scan_point in range(len(crystals)):
        # Get subset of reflections on this image
        _, _, _, _, z1, z2 = reflections["bbox"].parts()
        subrefls = reflections.select((i_scan_point >= z1) & (i_scan_point < z2))
        new_refls = subrefls.copy()
        new_refls['xyzobs.px.value'] = subrefls['xyzobs.px.value'] - [0.,0.,0.5]
        new_refls['imageset_id'] = flex.int(new_refls['xyzobs.px.value'].parts()[2].as_numpy_array())
        x, y, _ = subrefls['xyzobs.mm.value'].parts()
        new_refls['xyzobs.mm.value'] = flex.vec3_double(x, y, flex.double(len(new_refls), 0))
        new_refls['id'] = flex.int(subrefls['id']*len(crystals) + i_scan_point)
        new_reflections.extend(new_refls)

#        for refl in subrefls.rows():
#            new_refl = {}
#            for key in refl.keys():
#                new_refl[key] = refl[key]
#            new_refl["xyzobs.px.value"] = flex.double([refl['xyzobs.px.value'][0], refl['xyzobs.px.value'][1], refl['xyzobs.px.value'][2] - 0.5])
#            new_refl["imageset_id"] = int(refl['xyzobs.px.value'][2])
#            new_refl["id"] = 0
#            new_reflections.append({}) # Need to append a reflection table, not reflection
#            embed()
#            for key in new_refl:
#                new_reflections[key][-1] = new_refl[key] # Roundabout way to append reflection

# ----------------REFLECTIONS CREATED-------------------------------

# img_id = int(xyz_obs.px.value()[2] - 0.5)

    return (new_experiments, new_reflections)


@show_mail_handle_errors()
def run(args=None, phil=phil_scope):
    """
    Validate the arguments and load experiments/reflections for sequence_to_stills

    Arguments:
        args: The command line arguments to use. Defaults to sys.argv[1:]
        phil: The phil_scope. Defaults to the master phil_scope for this program
    """
    # The script usage
    usage = "usage: dials.sequence_to_stills [options] [param.phil] models.expt reflections.refl"

    # Create the parser
    parser = OptionParser(
        usage=usage,
        phil=phil,
        read_experiments=True,
        read_reflections=True,
        check_format=False,
        epilog=__doc__,
    )
    params, options = parser.parse_args(args=args, show_diff_phil=True)

    # Try to load the models and data
    if not params.input.experiments or not params.input.reflections:
        parser.print_help()
        return

    reflections, experiments = reflections_and_experiments_from_files(
        params.input.reflections, params.input.experiments
    )

    (new_experiments, new_reflections) = sequence_to_stills(
        experiments, reflections, params
    )
    # Write out the output experiments, reflections
    new_experiments.as_file(params.output.experiments)
    new_reflections.as_file(params.output.reflections)


if __name__ == "__main__":
    run()

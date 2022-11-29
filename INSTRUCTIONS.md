### Prerequisites:
 - A dials installation for which you have write permissions. This will likely be something that you have installed yourself
 - Diffraction images in a known (ideally scratch) location
 - Spacegroup and unit cell (estimated unit cell is fine)
 - (Optional) A custom pixel mask to remove undesirable detector areas. This can be created via the dials image viewer.

## Instructions (NOT including careless or phenix):
 - Clone this repo into a location where you can read and write, ideally scratch
 - Edit lines 3-15 of `pipeline_with_variables.sh` as appropriate for your system
    - `DIALS_PATH`: Absolute path to your dials_env.sh
    - `INSTALL_RS`: 0 if reciprocalspaceship is already installed in your cctbx.python installation, 1 if you need to install
    - `DIFF_IMG_DIR`: Absolute path to directory containing your images
    - `FULL_SYMMETRY_SPACE_GROUP`: full-symmetry space group
    - `CELL`: Unit cell, in format a la '"87.897,94.559,96.265,90.000,90.000,90.000"' (I do not understand this quoting scheme, just do it)
    - `PIXEL_MASK`: Path and filename of your custom pixel mask, or 0 if you do not have a mask
    - `DET_DIST`: Detector distance, in millimeters
    - `USE_SLURM_FOR_INTEGRATION`: If 1, integration jobs will be launched via SBATCH. If 0, all tasks will be run from the script itself. I recommend mode 1.
 - Create the `logs/` directory if it doesn't already exist. Log files will all go there, to keep them out of your hair
 - Edit the `pipeline_launch.sh` script:
    - Duplicate the file and give it an informative name and/or comments regarding its contents, if desired
    - Adjust the SBATCH header parameters as appropriate
    - In the body of the script, call (as many times as you would like)
        ```
        bash pipeline_with_variables.sh IMG_PREFIX OUT_DIR OSCILLATION [SPACEGROUP]
        ```
        where:
        - IMG_PREFIX: Prefix for the file names you want to process, e.g. `e011e_off_` or `e011f_200ns_`
        - OUT_DIR: Name of the directory which will be created inside laue-index-assigner to hold files from this pipeline. Name this however you would like
        - OSCILLATION: Rotation between frames, in degrees
        - SPACEGROUP: Optional. If omitted, defaults to the full-symmetry space group
 - Launch your jobs by running the `pipeline_launch.sh` script (presumably via SBATCH) 

### Using `precognition` for integration
tbd

### Instructions for careless:
After running the above pipeline, you should have two sets of integrated reflections, integrated_from_integrate.mtz and integrated_from_int_test.mtz. An understanding of the difference between these (and which is better) is rapidly evolving

Note that unlike the above scripts, which must be run through your cctbx.python installation, careless can be run via a conda environment containing careless.

The script `run_careless_p1_200ns_integrate_repeats.sh` should be somewhat useful as a template.

---
Below here are Rick's previous instructions
Hi, here are the instructions for the current pipeline:

First run config.sh to set environment variables, and run setpaths.sh for your DIALS installation.

If reciprocalspaceship is not installed in your cctbx.python environment, please install using
    cctbx.python -m pip install reciprocalspaceship

PIPELINE ORDER (this is encoded in pipeline.sh):

scan_varying_indexer.sh
sequence_to_stills_no_sb.py
dials_assign.py
assign_beams_to_stills.py
refine_initial.sh
refine_remove_outliers.sh
poly_predictor.py

Please run all Python files with cctbx.python. Integration is under development.

A quick rundown of the temporary DIALS files by name and in order of generation by above scripts:

strong_1.04.refl  <-- Strong spots at lambda=1.04 angstroms
expected_index.* <-- Output of initial monochromatic dials.index at 1.04 angstroms
refined_varying.* <-- Output of scan-varying monochromatic refinement
stills_no_sb.* <-- Output of script that generates stills without using shoeboxes 
optimized.* <-- Output of rotation optimization code for Miller indices
multi.* <-- Output that generates experiment beams for stills - one beam per reflection
ultra_refined.* <-- Output that freely refined wavelengths for reflections while constraining all other parameters
mega_ultra_refined.* <-- Output that serves as the final geometric refinement of the experiment
predicted.refl <-- Predictions generated using the refined experimental model

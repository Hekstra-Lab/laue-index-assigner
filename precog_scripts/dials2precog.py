"""
Python script to generate input files for Precognition integration after running DIALS
through geometry refinement. ALL SPACES HARDCODED DON'T TOUCH THEM. Hekstra lab use only --
other uses will vary hardcoded items.
"""

print('Importing libraries')
from tqdm import tqdm
import numpy as np
import gemmi
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
from IPython import embed

expt_file = "dials_temp_files/ultra_refined.expt"
refl_file = "dials_temp_files/ultra_refined.refl"

# Read data in
print('Reading DIALS experiments')
elist = ExperimentListFactory.from_json_file(expt_file)
print('Reading DIALS reflections')
refls = reflection_table.from_file(refl_file)
print('Computing reflection resolutions')
resolutions = refls.compute_d(elist).as_numpy_array()

# Loop over images
print('Generating Precognition files')
for i in tqdm(np.unique(refls['imageset_id'].as_numpy_array())):
    j = i+1 # 1-indexed count

    # Get objects for image
    expt = elist[i]
    det = expt.detector
    cryst = elist.crystals()[i]

    # Initialize output file
    img_name = f'e080_{j:03}.mccd'
    filename = f'test/{img_name}.inp'
    output = open(filename, 'w')
    
    # Get crystal line
    ucell = cryst.get_unit_cell().parameters()
    spacegroup = gemmi.SpaceGroup(cryst.get_space_group().type().universal_hermann_mauguin_symbol())
    crystal_line = f'   Crystal   {ucell[0]} {ucell[1]} {ucell[2]} {ucell[3]} {ucell[4]} {ucell[5]} {spacegroup.number}\n'

    # Get matrix line
    a = np.asarray(cryst.get_A()).reshape(3,3) 
    b = np.asarray(cryst.get_B()).reshape(3,3)
    dials2precog = np.asarray([1,0,0,0,-1,0,0,0,-1]).reshape(3,3)
    mat = (dials2precog@a@np.linalg.inv(b)).flatten()
    matrix_line=f'   Matrix    {mat[0]} {mat[1]} {mat[2]} {mat[3]} {mat[4]} {mat[5]} {mat[6]} {mat[7]} {mat[8]}\n'

    # Get omega line
    omega_line=f'   Omega     0.000 0.000\n'

    # Get goniometer line
    goniometer_line=f'   Goniometer 0.000 0.000 0.000\n\n' # Matrix already transformed for stills...not sure if this will be valid

    # Get format line
    format_line=f'   Format     RayonixMX340\n' # Hardcoded name randomly guessed by dev

    # Get distance line
    dist = np.abs(det.to_dict()['panels'][0]['origin'][2])
    distance_line=f'   Distance   {dist} 0.050\n' # Hardcoded uncertainty randomly guessed by dev

    # Get center line
    center = det.get_ray_intersection(expt.beam.get_s0())[1] # in mm
    center = np.asarray([center[0]/det.to_dict()['panels'][0]['pixel_size'][0], center[1]/det.to_dict()['panels'][0]['pixel_size'][1]]) # in px
    center_line=f'   Center     {center[0]} {center[1]} 0.20 0.20\n' # Hardcoded uncertainty guessed by dev

    # Get pixel line
    pix_size = det.to_dict()['panels'][0]['pixel_size']
    pixel_line=f'   Pixel      {pix_size[0]} {pix_size[1]} 0.000001 0.000001\n' # Harcoded uncertainty guessed by dev

    # Get swing line
    swing_line=f'   Swing      0.0 0.0 0.0 0.0\n' # Hardcoded guess by dev

    # Get tilt line
    # TODO: Fix for nonzero z detector axes later
    tilt = [0,0]
    tilt_line=f'   Tilt       {tilt[0]} {tilt[1]} 0.1 0.1\n' # Hardcoded uncertainty guessed by dev

    # Get bulge line
    bulge_line=f'   Bulge      0.0 0.0 0.0 0.0\n\n' # Hardcoded guess by dev

    # Get image line
    image_line=f'   Image      {img_name}\n'

    # Get resolution line
    res_img = resolutions[refls['imageset_id'].as_numpy_array() == i]
    resolution_line=f'   Resolution {np.min(res_img)} {np.max(res_img)}\n'

    # Get wavelength line
    lams = refls['Wavelength'].as_numpy_array()
    wavelength_line=f'   Wavelength {np.min(lams)} {np.max(lams)}\n'

    # Write data and close
    output.write('Input\n')
    output.write(crystal_line)
    output.write(matrix_line)
    output.write(omega_line)
    output.write(goniometer_line)
    output.write(format_line)
    output.write(distance_line)
    output.write(center_line)
    output.write(pixel_line)
    output.write(swing_line)
    output.write(tilt_line)
    output.write(bulge_line)
    output.write(image_line)
    output.write(resolution_line)
    output.write(wavelength_line)
    output.write('Quit')
    output.close()

print('Writing .lam file')
lam_lims = np.linspace(np.min(lams), np.max(lams), 11)
filename = f'test/lambda.lam'
output = open(filename, 'w')
for i in range(len(lam_lims)-1):
    lam_mid = (lam_lims[i] + lam_lims[i+1])/2
    lam_count = sum((lams > lam_lims[i]) & ( lams < lam_lims[i+1]))
    ratio = lam_count / len(lams)
    output.write(f'{lam_mid} {ratio}\n')
output.close()

print('Estimating spot size')
spot = [6, 6, 3]

print('Writing integration script')
filename = f'test/precog_integrate.inp'
output = open(filename, 'w')
output.write(f'diagnostic off\n')
output.write(f'busy off\n')
output.write(f'warning off\n')
for i in np.unique(refls['imageset_id'].as_numpy_array()):
    j = i+1
    output.write(f'@ e080_{j:03}.mccd.inp\n')
output.write(f'Input\n')
output.write(f' Image lambda.lam\n')
output.write(f' Spot {spot[0]} {spot[1]} {spot[2]}\n')
output.write(f' Quit\n')
output.write(f'Dataset linearAnalytical\n')
output.write(f' Out integrated\n')
output.write(f' Resolution {np.min(res_img)} {np.max(res_img)}\n')
output.write(f' Wavelength {np.min(lams)} {np.max(lams)}\n')
output.write(f' Quit\n')
output.write(f'Quit\n')
output.close()

print('Finished!')

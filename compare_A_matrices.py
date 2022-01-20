# This is a script to compare A matrices between DIALS and Precognition files

# NOTE: Python 3.5 or higher required. Matrix multiplication operator not defined in previous versions.

# NOTE: Build some functions to run this across entire datasets and histogram it. Throw it up on Github.

# In reciprocal space if not noted otherwise


from cctbx import crystal_orientation
from scitbx.matrix import sqr
from dxtbx.model import experiment_list
from dxtbx.model import crystal
import numpy as np
from gemmi import UnitCell,Fractional
import math
from IPython import embed
import glob
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

# Kevin's function for finding rotation angle between two A matrices

def rot_angle(crystal, other):
    """
    Calculate the angular rotation between two crystal objects.
    
    Parameters
    ----------
    crystal : MosaicCrystalSauter2014, or similar object with a get_A method that returns a tuple with length 9
    other   : MosaicCrystalSauter2014, or similar object with a get_A method that returns a tuple with length 9
    
    Returns
    -------
    angle : float
        The rotation angle between the two A matrices
    axis : float scitbx.matrix
        Axis of rotation between the two A-matrices. Exists as a 3x1 matrix.
    """
    ref_orientation = crystal_orientation.crystal_orientation(crystal.get_A(), crystal_orientation.basis_type.reciprocal)
    other_orientation = crystal_orientation.crystal_orientation(other.get_A(), crystal_orientation.basis_type.reciprocal)
    #aligned_ori = crystal_orientation.crystal_orientation(other.get_A(), crystal_orientation.basis_type.reciprocal)
    op_align = sqr(other_orientation.best_similarity_transformation(ref_orientation, math.inf, 1))
    aligned_ori = other_orientation.change_basis(op_align)
    
    U_ref   = ref_orientation.get_U_as_sqr()
    U_other = aligned_ori.get_U_as_sqr()
    missetting_rot = U_other* U_ref.inverse()
    angle,axis = missetting_rot.r3_rotation_matrix_as_unit_quaternion().unit_quaternion_as_axis_and_angle(deg=True)
    return angle,axis

def cell_params(crystal, other):
    """
    Calculate the magnitude of the difference in unit cell 
    parameters between two crystal objects.

    Parameters
    ----------

    Returns
    -------
    a : float

    b : float
    
    c : float

    alpha : float
 
    beta : float

    gamma : float

    """

    cell1 = np.asarray(crystal.get_unit_cell().parameters(), dtype=float)
    cell2 = np.asarray(other.get_unit_cell().parameters(), dtype=float)
    cell_diff = abs(cell1 - cell2)
    da,db,dc,dalpha,dbeta,dgamma = cell_diff
    return da,db,dc,dalpha,dbeta,dgamma

# Filenames for DIALS and Precognition Inputs
# TODO: Do a sanity check and make sure these actually line up
dials_filename_expt = "dials_temp_files/stills_no_sb.expt"
dials_filename_refl = "dials_temp_files/stills_no_sb.refl"
precog_filenames = sorted(glob.glob("data/e080_*.mccd.inp")) # "e080_001.mccd.inp"

# Initialize  arrays
angle_diffs = np.zeros(len(precog_filenames))
angle_axes = np.ndarray(shape=(len(precog_filenames),3), dtype=float)
da = np.zeros(len(precog_filenames))
db = np.zeros(len(precog_filenames))
dc = np.zeros(len(precog_filenames))
dalpha = np.zeros(len(precog_filenames))
dbeta = np.zeros(len(precog_filenames))
dgamma = np.zeros(len(precog_filenames))

# DIALS part
# Load experiment list from JSON	
# This should ONLY be ran on stills -- convert from sequence to stills
dials_experiment_model = experiment_list.ExperimentListFactory.from_json_file(dials_filename_expt)

def get_rotation_matrix(axis, angle):
    u = axis
    sin,cos = np.sin(angle),np.cos(angle)
    return cos*np.eye(3) + sin*np.cross(u, -np.eye(3)) + (1. - cos)*np.outer(u, u)

for i in np.arange(len(precog_filenames)):
    # Units for the following Precognition Parameters:
    # 
    # Unit Cell: Angstroms and Degrees
    # Space Group: Integer denoting space group
    # Missetting Matrix: ???
    # Omega Polar Orientation: Degrees
    # Goniometer (omega, chi, phi): Degrees
    # Format: Detector Name
    # Detector Distance/Uncertainty: mm/??
    # Beam Center/Uncertainty: pixel/pixel 
    # Pixel Size/Uncertainty: mm/mm
    # Swing Angle/Uncertainty: Degrees/??
    # Tilt Angle/Uncertainty: Degrees/??
    # Bulge Correction/Uncertainty: ??/??
    # Resolution Range: Angstroms
    # Wavelength Range: Angstroms
    
    # Precognition part
    # Parse Precognition file for unit cell params
    # NOTE: These conventions using Precognition conventions as listed in User Guide Release 5.0
    # Please check for correspondence with conventions used in DIALS
    # Do we need detector format? e.g. "RayonixMX340" ??
    # What is a bulge correction?
    # What is a swing angle?
    
    for line in open(precog_filenames[i]):
        rec = line.strip()
        if rec.startswith('Crystal'):
            unit_cell = rec.split()[1:7] # a,b,c,alpha,beta,gamma
            space_group = rec.split()[7]
        if rec.startswith('Matrix'):
            missetting_matrix = rec.split()[1:10]
        if rec.startswith('Omega'):
            omega_polar_orientation = rec.split()[1:3]
        if rec.startswith('Goniometer'):
            goniometer = rec.split()[1:4]
        if rec.startswith('Format'):
            detector_format = rec.split()[1]
        if rec.startswith('Distance'): 
            detector_distance = rec.split()[1]
            detector_distance_uncertainty = rec.split()[2]
        if rec.startswith('Center'):
            beam_center = rec.split()[1:3]
            beam_center_uncertainty = rec.split()[3:5]
        if rec.startswith('Pixel'):
            pixel_size = rec.split()[1:3]
            pixel_size_uncertainty = rec.split()[3:5]
        if rec.startswith('Swing'):
            swing_angles = rec.split()[1:3]
            swing_angles_uncertainty = rec.split()[3:5] # I'm guessing? User guide isn't explicit...
        if rec.startswith('Tilt'):
            detector_tilt_angles = rec.split()[1:3]
            detector_tilt_angles_uncertainty = rec.split()[3:5] # Also guessing this is uncertainty
        if rec.startswith('Bulge'):
            detector_bulge_correction = rec.split()[1:3]
            detector_bulge_correction_uncertainty = rec.split()[3:5] # More guessing...
    #    if rec.startswith('Image'): rec.split()[1] # I think this only means something to Precognition's indexer
        if rec.startswith('Resolution'):
            resolution_range = rec.split()[1:3]
        if rec.startswith('Wavelength'):
            wavelength_range = rec.split()[1:3]
    
    # Convert angles to radians and prepare matrices for calculation
    # o1 \in [0,pi)
    # o2 \in [0,2pi)
    M = np.array(missetting_matrix, dtype=float).reshape((3,3))
    o1 = np.deg2rad(float(omega_polar_orientation[0]))
    o2 = np.deg2rad(float(omega_polar_orientation[1]))
    gonio_phi = np.deg2rad(float(goniometer[2]))
    cell = UnitCell(*[float(i) for i in unit_cell]) # i is an iterating variable
    
    # Get rotation matrix from B to U
    R = get_rotation_matrix(np.array([0., 0., -1.]),  o1)
    temp  = get_rotation_matrix(np.array([0., 1., 0.]), o2)
    R = temp@R
    temp = get_rotation_matrix((R@np.array([0., 1., 0.])[:,None])[:,0], gonio_phi)
    R = temp@R
    
    # Create transposed orthogonalization matrix
    O = np.vstack((
        cell.orthogonalize(Fractional(1., 0., 0.)).tolist(),
        cell.orthogonalize(Fractional(0., 1., 0.)).tolist(),
        cell.orthogonalize(Fractional(0., 0., 1.)).tolist(),
    )).T
    
    # Compute U, B, A matrices
    precog2mosflm = np.array([ 
                    [  0,  0,  1],
                    [  0, -1,  0],
                    [  1,  0,  0]]) # Change from Precognition to MOSFLM convention (this is a left operator)
    precog_A = precog2mosflm@(R@M@np.linalg.inv(O))
    
    # precog_A = precog_U@precog_B # This is a lie
    # U is a properly oriented real-space crystallographic basis for frame in lab coordinate system
    # So is B
    
    # Create crystal object
    precog_crystal_model = crystal.CrystalFactory.from_mosflm_matrix(precog_A.flatten())
    
    
    # Compare A matrices and print information
    j = int(i)
    dials_crystal_model = dials_experiment_model[j].crystal
    angle_diffs[j], angle_axes[j] = rot_angle(dials_crystal_model, precog_crystal_model) 
    da[j],db[j],dc[j],dalpha[j],dbeta[j],dgamma[j] = cell_params(dials_crystal_model, precog_crystal_model)


#-----------------------------------------------------------------------------
# Plot a histogram of the angle differences
n_bins = 30

frames = np.arange(len(precog_filenames))
ax = plt.figure().gca()
plt.plot(frames,angle_diffs)
plt.grid(axis='y', alpha=1)
plt.xlabel("Frames")
plt.ylabel("Angular differences (degrees)")
plt.title("DIALS vs Precognition Crystal Model Angular Offsets by Frame")
#ax.yaxis.set_major_locator(MaxNLocator(integer=True)) # Crystals come in integers units 
plt.savefig("diff_angles_per_frame.png")

ax = plt.figure().gca()
plt.plot(frames,angle_axes[:,0])
plt.grid(axis='y', alpha=1)
plt.xlabel("Frames")
plt.ylabel("Rotation Axis x")
plt.title("DIALS vs Precognition Crystal Model Rotation Axis X-Components by Frame")
plt.savefig("axis_x_per_frame.png")

ax = plt.figure().gca()
plt.plot(frames,angle_axes[:,1])
plt.grid(axis='y', alpha=1)
plt.xlabel("Frames")
plt.ylabel("Rotation Axis y")
plt.title("DIALS vs Precognition Crystal Model Rotation Axis Y-Components by Frame")
plt.savefig("axis_y_per_frame.png")

ax = plt.figure().gca()
plt.plot(frames,angle_axes[:,2])
plt.grid(axis='y', alpha=1)
plt.xlabel("Frames")
plt.ylabel("Rotation Axis z")
plt.title("DIALS vs Precognition Crystal Model Rotation Axis Z-Components by Frame")
plt.savefig("axis_z_per_frame.png")


nrows = 2
ncols = 3
fig, axs = plt.subplots(2,3, sharex=True)
fig.suptitle('Cell Parameters by Frame')
fig.set_size_inches(8,5)
axs[0,0].plot(frames, da)
axs[0,0].set_title(r'$\Delta a$')
axs[0,1].plot(frames, db)
axs[0,1].set_title(r'$\Delta b$')
axs[0,2].plot(frames, dc)
axs[0,2].set_title(r'$\Delta c$')
axs[1,0].plot(frames, dalpha)
axs[1,0].set_title(r'$\Delta \alpha$')
axs[1,1].plot(frames, dbeta)
axs[1,1].set_title(r'$\Delta \beta$')
axs[1,2].plot(frames, dgamma)
axs[1,2].set_title(r'$\Delta \gamma$')
for ax in axs.flat:
    ax.set(xlabel="Frames")
for ax in axs[0,]:
    ax.set(ylabel="Angstroms")
for ax in axs[1,]:
    ax.set(ylabel="Degrees")
for ax in axs.flat:
    ax.label_outer()
plt.savefig("cell_params_per_frame.png")

from diffgeolib import *
from tqdm import trange,tqdm
from matplotlib import pyplot as plt
from scitbx import matrix
from IPython import embed
from glob import glob
from utils.precognition import parse_ii_inp_file_pairs
import numpy as np
import reciprocalspaceship as rs
import gemmi
import pandas as pd

#expt_file = "/home/rahewitt/laue_indexer/laue-index-assigner/dials_temp_files/updated.expt"
expt_file = "/home/rahewitt/laue_indexer/laue-index-assigner/dials_temp_files/refined_varying.expt"
#expt_file = "/home/rahewitt/Downloads/peak_refined.expt"
refl_file = "optimized.refl"

#ds = rs.read_precognition(ii_file).reset_index()
print('parsing precog files')
precog_df = parse_ii_inp_file_pairs(
    sorted(glob('data/e080_???.mccd.ii')),
    sorted(glob('data/e080_???.mccd.inp')),
    ).reset_index()

print('reading DIALS files')
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
elist = ExperimentListFactory.from_json_file(expt_file)
cryst = elist.crystals()[0]
unit_cell = cryst.get_unit_cell()
precog_df.spacegroup = gemmi.SpaceGroup(cryst.get_space_group().type().universal_hermann_mauguin_symbol())
precog_df.cell = gemmi.UnitCell(*unit_cell.parameters())
refls = reflection_table.from_file(refl_file)

print('generating DIALS dataframe')
dials_df = pd.DataFrame(index=np.zeros(len(refls)))
dials_df['X'] = refls['xyzobs.px.value'].parts()[0].as_numpy_array()
dials_df['Y'] = refls['xyzobs.px.value'].parts()[1].as_numpy_array()
dials_df['H'] = refls['miller_index'].parts()[0].as_numpy_array()
dials_df['K'] = refls['miller_index'].parts()[1].as_numpy_array()
dials_df['L'] = refls['miller_index'].parts()[2].as_numpy_array()
dials_df['Wavelength'] = refls['Wavelength'].as_numpy_array()
#dials_df['BATCH'] = refls['shoebox'].bounding_boxes().parts()[4]
dials_df['BATCH'] = refls['xyzobs.px.value'].parts()[2].as_numpy_array() - 0.5

print('getting HKL matrices')
dials_hkl = dials_df[['H','K','L']].to_numpy().astype(int).T
precog_hkl = precog_df[['H','K','L']].to_numpy().astype(int).T

print('initializing metrics')
percent_correct = np.zeros(elist[0].imageset.size())

# Quiver plot of HKLs normalized to visualize density
num_refls = len(dials_hkl[0,:])
sphere_radius = 1
abc = np.asarray(elist[0].crystal.get_unit_cell().parameters()[0:3]) # Not generalized for all space groups
abc_norm = abc/max(abc)
uvw = dials_hkl.astype(float)
uvw[0,:] = uvw[0,:]*abc_norm[0] # Transform from H to Cartesian U
uvw[1,:] = uvw[1,:]*abc_norm[1] # Transform from K to Cartesian V
uvw[2,:] = uvw[2,:]*abc_norm[2] # Transform from L to Cartesian W
uvw = sphere_radius*uvw/np.linalg.norm(uvw, axis=0) # Normalize uvw
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Normalized HKL vectors')
ax.set_xlim([-sphere_radius,sphere_radius])
ax.set_ylim([-sphere_radius,sphere_radius])
ax.set_zlim([-sphere_radius,sphere_radius])
ax.set_xlabel('normalized H')
ax.set_ylabel('normalized K')
ax.set_zlabel('normalized L')
ax.scatter(uvw[0],uvw[1],uvw[2], marker='.')
plt.show()

# Iterate by frame and match HKLs, seeing what percentage are correct
for i in tqdm(np.arange(elist[0].imageset.size())):
    # Get reflection indices from each batch
    precog_idx = (precog_df['BATCH'] == i).astype(bool)
    dials_idx = (dials_df['BATCH'] == i).astype(bool)

    # Get XY positions for refls
    dials_xy = dials_df[['X', 'Y']].to_numpy().astype(float)[dials_idx]
    precog_xy = precog_df[['X', 'Y']].to_numpy().astype(float)[precog_idx]
    precog_xy_filtered = np.zeros(dials_xy.shape)
    # Filter out extraneous precog refls
    precog_filter = np.full(len(precog_xy), np.inf)
    for j in np.arange(len(dials_xy)):                                                  
        precog_filter[j] = np.argmin(np.linalg.norm(dials_xy[j,:] - precog_xy, axis=-1))
    precog_img_hkl = np.zeros((3,len(dials_hkl[:, dials_idx].T)))
    for j in np.arange(len(dials_xy[:, 0])):                                              
        precog_img_hkl[:,j] = precog_hkl.T[precog_idx, :][precog_filter[j].astype(int), :]
        precog_xy_filtered[j,:] = precog_xy[precog_filter[j].astype(int)]

    # Align precog to DIALS hkls
    aligned_hkls = align_hkls(dials_hkl[:, dials_idx].T, precog_img_hkl.T, precog_df.spacegroup)

    if i in [0, 100, 178]:
        x = dials_xy[:,0]
        y = dials_xy[:,1]
#        bot = np.min([aligned_hkls[:,0].min(), aligned_hkls[:,1].min(), aligned_hkls[:,2].min()])
#        top = np.max([aligned_hkls[:,0].max(), aligned_hkls[:,1].max(), aligned_hkls[:,2].max()])
        fig, axs = plt.subplots(2,3) 
        fig.suptitle(f'DIALS vs Precog HKLS for Frame {i}')
        bot = -30
        top = 30
        norm = plt.Normalize(bot, top)
        cmap = plt.get_cmap('viridis')
        sm = plt.cm.ScalarMappable(norm,cmap)
        fig.colorbar(sm, ax=axs.ravel().tolist()).set_label('Miller Index Scale')

        z = dials_hkl[0, dials_idx]
        axs[0,0].scatter(x, y, c=z, cmap=cmap, norm=norm, alpha=0.3)
        z = dials_hkl[1, dials_idx]
        axs[0,1].scatter(x, y, c=z, cmap=cmap, norm=norm, alpha=0.3)
        z = dials_hkl[2, dials_idx]
        axs[0,2].scatter(x, y, c=z, cmap=cmap, norm=norm, alpha=0.3)
        z = aligned_hkls[:,0]
        axs[1,0].scatter(x, y, c=z, cmap=cmap, norm=norm, alpha=0.3)
        z = aligned_hkls[:,1]
        axs[1,1].scatter(x, y, c=z, cmap=cmap, norm=norm, alpha=0.3)
        z = aligned_hkls[:,2]
        axs[1,2].scatter(x, y, c=z, cmap=cmap, norm=norm, alpha=0.3)

        for m, row in enumerate(axs):
            for j, cell in enumerate(row):
                if m==0:
                    cell.set_ylabel('DIALS')
                else:
                    cell.set_ylabel('Precognition')
                if j==0:
                    cell.set_xlabel('H')
                if j==1:
                    cell.set_xlabel('K')
                if j==2:
                    cell.set_xlabel('L')
        for ax in axs.flat:
            ax.label_outer()
        plt.show()


    # Check correctness of matching
    correct = is_ray_equivalent(aligned_hkls, dials_hkl[:,dials_idx].T)
    print(sum(correct)/len(correct))
    percent_correct[i] = sum(correct)/len(correct)

    plt.hist2d(dials_xy[:,0], dials_xy[:,1], bins=100, density=True)
    plt.title('DIALS XY HIST2D')
    plt.show()
    plt.hist2d(precog_xy_filtered[:,0], precog_xy_filtered[:,1], bins=100, density=True)
    plt.title('PRECOG XY HIST2D')
    plt.show()
    x_diff = precog_xy_filtered[:,0] - dials_xy[:,0]
    y_diff = precog_xy_filtered[:,1] - dials_xy[:,1]
    plt.hist2d(x_diff, y_diff, bins=20, range=[[-2,2], [-2,2]])
    plt.title('PRECOG - DIALS XY HIST2D')
    plt.show()
    plt.hist2d(x_diff[~correct], y_diff[~correct], bins=20, range=[[-200,200], [-200,200]])
    plt.title('PRECOG - DIALS XY HIST2D INCORRECT ONLY')
    plt.show()

    plt.figure()
    plt.scatter(dials_xy[:,0][~correct], dials_xy[:,1][~correct], c='b', alpha=0.5)
    plt.scatter(precog_xy_filtered[:,0][~correct], precog_xy_filtered[:,1][~correct], c='r', alpha=0.5)
    for i in range(len(dials_xy[:,0][~correct])):
        plt.arrow(dials_xy[:,0][~correct][i], dials_xy[:,1][~correct][i], x_diff[~correct][i], y_diff[~correct][i])
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.title('DIALS vs Precog Incorrectly Indexed Spot Centroids')
    plt.legend(title='Blue:DIALS XY (pixels)\nRed:Precognition XY (pixels)\nArrows connect centroids of same spot.')
    plt.show()

    if (i == 2):
        pix_position = dials_xy
        plt.figure()
        plt.title("Millers After Optimization")
        plt.plot(pix_position[correct,0], pix_position[correct,1], 'k.', label='Correct')
        plt.plot(pix_position[~correct,0], pix_position[~correct,1], 'r.', label='Incorrect')
        plt.legend()
        plt.show()

    # Quiver plot of HKLs normalized to visualize density
    num_refls = len(dials_hkl[0,:])
    sphere_radius = 1
    abc = np.asarray(elist[0].crystal.get_unit_cell().parameters()[0:3]) # Not generalized for all space groups
    abc_norm = abc/max(abc)
    uvw = dials_hkl[:,dials_idx].astype(float)
    uvw[0,:] = uvw[0,:]*abc_norm[0] # Transform from H to Cartesian U
    uvw[1,:] = uvw[1,:]*abc_norm[1] # Transform from K to Cartesian V
    uvw[2,:] = uvw[2,:]*abc_norm[2] # Transform from L to Cartesian W
    uvw = sphere_radius*uvw/np.linalg.norm(uvw, axis=0) # Normalize uvw
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Normalized HKL vectors')
    ax.set_xlim([-sphere_radius,sphere_radius])
    ax.set_ylim([-sphere_radius,sphere_radius])
    ax.set_zlim([-sphere_radius,sphere_radius])
    ax.set_xlabel('normalized H')
    ax.set_ylabel('normalized K')
    ax.set_zlabel('normalized L')
    ax.scatter(uvw[0],uvw[1],uvw[2], marker='.', c=~correct, cmap='bwr')
    plt.show()
    
plt.plot(np.arange(len(percent_correct)), percent_correct)
plt.xlabel('Image Number')
plt.ylabel('Fraction of Reflections Indexed Correctly')
plt.title('Fraction Reflections Correctly Indexed by Image')
plt.show()


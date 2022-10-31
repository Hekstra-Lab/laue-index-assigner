import numpy as np
import pandas as pd
from dxtbx.model import ExperimentList
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
from dials.array_family import flex

expt_file = "dials_temp_files/mega_ultra_refined.expt"
refl_file = "dials_temp_files/mega_ultra_refined.refl"
elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)
refls = reflection_table.from_file(refl_file)

def gen_experiment_identifiers(refls, elist):
    """Generates a mapping of the ID column in a reflection table 
    to the identifiers in an experiment list. The values of the ID
    column in the reflection table are taken to be the indices of
    the identifiers in order of the experiment list."""
    # Delete old mapping
    for k in refls.experiment_identifiers().keys():
        del refls.experiment_identifiers()[k]

    # Make arrays for keys and values
    indices = refls['id'].as_numpy_array()
    identifiers = np.empty_like(indices, dtype=np.dtype('U12'))

    # Populate identifiers based on indices
    for i, j in enumerate(indices):
        identifiers[i] = str(j)
        refls.experiment_identifiers()[int(j)] = identifiers[i]

    return refls

def get_rmsds(refls, refined=True):
    """Extracts some key data columns and gives returns a pandas table of the data, along with an array of RMSDs per image in a numpy array of format (x,y,total)."""
    # Use subset of reflection table if appropriate
    if refined:
        refls = refls.select(refls.get_flags(refls.flags.used_in_refinement))

    # Extract data from reflection table
    df = pd.DataFrame(data =
    np.asarray([refls['xyzobs.px.value'].parts()[0].as_numpy_array(),
        refls['xyzobs.px.value'].parts()[1].as_numpy_array(),
        refls['xyzcal.px'].parts()[0].as_numpy_array(),
        refls['xyzcal.px'].parts()[1].as_numpy_array(),
        refls['imageset_id'].as_numpy_array()]).T,
        columns = ['X','Y','Xpred','Ypred','imageset_id'])

    # Iterate over images to get RMSDs  
    img_ids = np.unique(df['imageset_id'].to_numpy(int))
    rmsds = np.zeros(shape=(len(img_ids), 3))
    for img in img_ids:
        img_df = df.loc[df['imageset_id'] == img]
        xy  = img_df[['X', 'Y']].to_numpy(float)
        xy_pred  = img_df[['Xpred', 'Ypred']].to_numpy(float)

        # Get residuals
        x_resids = xy_pred[:,0] - xy[:,0]
        y_resids = xy_pred[:,1] - xy[:,1]

        # Calculate RMSDs
        rmsd_x = np.sqrt(np.mean(x_resids**2))
        rmsd_y = np.sqrt(np.mean(y_resids**2))
        rmsd_tot = np.sqrt(np.mean(x_resids**2 + y_resids**2))

        # Store RMSDs
        idx = img - img_ids[0]
        rmsds[idx, 0] = rmsd_x
        rmsds[idx, 1] = rmsd_y
        rmsds[idx, 2] = rmsd_tot

    # Return data to user
    return df, rmsds

def plot_resids_by_image(refls, refined=True, image=-1):
    """Plot X, Y, Total residuals over images. A valid image number adds a 2D histogram of X,Y residuals for that image."""
    # Dependencies
    from matplotlib import pyplot as plt

    # Extract data from reflection table
    df, rmsds = get_rmsds(refls, refined=refined)
    imgs = np.unique(df['imageset_id'])

    # Plot data
    fig, axs = plt.subplots(3)
    axs[0].plot(imgs, rmsds[:, 0], 'r')
    axs[0].set(ylabel='X RMSD (px)')
    axs[1].plot(imgs, rmsds[:, 1], 'b')
    axs[1].set(ylabel='Y RMSD (px)')
    axs[2].plot(imgs, rmsds[:, 2], 'm')
    axs[2].set(xlabel='Image')
    axs[2].set(ylabel='Total RMSD (px)')
    fig.suptitle('Centroid RMSDs per Image')
    plt.show()

    if image >= 0:
        if not (image in imgs):
            print(f'Invalid image number. This reflection table has images {int(min(imgs))} to {int(max(imgs))}.')
            return
        img_df = df[df['imageset_id'] == image]
        resids = np.zeros(shape=(2, len(img_df)))
        resids[0, :] = (img_df['X'] - img_df['Xpred']).to_numpy(float)
        resids[1, :] = (img_df['Y'] - img_df['Ypred']).to_numpy(float)
        plt.hist2d(resids[0, :], resids[1, :], bins=len(img_df)//10)
        plt.colorbar()
        plt.title(f'X,Y Residuals for Image {image}')
        plt.xlabel('X Residuals (px)')
        plt.ylabel('Y Residuals (px)')
        plt.show()
    

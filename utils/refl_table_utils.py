import numpy as np
import pandas as pd
from dxtbx.model import ExperimentList
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family.flex import reflection_table
from dials.array_family import flex
from tqdm import trange

def get_rmsds(refls, refined=True, custom_inliers=None):
    """Extracts some key data columns and gives returns a pandas table of the data, along with an array of RMSDs per image in a numpy array of format (x,y,total)."""
    # Use subset of reflection table if appropriate
    if refined:
        inliers = refls.select(refls.get_flags(refls.flags.used_in_refinement))
    
    if custom_inliers is not None:
        if len(custom_inliers) != len(refls):
            print('Error: Custom inliers array not equal to reflection table length.')
            return 
        inliers = custom_inliers


    # Extract data from reflection table
    df = pd.DataFrame(data =
    np.asarray([refls['xyzobs.px.value'].parts()[0].as_numpy_array(),
        refls['xyzobs.px.value'].parts()[1].as_numpy_array(),
        refls['xyzcal.px'].parts()[0].as_numpy_array(),
        refls['xyzcal.px'].parts()[1].as_numpy_array(),
        refls['miller_index'].as_vec3_double().parts()[0].as_numpy_array(),
        refls['miller_index'].as_vec3_double().parts()[1].as_numpy_array(),
        refls['miller_index'].as_vec3_double().parts()[2].as_numpy_array(),
        refls['imageset_id'].as_numpy_array()]).T,
#        inliers]).T,
        columns = ['X','Y','Xpred','Ypred','H','K','L','imageset_id'])#, 'inlier'])

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

def plot_resids_by_image(refls, refined=True, density=True, image=-1, custom_inliers=None):
    """Plot X, Y, Total residuals over images. A valid image number adds a 2D histogram of X,Y residuals for that image. A KDE is used to generate a density plot if required."""
    # Dependencies
    from matplotlib import pyplot as plt

    # Extract data from reflection table
    df, rmsds = get_rmsds(refls, refined=refined, custom_inliers=custom_inliers)
    imgs = np.unique(df['imageset_id'])
    spot_counts = np.zeros(len(imgs))
    for i in trange(len(spot_counts)):
        spot_counts[i] = sum(df['imageset_id'] == i)

    # Plot data
    fig, axs = plt.subplots(4)
    axs[0].plot(imgs, rmsds[:, 0], 'r')
    axs[0].set(ylabel='X RMSD (px)')
    axs[1].plot(imgs, rmsds[:, 1], 'b')
    axs[1].set(ylabel='Y RMSD (px)')
    axs[2].plot(imgs, rmsds[:, 2], 'm')
    axs[2].set(ylabel='Total RMSD (px)')
    axs[3].plot(imgs, spot_counts, 'k')
    axs[3].set(xlabel='Image')
    axs[3].set(ylabel='Total Spots')
    fig.suptitle('Centroid RMSDs per Image')
    plt.show()

    if image >= 0:
        if not (image in imgs):
            print(f'Invalid image number. This reflection table has images {int(min(imgs))} to {int(max(imgs))}.')
            return
        img_df = df[df['imageset_id'] == image]
        resids = np.zeros(shape=(3, len(img_df)))
        resids[0, :] = (img_df['X'] - img_df['Xpred']).to_numpy(float)
        resids[1, :] = (img_df['Y'] - img_df['Ypred']).to_numpy(float)
        resids[2, :] = np.sqrt(resids[1, :]**2 + resids[0, :]**2)
        plt.hist2d(resids[0, :], resids[1, :], bins=len(img_df)//10)
        plt.colorbar()
        plt.title(f'X,Y Residuals for Image {image}')
        plt.xlabel('X Residuals (px)')
        plt.ylabel('Y Residuals (px)')
        plt.show()
 
        inliers = img_df['inlier'].to_numpy(bool)
        outliers = np.logical_not(inliers)
        plt.scatter(resids[0, :][outliers], resids[1, :][outliers], c='red')
        plt.scatter(resids[0, :][inliers], resids[1, :][inliers], c='green')
        plt.title('Residuals Colored by Outlier Rejection')
        plt.xlabel('X Residuals (px)')
        plt.ylabel('Y Residuals (px)')
        plt.legend(['Outlier','Inlier'])
        plt.show()

        plt.scatter(img_df['X'][outliers], img_df['Y'][outliers], c='red')
        plt.scatter(img_df['X'][inliers], img_df['Y'][inliers], c='green')
        plt.title('Observed Centroids Colored by Outlier Rejection')
        plt.xlabel('X (px)')
        plt.ylabel('Y (px)')
        plt.legend(['Outlier','Inlier'])
        plt.show()

        if density:
            import seaborn as sns
            quantiles = np.logspace(-10, 0, num=11)
            sns.kdeplot(resids[2, :], x=resids[0, :], y=resids[1, :], levels=quantiles)
            plt.xlabel('X Residuals (px)')
            plt.ylabel('Y Residuals (px)')
            plt.title(f'X,Y Residuals Density Plot for Image {image}')
            plt.show()
            
def miller_intersection(refls_list):                                                                                                                 
    """Get the intersection of all reflection tables according to Miller indices, and output the reflection tables truncated to that intersection."""
    # Dependencies
    from functools import reduce

    # Extract data from reflection tables in lists                                                                                                   
    df_list = []
    for refls in refls_list:
        df = pd.DataFrame(data =
        np.asarray([refls['xyzobs.px.value'].parts()[0].as_numpy_array(),
            refls['xyzobs.px.value'].parts()[1].as_numpy_array(),
            refls['xyzcal.px'].parts()[0].as_numpy_array(),
            refls['xyzcal.px'].parts()[1].as_numpy_array(),
            refls['miller_index'].as_vec3_double().parts()[0].as_numpy_array(),
            refls['miller_index'].as_vec3_double().parts()[1].as_numpy_array(),
            refls['miller_index'].as_vec3_double().parts()[2].as_numpy_array(),
            refls['imageset_id'].as_numpy_array()]).T,
            columns = ['X','Y','Xpred','Ypred','H','K','L','imageset_id'])
        df_list.append(df)

    # Reset column names
    for i, df in enumerate(df_list, start=1):                                                     
        df.rename(columns={col:'{}_{}'.format(col, i) for col in ('X', 'Y', 'Xpred', 'Ypred')}, inplace=True)

    # Merge data
    df_common = reduce( lambda left,right: pd.merge(left, right, on=['H','K','L','imageset_id'], how='inner'), df_list)
    return df_common


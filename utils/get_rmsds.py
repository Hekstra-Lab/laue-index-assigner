import numpy as np
import pandas as pd
from dials.array_family.flex import reflection_table

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
        xy  = df[['X', 'Y']].to_numpy(float)
        xy_pred  = df[['Xpred', 'Ypred']].to_numpy(float)

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

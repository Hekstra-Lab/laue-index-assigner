import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from dials.array_family import flex
from scipy.spatial import KDTree
from dxtbx.model.experiment_list import ExperimentList
from scipy.stats import multivariate_normal,poisson,norm
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D

refl_file = 'dials_temp_files/predicted.refl'
expt_file = 'dials_temp_files/imported.expt'
refls = flex.reflection_table.from_file(refl_file)
elist = ExperimentList.from_file(expt_file)

imageset_id = 46
isigi_cutoff = 2. #i over sigma cutoff for strong spots profiles
all_spots = refls.select(refls['imageset_id'] == imageset_id)['xyzcal.px'].as_numpy_array()[:,:2].astype('float32')
pixels = elist.imagesets()[0].get_raw_data(imageset_id)[0].as_numpy_array().astype('float32')

class Profile():
    def __init__(self, x, y, counts, cen_x=None, cen_y=None, fg_cutoff=1.0, bg_cutoff=3., minfrac=0.10, eps=1e-5, frac_step_size=0.50):
        self.frac_step_size = frac_step_size
        self.eps = eps
        self.minfrac = minfrac
        self.fg_cutoff = fg_cutoff
        self.bg_cutoff = bg_cutoff
        self.counts = counts
        self.fg_mask = np.ones_like(counts, dtype=bool)
        self.bg_mask = np.ones_like(counts, dtype=bool)

        self.x = x
        self.y = y
        self.cen_x = cen_x
        if self.cen_x is None:
            self.cen_x = np.average(self.x, weights=self.counts)
        self.cen_y = cen_y
        if self.cen_y is None:
            self.cen_y = np.average(self.y, weights=self.counts)

        self.scale = np.cov(self.difference_vectors.T, aweights=counts)
        self.slope = np.zeros(2)
        self.intercept = 0.
        self.update_mask()

    def set_profile_params(self, scale, slope, intercept, cen_x, cen_y): #might need more later just check
        self.scale = scale
        self.slope = slope
        self.intercept = intercept
        self.cen_x = cen_x
        self.cen_y = cen_y

        self.update_mask()
        self.update_background_plane()
        self.integrate()
        return self

    def fit(self, nsteps=20): #This number is probably larger than necessary
        self.success = True
        for i in range(nsteps):
            try:
                self.update_background_plane(alpha=self.frac_step_size)
                self.update_profile(alpha=self.frac_step_size)
                self.update_mask()
            except np.linalg.LinAlgError:
                self.success = False
                break

    @classmethod
    def from_dataframe(cls, df, x_key='dx', y_key='dy', count_key='counts', **kwargs):
        x = df[x_key].astype('float32')
        y = df[y_key].astype('float32')
        counts = df[count_key].astype('float32')
        return cls(x, y, counts, **kwargs)

    @property
    def difference_vectors(self):
        return np.column_stack((self.x - self.cen_x, self.y - self.cen_y))

    @property
    def background(self):
        return self.difference_vectors@self.slope + self.intercept

    def update_mask(self, fg_cutoff=None, bg_cutoff=None):
        if fg_cutoff is None:
            fg_cutoff = self.fg_cutoff
        if bg_cutoff is None:
            bg_cutoff = self.bg_cutoff

        dx = self.difference_vectors
        Z = np.sqrt(np.squeeze(dx[...,None,:]@np.linalg.inv(self.scale)@dx[...,:,None]))

        self.fg_mask = Z <= fg_cutoff
        self.bg_mask = Z >= bg_cutoff

        # There must be _some_ pixels in each group otherwise numerical issues happen
        minpix = np.round(len(self.counts) * self.minfrac).astype('int')
        zorder = np.argsort(Z)

        self.fg_mask[zorder[:minpix]] = True
        self.bg_mask[zorder[:minpix]] = False
        self.bg_mask[zorder[-minpix:]] = True
        self.fg_mask[zorder[-minpix:]] = False

    def update_background_plane(self, alpha=0.9):
        y = self.counts
        X = np.pad(self.difference_vectors, [[0, 0], [0, 1]], constant_values=1.)
        weights = np.reciprocal(self.counts)  #Inverse variance poisson because #stats
        weights = weights * (self.bg_mask).astype('float') 
        weights = weights / weights.mean()

        # Weighted least squares

        #beta = np.linalg.inv(X.T@np.diag(weights)@X)@X.T@y
        ## if really desperate 
        #try:
        #    beta = np.linalg.inv(X.T@np.diag(weights)@X)@X.T@y
        #except np.linalg.LinAlgError:
        #    beta = np.zeros(3) #Don't look at this. 

        beta = np.linalg.pinv(X.T@np.diag(weights)@X)@X.T@y #if desperate, break glass
        slope = beta[:2]
        intercept = beta[2]
        self.slope = (1. - alpha) * self.slope + alpha * slope
        self.intercept = (1. - alpha) * self.intercept + alpha * intercept

    def update_profile(self, alpha=0.9):
        weights = self.counts - self.background
        weights = np.maximum(weights, self.eps) #These can be negative so we need to make them > zero(ish)
        weights = weights * self.fg_mask
        weights = weights / weights.mean()

        scale = np.cov(self.difference_vectors.T, aweights=weights)

        cen_x = np.average(self.x, weights=weights)
        cen_y = np.average(self.y, weights=weights)

        self.scale = (1. - alpha) * self.scale + alpha * scale
        self.cen_x = (1. - alpha) * self.cen_x + alpha * cen_x
        self.cen_y = (1. - alpha) * self.cen_y + alpha * cen_y

    def integrate(self):
        bg = self.background
        var = bg + self.counts

        self.I = ((self.counts - bg) * self.fg_mask).sum()
        self.SigI = np.sqrt(((self.counts + bg) * self.fg_mask).sum())

class SegmentedImage():
    def __init__(self, pixels, centroids, radius=20):
        super().__init__()
        self.radius = radius
        pixels = np.array(pixels).astype('float32')

        x,y = np.indices(pixels.shape)
        xy = np.dstack(np.indices(pixels.shape)).astype('float32')[:,:,::-1]
        k = KDTree(centroids)
        distance,labels = k.query(xy)
        distance = distance.astype('float32')
        distance_vectors = centroids[labels] - xy
        mask = (distance <= radius) & (pixels > 0)

        self.data = pd.DataFrame({
            'x' : xy[mask][:,0].astype('int'),
            'y' : xy[mask][:,1].astype('int'),
            'counts' : pixels[mask],
            'label' : labels[mask],
            'distance' : distance[mask],
            'dx' : distance_vectors[mask][:,0],
            'dy' : distance_vectors[mask][:,1],
            'cen_x' : centroids[labels[mask], 0],
            'cen_y' : centroids[labels[mask], 1],
        })
        self.profiles = self.data.groupby('label').apply(Profile.from_dataframe)
        [p.fit() for p in self.profiles]
        [p.integrate() for p in self.profiles]
        self.scale = np.stack([p.scale for p in self.profiles])
        self.loc = np.stack([(p.cen_x , p.cen_y) for p in self.profiles]) 
    
        g = self.data.groupby('label')
        self.bboxes = np.column_stack([
            g.min().x,
            g.max().x+1,
            g.min().y,
            g.max().y+1,
        ])
        self.pixels = pixels
        self.centroids = centroids[np.unique(self.data.label)]
        self.labels = labels
        self.distance = distance
        self.distance_vectors = distance_vectors

    def integrate(self, isigi_cutoff=2., knn=5):
        # Implement k nearest-neighbors for weak spots to average profile params from nearby strong spots
        strong_idx = np.array([i for i,p in enumerate(self.profiles) if p.I/p.SigI > isigi_cutoff])
        strong_centroids = self.centroids[strong_idx]
        for i, p in enumerate(self.profiles):
            scale = np.zeros_like(p.scale)
            slope = np.zeros_like(p.slope)
            intercept = np.zeros_like(p.intercept)
            cen_x = np.zeros_like(p.cen_x)
            cen_y = np.zeros_like(p.cen_y)

            # Average over the knn profiles
            for idx in strong_idx[np.argsort(np.linalg.norm(strong_centroids - self.centroids[i], axis=-1))][:knn]:
                nearest_neighbor = self.profiles.iloc[idx]
                scale += nearest_neighbor.scale / knn
                slope += nearest_neighbor.slope / knn
                intercept += nearest_neighbor.intercept / knn
                cen_x += nearest_neighbor.cen_x / knn
                cen_y += nearest_neighbor.cen_y / knn

            p.set_profile_params(
                scale,
                slope,
                intercept,
                cen_x,
                cen_y,
            )
        self.scale = np.stack([p.scale for p in self.profiles])
        self.loc = np.stack([(p.cen_x , p.cen_y) for p in self.profiles]) 


    def plot_image(self, **kwargs):
        vmin = kwargs.pop('vmin', 5.)
        vmax = kwargs.pop('vmax', 50.)
        plt.imshow(self.pixels, cmap='cividis', origin='lower', vmin=vmin, vmax=vmax, **kwargs)

    def plot_bboxes(self, **kwargs):
        self.plot_image(**kwargs)
        plt.plot(self.bboxes[:,[0, 0, 1, 1, 0]].T, self.bboxes[:,[3, 2, 2, 3, 3]].T, 'w-')

    def plot_profiles(self, isigi_cutoff=None, **kwargs):
        from matplotlib.patches import Ellipse
        from matplotlib.transforms import Affine2D
        self.plot_image(**kwargs)

        ax = plt.gca()
        pearson = self.scale[...,0, 1] / np.sqrt(self.scale[...,0,0] * self.scale[...,1,1])

        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        loc_x,loc_y = (-self.loc + self.centroids).T

        # you may wish to plot multiple contours such as:
        # for n_std in [1., 2., 3.]:
        #    ...
        #
        # but let's just do 2 std devs for now.
        for n_std in [2.,]:
            scale_x,scale_y = np.sqrt(np.diagonal(self.scale, axis1=-2, axis2=-1)).T * n_std
            for i in range(len(self.scale)):
                ecolor = 'r'
                p = self.profiles.iloc[i]
                if p.success:
                    ecolor = 'w'
                    if isigi_cutoff is not None:
                        isigi = p.I / p.SigI
                        if isigi >= isigi_cutoff:
                            ecolor = 'y'
                ellipse = Ellipse((0, 0), width=ell_radius_x[i] * 2, height=ell_radius_y[i] * 2,
                    facecolor='none', edgecolor=ecolor)
                transf = Affine2D() \
                    .rotate_deg(45.) \
                    .scale(scale_x[i], scale_y[i]) \
                    .translate(loc_x[i], loc_y[i])
                ellipse.set_transform(transf + ax.transData)
                ax.add_patch(ellipse)

sim = SegmentedImage(pixels, all_spots)
sim.integrate(isigi_cutoff)
sim.plot_profiles()
plt.show()

# TODO: Make this into an mtz


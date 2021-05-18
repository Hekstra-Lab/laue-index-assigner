import numpy as np
import reciprocalspaceship as rs
import gemmi
from IPython import embed


"""
This is just a place to stash useful functions and classes for diffraction geometry calculations
"""

def hkl2ray(hkl, wavelength=None):
    """ 
    Convert a miller index to the shortest member of its central ray. 
    Optionally, adjust its wavelength accordingly.

    Parameters
    ----------
    hkl : array
        `n x 3` array of miller indices. the dtype must be interpretable as an integeger
    wavelength : array (optional)
        length `n` array of wavelengths corresponding to each miller index

    Returns
    -------
    reduced_hkl : array
        The miller index of the shortest vector on the same central ray as hkl
    reduced_wavelength : array (optional)
        The wavelengths corresponding to reduced_hkl
    """ 
    gcd = np.gcd.reduce(hkl.astype(int), axis=1)
    if wavelength is not None:
        return hkl/gcd[:,None], wavelength*gcd
    else:
        return hkl/gcd[:,None]

def is_ray_equivalent(hkl1, hkl2):
    """
    Test for equivalency between two miller indices in a Laue experiment. Returns a boolean array for each of the `n` hkls in `hkl{1,2}`.
    """
    return np.all(np.isclose(hkl2ray(hkl1),  hkl2ray(hkl2)), axis=1)

def align_hkls(reference, target, spacegroup, anomalous=True):
    """
    Use the point group operators in `spacegroup` to align target hkls to reference.    

    Parameters
    ----------
    reference : array
        n x 3 array of miller indices
    target : array
        n x 3 array of miller indices
    spacegroup : gemmi.Spacegroup
        The space group of reference/target.
    anomalous : bool(optional)
        If true, test Friedel symmetries too.

    Returns
    -------
    aligned : array
        n x 3 array of miller indices equivalent to target
    """
    aligned = target
    cc = -1.
    for op in spacegroup.operations():
        aligned_ = rs.utils.apply_to_hkl(target, op)
        cc_mat = np.corrcoef(aligned_.T, reference.T)
        cc_ = np.trace(cc_mat[:3,3:])
        if cc_ > cc:
            aligned = aligned_
            cc = cc_
    if anomalous:
        for op in spacegroup.operations():
            aligned_ = -rs.utils.apply_to_hkl(target, op)
            cc_mat = np.corrcoef(aligned_.T, reference.T)
            cc_ = np.trace(cc_mat[:3,3:])
            if cc_ > cc:
                aligned = aligned_
                cc = cc_
    return aligned


class Detector():
    def __init__(self, ax1, ax2, ori):
        """ 
        This is a simple class to represent a detector panel. 
        """
        super().__init__()
        self.D = np.vstack((
            ax1,
            ax2,
            ori,
        ))

    @classmethod
    def from_inp_file(cls, filename, **kwargs):
        """ doesn't account for tilt yet """
        for line in  open(filename):
            if 'Distance' in line:
                ddist = float(line.split()[1])
            if 'Center' in line:
                beam = np.array([float(line.split()[1]), float(line.split()[1])])
            if 'Pixel' in line:
                size = np.array([float(line.split()[1]), float(line.split()[1])])

        ax1 = np.array([size[0], 0., 0.])
        ax2 = np.array([0., size[1], 0.])
        ori = np.array([
                -beam[0] * size[0], 
                -beam[1] * size[1], 
                ddist
            ])
        return cls(ax1, ax2, ori, **kwargs)

    @classmethod
    def from_expt_file(cls, filename, **kwargs):
        """
        Return a generator of Detector instances for each panel described in `filename`
        """
        from dxtbx.model.experiment_list import ExperimentListFactory
        elist = ExperimentListFactory.from_json_file(filename)
        for detector in elist.detectors():
            for panel in detector.iter_panels():
                ori = np.array(panel.get_origin())
                size = panel.get_pixel_size()
                ax1 = np.array(panel.get_fast_axis())*size[0]
                ax2 = np.array(panel.get_slow_axis())*size[1]
                yield cls(ax1, ax2, ori, **kwargs)

    def pix2lab(self, xy, D=None):
        """ Map xy to 3d coordinates in the lab frame in mm"""
        if D is None:
            D = self.D
        xy = np.pad(xy, ((0,0), (0,1)), constant_values=1.)
        S1 = xy@D
        return S1

    def lab2pix(self, xyz, D=None):
        """ Map 3d coordinates in the lab frame in mm to pixels """
        if D is None:
            D = self.D
        Dinv = np.linalg.inv(D.double()).float()
        return (xyz@Dinv)[:,:2]

    def s12pix(self, s1, D=None):
        """ Project a scattered beam wavevector into pixel coordinates. """
        if D is None:
            D = self.D
        Dinv = np.linalg.inv(D.double()).float()
        xya = s1@Dinv
        xy = xya[:,:2]/xya[:,2,None]
        return xy

def orthogonalization(a, b, c, alpha, beta, gamma):
    """ Compute the orthogonalization matrix from cell params """
    alpha,beta,gamma = np.pi*alpha/180.,np.pi*beta/180.,np.pi*gamma/180.
    cosa = np.cos(alpha)
    cosb = np.cos(beta)
    cosg = np.cos(gamma)
    sing = np.sin(gamma)
    V = a*b*c*np.sqrt(1. - cosa*cosa-cosb*cosb-cosg*cosg + 2*cosa*cosb*cosg)
    return np.array([
        [a, b*cosg, c*cosb],
        [0., b*sing, c*(cosa-cosb*cosg)/sing],
        [0., 0., V/a/b/sing],
    ], device=a.device)

def mat_to_rot_xyz(R, deg=True):
    """ Decompose a rotation matrix into euler angles """
    if R[2, 0] < 1:
        if R[2,0] > -1:
            rot_y = np.asin(-R[2, 0])
            rot_z = np.atan2(R[1, 0], R[0, 0])
            rot_x = np.atan2(R[2, 1], R[2, 2])
        else:
            rot_y = np.pi/2.
            rot_z = -np.atan2(-R[1,2], R[1,1])
            rot_x = 0.
    else:
        rot_y = np.pi/2.
        rot_z = np.atan2(-R[1,2], R[1,1])
        rot_x = 0.
    if deg:
        rot_x = np.rad2deg(rot_x) 
        rot_y = np.rad2deg(rot_y)
        rot_z = np.rad2deg(rot_z)
    return rot_x, rot_y, rot_z

def rot_xyz_to_mat(rot_x, rot_y, rot_z, deg=True):
    """ Convert euler angles into a rotation matrix """
    if deg:
        rot_x = np.deg2rad(rot_x)
        rot_y = np.deg2rad(rot_y)
        rot_z = np.deg2rad(rot_z)
    cx,sx = np.cos(rot_x),np.sin(rot_x)
    cy,sy = np.cos(rot_y),np.sin(rot_y)
    cz,sz = np.cos(rot_z),np.sin(rot_z)

    return np.array([
        [cy*cz, cz*sx*sy-cx*sz, cx*cz*sy+sx*sz],
        [cy*sz, cx*cz+sx*sy*sz, -cz*sx+cx*sy*sz],
        [-sy, cy*sx, cx*cy],
    ], device=rot_x.device)

class LaueAssigner():
    """
    This class can be used to assign miller indices to Laue data with 
    poor initial geometry.
    """
    def __init__(self, s0, s1_obs, lam_min, lam_max, hmax, RB):
        """
        Parameters
        ----------
        s0 : array
            The (potentially unnormalized) direction of the incoming beam wavevector.
        s1_obs : array
            The `n x 3` array of observed scattered beam wavevector directions. These will
            be normalized. 
        lam_min : float
            The lower end of the wavelength range of the beam.
        lam_max : float
            The upper end of the wavelength range of the beam.
        hmax : array
            A vector [hmax, kmax, lmax] specifying the largest miller indices to be 
            considered along each of the reciprocal basis vectors. 
        RB : array
            The approximate indexing solution. This matrix is called Astar some places, 
            but it contains the reciprocal basis vectors as its *rows*. 
            ```python
            RB = np.array([
                [astar_1, astar_2, astar_3], 
                [bstar_1, bstar_2, bstar_3], 
                [cstar_1, cstar_2, cstar_3], 
            ])
            ```
            It is equal to an orthogonal matrix `R` times the matrix `B` which is related
            to the orthogonalization matrix, `O`, by
            ```python
            B = np.linalg.inv(O).T
            ```
        """
        self.s0 = np.array(s0) / np.linalg.norm(s0)
        self.lam_min,self.lam_max = lam_min,lam_max
        self.hmax = hmax
        self.s1_obs = np.array(s1_obs) / np.linalg.norm(s1_obs, axis=1)[:,None]
        self.H = None
        self.RB = np.array(RB)
        self.ewald_offset = None
        self.wavelengths = None
        self.assign()

    def assign(self):
        """
        This function updates H with the current best miller indices by searching the entire
        set of feasible reflections specified by self.lam_min, self.lam_max, self.hmax, and the 
        current value of self.RB. It also updates self.ewald_offset which is used to weight 
        the optimization objective. It will also set self.L to the current most probable
        wavelengths.

        Returns
        -------
        self : LaueAssigner
            This method returns self to enable chaining.
        """
        s0 = self.s0
        s1 = self.s1_obs
        Q = (s1 - s0)
        RB = self.RB
        lam_min,lam_max = self.lam_min,self.lam_max

        # This block generates the entire reciprocal grid and then removes infeasible reflections
        # TODO: to save memory, write a cython version of this using a triply nested for-loop 
        # which never instantiates the infeasible reflecitons
        hmax,kmax,lmax = self.hmax
        Hall = np.mgrid[
            -hmax:hmax+1:1,
            -kmax:kmax+1:1,
            -lmax:lmax+1:1,
        ].reshape((3, -1)).T
        Hall = Hall[np.any(Hall!=0, axis=1)] #<-- remove 0,0,0
        Hall = Hall[((Hall / self.hmax)**2.).sum(-1) <= 1.]

        Qpred = (RB@Hall.T).T
        feasible = (
            (np.linalg.norm(Qpred + s0/lam_min, axis=1) < 1/lam_min) & 
            (np.linalg.norm(Qpred + s0/lam_max, axis=1) > 1/lam_max)
        ) 
        Qpred = Qpred[feasible]
        lpred = - 2. * (Qpred@(s0)) * np.linalg.norm(Qpred, axis=1)**-2.
        Qlpred = Qpred*lpred[:,None]

        # This block matches the predicted scattering vectors to the observed ones
        # TODO: add some checks for redundant assignments. 
        distmat = np.linalg.norm(Q[:,None,:] - Qlpred[None,:,:], axis=-1)
        idx = np.argmin(distmat, axis=1)
        H_best = Hall[feasible][idx]
        self.ewald_offset = np.linalg.norm(Qlpred[idx] - Q, axis=1)
        self.H = H_best
        self.wavelengths = lpred[idx]
        return self

    def optimize_bases(self, rlp_radius=0.002, steps=10):
        """
        Use convex optimization to improve the quality of the indexing matrix, self.RB.
        This method solves a weighted L1 norm mnimization problem 

        ```python
        minimize ||w*(k*Qpred.T - RB@H.T)||_1
             st.  1./lam_max <= k <= 1./lam_min
        ```
        Where the weights, w, are
        ```python
        w = np.exp(-0.5*(ewald_offset/rlp_radius)**2.)
        w /= w.mean()
        ```
        After optimizing the bases, this method will call `self.assign` to update 
        `self.H` to be consistent with the new bases. 

        Parameters
        ----------
        rlp_radius : float (optional)
            An approximate value for the radius of the reciprocal lattice points in 
            inverse angstroms. This is used for weighting the objective function.
            A suitable default is `0.002`.
        steps : int (optional)
            The number of cycles of convex optimization to run. The default is 10 which 
            seems sufficient. 

        Returns
        -------
        self : LaueAssigner
            This method returns self to enable chaining.
        """
        import cvxpy as cvx
        s0 = self.s0
        s1 = self.s1_obs
        Q = (s1 - s0)
        lam_min,lam_max = self.lam_min,self.lam_max

        hkl_pred,r = self.H, self.ewald_offset

        k = cvx.Variable(len(self.s1_obs))
        RB = cvx.Variable((3,3))
        cons = [
            k >= 1./lam_max,
            k <= 1./lam_min,
        ]

        losses = []
        correct = []

        for i in range(steps):
            w = np.exp(-0.5*(r/rlp_radius)**2.)
            w /= w.mean()
            Qpred = RB@hkl_pred.T
            Ql = cvx.multiply(Q, k[:,None]).T
            resid = Qpred - Ql
            loss = cvx.norm1(cvx.multiply(w[None,:], resid)) 
            p = cvx.Problem(
                cvx.Minimize(loss),
                cons,
            )
            p.solve(solver='ECOS', verbose=False, max_iters=1000)
            r = np.linalg.norm(Ql.value - Qpred.value, axis=0)
            hkl_pred = (np.linalg.inv(RB.value)@Ql.value).T
            hkl_pred = np.round(hkl_pred)
            #TODO: implement early termination by checking loss.value for convergence

        self.RB = RB.value
        self.assign()


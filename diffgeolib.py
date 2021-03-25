import numpy as np
import reciprocalspaceship as rs
import gemmi


"""
This is just a place to stash useful functions and classes for diffraction geometry calculations
"""

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
    cc = np.corrcoef(reference, aligned).sum()
    for op in spacegroup.operations():
        aligned_ = rs.utils.apply_to_hkl(target, op)
        cc_mat = aligned_.T@reference
        cc_ = np.trace(cc_mat)
        if cc_ > cc:
            aligned = aligned_
            cc = cc_
    if anomalous:
        for op in spacegroup.operations():
            aligned_ = -rs.utils.apply_to_hkl(target, op)
            cc_ = np.corrcoef(reference, aligned_).sum()
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
        if D is None:
            D = self.D
        xy = np.pad(xy, ((0,0), (0,1)), constant_values=1.)
        S1 = xy@D
        return S1

    def lab2pix(self, xyz, D=None):
        if D is None:
            D = self.D
        Dinv = np.linalg.inv(D.double()).float()
        return (xyz@Dinv)[:,:2]

    def s12pix(self, s1, D=None):
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


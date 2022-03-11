import numpy as np
import reciprocalspaceship as rs
import gemmi
from diffgeolib import hkl2ray
from IPython import embed

class LauePredictor():
    """
    An object to predict spots given a Laue experiment.
    """
    def __init__(self, s0, cell, R, lam_min, lam_max, dmin, spacegroup='1'):
        """
        Parameters
        ----------
        s0 : array
            a 3 vector indicating the direction of the incoming beam wavevector.
            This can be any length, it will be unit normalized in the constructor.
        cell : iterable or gemmi.UnitCell
            A tuple or list of unit cell params (a, b, c, alpha, beta, gamma) or a gemmi.UnitCell object
        R : array
            A 3x3 rotation matrix corresponding to the crystal orientation for the frame.
        lam_min : float
            The lower end of the wavelength range of the beam.
        lam_max : float
            The upper end of the wavelength range of the beam.
        dmin : float
            The maximum resolution of the model
        spacegroup : gemmi.SpaceGroup (optional)
            Anything that the gemmi.SpaceGroup constructor understands.
        """
        if not isinstance(cell, gemmi.UnitCell):
            cell = gemmi.UnitCell(*cell)
        self.cell = cell

        if not isinstance(spacegroup, gemmi.SpaceGroup):
            spacegroup = gemmi.SpaceGroup(spacegroup)
        self.spacegroup = spacegroup

        self.R = R
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.dmin = dmin
        self.B = np.array(self.cell.fractionalization_matrix).T

        # self.s{0,1} are dynamically masked by their outlier status
        self.s0 = s0 / np.linalg.norm(s0)

        # Initialize the full reciprocal grid
        hmax,kmax,lmax = self.cell.get_hkl_limits(dmin)
        Hall = np.mgrid[
            -hmax:hmax+1:1.,
            -kmax:kmax+1:1.,
            -lmax:lmax+1:1.,
        ].reshape((3, -1)).T
        Hall = Hall[np.any(Hall != 0, axis=1)]
        d = cell.calculate_d_array(Hall)
        Hall = Hall[d >= dmin]

        # TODO: Consider adding a flag to remove any systematic absences in a supplied space group

    @property
    def RB(self):
        return self.R@self.B

    def predict_s1(self):
        """ 
        Predicts all s1 vectors for all feasible spots given some resolution-dependent bandwidth

        This method provides:
            s1_pred -- predicted feasible s1 vectors
        """
        # Generate the feasible set of reflections from the current geometry
        Hall = self.Hall
        qall = (self.RB@Hall.T).T
        feasible = (
            (np.linalg.norm(qall + self.s0/self.lam_min, axis=-1) < 1/self.lam_min) & 
            (np.linalg.norm(qall + self.s0/self.lam_max, axis=-1) > 1/self.lam_max)
        ) 
        Hall = Hall[feasible]
        qall = qall[feasible]

        # Remove harmonics from the feasible set
        Raypred = hkl2ray(Hall)
        _,idx   = np.unique(Raypred, return_index=True, axis=0)
        Hall = Hall[idx]
        qall = qall[idx]

        # Filter reflections which do not satisfy the resolution-dependent bandwidth
        # TODO: Skip for now and implement this filtration later just to see -- we'll overpredict at high resolution

        # For each q, find the wavelength of the Ewald sphere it lies on
        lams = -2.*(self.s0 * qall).sum(-1) / (qall*qall).sum(-1)

        # Using this wavelength per q, generate s1 vectors
        s0 = self.s0 / lams
        s1_pred = qall + s0

        # Write s1 predictions
        return s1_pred

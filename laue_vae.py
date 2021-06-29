import tensorflow as tf
import numpy as np
from tensorflow import keras as tfk
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from scipy.optimize import linear_sum_assignment


batch_size = 32
n_layers  = 20
mlp_width = 10
hmax = 50
kl_weight=1.

class SpotPredictor(tfk.layers.Layer):
    def __init__(self, hkl_max, s_0, lam_min, lam_max, Dmat, pix_size):
        super().__init__()
        self.pix_size=pix_size
        self.lam_min,self.lam_max = lam_min,lam_max
        self.s_0 = s_0
        self.hkl_max = hkl_max
        h,k,l = self.hkl_max
        self.Hall = np.mgrid[
            -h:h+1:1.,
            -k:k+1:1.,
            -l:l+1:1.,
        ].reshape((3, -1)).T.astype(np.float32)
        self.Hall = self.Hall[~np.all(self.Hall == 0., axis=1)]
        self.Dinv = np.linalg.inv(Dmat).astype('float32')

    def feasible_millers(self, RB):
        l = self.wavelength(RB, self.Hall)
        feasible = (l >= self.lam_min) & (l <= self.lam_max)
        return tf.boolean_mask(self.Hall, feasible, axis=-1)

    def wavelength(self, RB, H):
        Q = tf.transpose(RB@self.Hall.T)
        num = 2.*Q@self.s_0[:,None] 
        den = tf.linalg.norm(Q, axis=-1)[:,None]**2
        return num / den

    def call(self, RB):
        Q = tf.transpose(RB@self.Hall.T)
        num = -2.*Q@self.s_0[:,None] 
        den = tf.linalg.norm(Q, axis=-1)[:,None]**2
        l = num / den
        s1 = Q + self.s_0/l
        xya = s1 @ self.Dinv
        x,y,a = tf.unstack(xya, axis=-1)
        x,y = x/a,y/a
        x,y = x[:,None],y[:,None]
        xy = tf.concat((x, y), axis=-1)
        x_max,y_max = self.pix_size
        feasible = (l >= self.lam_min) & (l <= self.lam_max) & (x >= 0) & (x <= x_max) & (y >= 0) & (y <= y_max)
        return tf.boolean_mask(xy[...,None,:], feasible[...,:])

class H2Q(tfk.layers.Layer):
    def call(self, H, RB):
        return RB@H

from diffgeolib import Detector
Dmat = next(Detector.from_expt_file("dials_temp_files/refined_varying.expt")).D.astype(np.float32)
ig = SpotPredictor((50, 50, 50), np.array([0, 0, -1.]), 1., 1.2, Dmat, (3840, 3840))
RB = tf.eye(3)/50.
xy = ig.call(RB)


class Cell2O(tfk.layers.Layer):
    def __init__(self,  deg=True, verbose=False):
        super().__init__()
        self.deg = deg
        self.verbose = verbose

    def call(self, inputs):
        a,b,c,alpha,beta,gamma = tf.unstack(inputs, axis=-1)
        if self.deg:
            alpha,beta,gamma = np.pi*alpha/180.,np.pi*beta/180.,np.pi*gamma/180.

        #Trig funcs
        ca = tf.cos(alpha)
        sa = tf.sin(alpha)
        cb = tf.cos(beta)
        sb = tf.sin(beta)
        cg = tf.cos(gamma)
        sg = tf.sin(gamma)

        #Cell volume
        V = a*b*c*tf.sqrt(1. - ca*ca - cb*cb - cg*cg + 2.*ca*cb*cg)

        #Construct columns
        A = tf.stack([a, tf.zeros_like(a), tf.zeros_like(a)], axis=-1)
        B = tf.stack([b*cg, b*sg, tf.zeros_like(a)], axis=-1)
        C = tf.stack([c*cb, c*(ca-cb*cg)/sg, V/a/b/sg], axis=-1)

        #Construct matrices
        O = tf.stack([A, B, C], axis=-1)

        if self.verbose:
            print(f"""
            a: {a}
            b: {b}
            c: {c}
            alpha: {alpha}
            beta : {beta}
            gamma: {gamma}
            V: {V}
            """)

        return O

class O2B(tfk.layers.Layer):
    def call(self, O):
        Oinv = tf.linalg.inv(O)
        return tf.linalg.matrix_transpose(Oinv)

class MatInverse(tfk.layers.Layer):
    def call(self, inputs):
        return tf.linalg.inv(inputs)

class SigmaSoftplus(tfk.layers.Layer):
    def call(self, inputs):
        mu, sigma = tf.unstack(inputs, axis=-1)
        sigma = tf.math.softplus(sigma)
        return tf.stack((mu, sigma), axis=-1)

class NormalVariational(tfk.layers.Layer):
    def __init__(self, prior, kl_weight=1., low=1e-32, high=1e32):
        super().__init__()
        self.low = low
        self.high = high
        self.prior = prior

    def call(self, inputs):
        mu, sigma = tf.unstack(inputs, axis=-1)
        q = tfd.TruncatedNormal(mu, sigma, self.low, self.high)
        z = q.sample()
        if self.prior is not None:
            kl_div = q.log_prob(z) - self.prior.log_prob(z)
            self.add_loss(kl_div)
        return  z

class Ortho6DToMatrix(tfk.layers.Layer):
    def call(self, ortho_6d):
        x_raw,y_raw = tf.unstack(ortho_6d, axis=-1)
        x = tf.linalg.normalize(x_raw, axis=-1)[0]
        z = tf.linalg.cross(x, y_raw)
        z = tf.linalg.normalize(z, axis=-1)[0]
        y = np.cross(z, x)
        matrix = tf.stack((x, y, z), axis=-1)
        return matrix

class LinearSumLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        cost = tf.reduce_sum((y_true[...,:,None] - y_pred[...,None,:])**2., axis=-1)
        idx = tf.numpy_function(lambda c: linear_sum_assignment(c)[1], cost)
        y_pred = tf.gather(y_pred, idx)
        return self._base_loss(y_true, y_pred)


def loader(expt_file, refl_file, batch_size=32):
    def dials_gen():
        from dxtbx.model.experiment_list import ExperimentListFactory
        from dials.array_family.flex import reflection_table
        from dials.array_family import flex
        elist = ExperimentListFactory.from_json_file(expt_file, check_format=False)
        refls = reflection_table.from_file(refl_file)
        for i in range(elist[0].imageset.size()):
            idx = refls['xyzobs.px.value'].parts()[2] - 0.5 == i 
            subrefls = refls.select(idx) 
            pix_position = subrefls['xyzobs.px.value'].as_numpy_array()[:,:2].astype('float32')
            I = subrefls['intensity.sum.value'].as_numpy_array()
            idx = np.argsort(I)[::-1][:batch_size]
            yield(pix_position[idx][None,:,:])
    return dials_gen

# Load DIALS files                                                   
expt_file = "dials_temp_files/refined_varying.expt" #<-- Fill this in
refl_file = "dials_temp_files/refined_varying.refl" #<-- Fill this in
data = tf.data.Dataset.from_generator(loader(expt_file, refl_file), output_types=tf.float32)
for x in data:
    break

cell_loc  = np.array([34., 45., 98., 90.0, 90.0, 90.0], dtype='float32')
cell_scale= np.array([ 1.,  1.,  2.,  0.1,  0.1,  0.1], dtype='float32')
pcell = tfd.Normal(cell_loc, cell_scale)

tf.keras.Input((batch_size, 2))

blayers = []
blayers.append(tfk.layers.Flatten())
blayers.extend([tfk.layers.Dense(batch_size*2, activation=tfk.layers.LeakyReLU()) for i in range(n_layers)])
blayers.extend([tfk.layers.Dense(mlp_width, activation=tfk.layers.LeakyReLU()) for i in range(n_layers)])
blayers.append(tfk.layers.Dense(12, activation='softplus', bias_initializer=tfk.initializers.Constant(value=90.)))
blayers.append(tfk.layers.Reshape((6, 2)))
blayers.append(NormalVariational(pcell, kl_weight))
blayers.append(Cell2O(verbose=True))
blayers.append(O2B())

rlayers = []
rlayers.append(tfk.layers.Flatten())
rlayers.extend([tfk.layers.Dense(batch_size*2, activation=tfk.layers.LeakyReLU()) for i in range(n_layers)])
rlayers.extend([tfk.layers.Dense(mlp_width, activation=tfk.layers.LeakyReLU()) for i in range(n_layers)])
rlayers.append(tfk.layers.Dense(6, activation='linear'))
rlayers.append(tfk.layers.Reshape((3, 2)))
rlayers.append(Ortho6DToMatrix())

basis_model = tfk.Sequential(blayers)
rot_model   = tfk.Sequential(rlayers)

basis_model(x)
rot_model(x)

from IPython import embed
embed(colors='linux')


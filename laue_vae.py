import tensorflow as tf
import numpy as np
from tensorflow import keras as tfk
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from scipy.optimize import linear_sum_assignment


xy_batch_size = 32
n_layers  = 20
mpl_width = 10
hmax = 50

class IndexGenerator(tfk.layers.Layer):
    def __init__(self, hkl_max, s_0, lam_min, lam_max):
        super().__init__()
        self.lam_min,self.lam_max = lam_min,lam_max
        self.s_0 = s_0
        self.hkl_max = hkl_max
        h,k,l = self.hkl_max
        self.Hall = np.mgrid[
            -h:h+1:1,
            -k:k+1:1,
            -l:l+1:1,
        ].reshape((3, -1)).T

    def call(self, RB):
        Q = tf.transpose(RB@self.Hall.T)
        feasible = (
            (tf.linalg.norm(Q + self.s_0/self.lam_min, ord=2, axis=-1) < 1/self.lam_min) &
            (tf.linalg.norm(Q + self.s_0/self.lam_max, ord=2, axis=-1) > 1/self.lam_max) 
        )
                #return tf.gather(Q, feasible, axis=-2)
        return tf.boolean_mask(self.Hall, feasible, axis=-2)

ig = IndexGenerator((10, 10, 10), np.array([0, 0, -1.]), 1., 1.2)
h = ig(tf.eye(3)/50.)

from IPython import embed
from sys import exit
embed(colors='linux')
exit()

class SigmaSoftplus(tfk.layers.Layer):
    def call(self, inputs):
        mu, sigma = tf.unstack(inputs, axis=-1)
        sigma = tf.math.softplus(sigma)
        return tf.stack((mu, sigma), axis=-1)

class NormalVariational(tfk.layers.Layer):
    def __init__(self, prior, kl_weight=1.):
        self.prior = prior
        super().__init__()

    def call(self, inputs):
        mu, sigma = tf.unstack(inputs, axis=-1)
        q = tfd.Normal(mu, sigma)
        z = q.sample()
        if self.prior is not None:
            kl_div = q.log_prob(z) - self.prior.log_prob(z)
            self.add_loss(kl_div)
        return  z

class ConcatMillerLayer():
    def __init__(self, hmax, base_loss=None):
        super().__init__()
        self.hmax
        h,k,l = self.hmax
        self.Hall = np.mgrid[
                -h:h+1:2*h+1,
                -k:k+1:2*k+1,
                -l:l+1:2*l+1,
            ].reshape((-1, 3))
        if base_loss is None:
            self._base_loss = tfk.losses.MSE()

    def call(self, inputs):
        z = self.inputs
        return tf.stack((z*tf.ones((self.Hall.shape[0], 1)), self.Hall), axis=-1)

class Ortho6DToMatrix(tfk.layers.Layers):
    def call(self, ortho_6d):
        x_raw,y_raw = tf.unstack(orth_6d, axis=-1)
        x = tf.linalg.normalize(x_raw, axis=-1)
        z = tf.linalg.cross(x, y_raw)
        z = tf.linalg.normalize(z_raw, axis=-1)
        y = np.cross(z, x)
        matrix = tf.stack((x, y, z), axis=-1)
        return matrix

class LinearSumLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        cost = (y_true[...,:,None] - y_pred[...,None,:])**2.
        idx = tf.numpy_function(lambda c: linear_sum_assignment(c)[1], cost)
        y_pred = tf.gather(y_pred, idx)
        return self._base_loss(y_true, y_pred)

cell_loc  = np.array([34., 45., 98., 90.0, 90.0, 90.0])
cell_scale= np.array([ 1.,  1.,  2.,  0.1,  0.1,  0.1])
pcell = tfd.Normal(cell_loc, cell_std)

blayers = []
blayers.extend([tfk.layers.Dense((xy_batch_size, 2), activation=tfk.activations.LeakyReLU()) for i in range(n_layers)])
blayers.extend([tfk.layers.Dense(mlp_width, activation=tfk.activations.LeakyReLU()) for i in range(n_layers)])
rlayers.append(tfk.layers.Dense(6, activation='linear'))
blayers.append(SigmaSoftplus())
blayers.append(NormalVariational())

rlayers = []
blayers.extend([tfk.layers.Dense((xy_batch_size, 2), activation=tfk.activations.LeakyReLU()) for i in range(n_layers)])
rlayers.extend([tfk.layers.Dense(mlp_width, activation=tfk.activations.LeakyReLU()) for i in range(n_layers)])
rlayers.append(tfk.layers.Dense(6, activation='linear'))
rlayers.append(Ortho6DToMatrix())

dlayers = []
dlayers.append(tfk.layers.Dense(4))
dlayers.extend([tfk.layers.Dense(mlp_width, activation=tfk.activations.LeakyReLU()) for i in range(n_layers)])
dlayers.append(2, activation='linear')



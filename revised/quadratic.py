import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

import tensorflow as tf
from tensorflow.keras.losses import Loss

class Quadratic():
    
    def __init__(self, random_seed, **kwargs):

        npr.seed(random_seed)
        self.W= npr.normal(loc = 0.0, scale = 1e0, size=(10,10))
        self.y = npr.normal(loc = 0.0, scale = 1e0, size=(10,1))
        self.theta = np.linalg.inv(self.W) @ self.y 
        # calculated value of theta that minimizes loss
        # approximate value given by optimizers must be very close, element-wise
        
    def __call__(self, theta):
        return np.mean(np.sum((self.W @ theta - self.y)**2, axis=1))
    
    def grad(self, theta):
        return grad(self.__call__)(theta)


class QuadraticLoss(Loss):
    
    def __init__(self, random_seed):
        super().__init__()
        
        npr.seed(random_seed)
        self.W= npr.normal(loc = 0.0, scale = 1e0, size=(10,10))
        self.y = npr.normal(loc = 0.0, scale = 1e0, size=(10,1))
        
    def call(self, theta, required = None):
        return tf.math.reduce_mean(tf.math.reduce_sum((self.W@theta-self.y)**2, axis=1))
    
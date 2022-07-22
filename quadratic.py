import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

def quadratic_data(seed, batch_size=128, dim=10):
    """
    seed: an integer seed using which the function will return a batch of 128 parameters
    batch_size: dimension 1 of data
    dim: dimension of W,y,theta. W is square and y,theta are columns
    """
    npr.seed(seed)
    w = npr.normal(loc=0.0, scale=1e0, size=(batch_size,dim,dim))
    y = npr.normal(loc=0.0, scale=1e0, size=(batch_size,dim,1))
    theta = npr.normal(loc=0.0, scale=1e0, size=(batch_size,dim,1))
    return w, y, theta

def quadratic_task(optimizer, steps=150, lr = 0.05, seeds = range(100)): 
    """
    the optimizer is tasked with optimizing a batch of 128 functions at one go.
    optimizer: adam, sgd, rmsprop
    steps: iterations of the optimizer
    lr: learning rate(fixed)
    seeds: random seeds for sampling 128 functions, [0.....99] default.
    """
    
    losses = np.zeros((len(seeds), steps))
    
    def f(theta):
        return np.mean(np.sum((w@theta-y)**2, axis=1))
    
    def grad_func(theta, iter=0):
        return grad(f)(theta)
        
    def callback(params, iter, gradient):
        losses[i,iter] = f(params)
    
    for i,seed in enumerate(seeds):
        
        w, y, theta = quadratic_data(seed)
        theta = optimizer(grad_func, 
                  theta,
                  step_size=lr, 
                  num_iters=steps, 
                  callback=callback)
    
    return np.mean(losses, axis=0)

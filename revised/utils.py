import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam, sgd, rmsprop


plt.rcParams['figure.figsize'] = (17,7)
plt.style.use('default')


def Fit_learned(model, quadratic_class, loss_class, lr = 0.003, num_diff_functions = 100, optimize_each_for_steps = 100, unrolls = 20):
    
    states = None
    meta_optimizer = tf.keras.optimizers.Adam(0.005)
    
    for func in tqdm(range(num_diff_functions), 'Number of Functions Optimized'):
        
        Quadratic = quadratic_class(func)
        loss = loss_class(func) # initialize the loss object
        
        theta =tf.random.normal(shape=(10,1))
        optimizee_gradients = Quadratic.grad(theta.numpy())
        
        
        sum_losses = tf.zeros(())
        for steps in range(optimize_each_for_steps):
            
            with tf.GradientTape() as tape:
                    update, states = model(optimizee_gradients, states)
                    theta += update
                    sum_losses += loss(theta, None)
                    
            optimizee_gradients = Quadratic.grad(theta.numpy())
            
            if (steps+1) % unrolls == 0:
                lstm_grads = tape.gradient(sum_losses, model.variables)
                meta_optimizer.apply_gradients(zip(lstm_grads, model.variables))
                sum_losses = tf.zeros(())


def Evaluate_learned(model, quadratic_class, seed, num_diff_functions = 100, optimize_each_for_steps = 100, should_plot = False):
    
    states = None

    loss_progression = np.zeros([num_diff_functions, optimize_each_for_steps])

    for func in tqdm(range(num_diff_functions), 'Number of Functions Evaluated on'):
        
        Quadratic = quadratic_class(seed + func)

        theta = tf.random.normal(shape=(10,1))
        
        optimizee_gradients = Quadratic.grad(theta.numpy())

        for steps in range(optimize_each_for_steps):
            
            loss_progression[func, steps] = Quadratic(theta)
        
            update, states = model(optimizee_gradients, states)
            
            theta += update
                    
            optimizee_gradients = Quadratic.grad(theta.numpy())
                    
                
    
    mean_losses = np.mean(loss_progression, axis = 0)
    
    if should_plot:
        
        plt.style.use('default')
        fig, ax = plt.subplots(figsize = (7,4))
        ax.semilogy(mean_losses)
        ax.grid()
        ax.set_xlabel("Iterations")
        ax.set_ylabel(f"Average Loss of {num_diff_functions} functions")
        fig.suptitle("Mean Losses vs Iterations");
        
    return mean_losses



def Solve_learned(model, Quad_object, iterations = 100, theta = None):
    '''
    takes one instance of quadratic class and 'Solves' that optimization problem.
    returns the found_optimal and losses per epoch as a python list
    '''
    
    if theta is None:
        theta = tf.random.normal(shape=(10,1))
        
    losses = []
    optimizee_gradients = Quad_object.grad(theta.numpy())
    states = None
    
    for _ in tqdm(range(iterations), 'Iterations'):
        
        losses.append(Quad_object(theta))
        update, states = model(optimizee_gradients, states) 
        theta += update
        optimizee_gradients = Quad_object.grad(theta.numpy())
          
    return theta, losses


def Evaluate_others(quadratic_class, seed, lr = 0.003, num_diff_functions = 100, optimize_each_for_steps = 100, should_plot = False):
    
    optimizers = [sgd, adam, rmsprop]
    
    loss_optimizers = np.zeros([len(optimizers), optimize_each_for_steps])
    loss_progression = np.zeros([num_diff_functions, optimize_each_for_steps])
    
    if should_plot:
            fig, ax = plt.subplots(figsize = (8,5))
            fig.suptitle(f"Loss vs Iterations for Adam, SGD, RMSProp, LR: {lr}")
            ax.grid()
            ax.set_xlabel("Iterations")
            ax.set_ylabel(f"Average Loss of {num_diff_functions} functions")

        
    for i in tqdm(range(len(optimizers)), "Optimizers"):
        
        optimizer = optimizers[i]
        
        for func in range(num_diff_functions):
            
            theta = theta =tf.random.normal(shape=(10,1)).numpy()
            q = quadratic_class(func+seed)
            
            def grad_func(theta, iter=0):
                return q.grad(theta)
            
            def append_loss(params, iter, gradient):
                loss_progression[func, iter] = q(params)
        
            theta = optimizer(grad_func, 
                              theta,
                              step_size = lr,
                              num_iters = optimize_each_for_steps, 
                              callback = append_loss)
            
        loss_optimizers[i, :] = np.mean(loss_progression, axis = 0)
        
        if should_plot:
            ax.semilogy(loss_optimizers[i, :])
            
    if should_plot:
        ax.legend(['sgd', 'adam', 'rmsprop'])
    
    return loss_optimizers

def Evaluate(model, Quadratic_class, random_seed, numFunctions, stepsOfOptimization, lr_for_other, should_plot = True):
    
    loss_optimizers = Evaluate_others(Quadratic_class, 
                                      seed = random_seed, 
                                      lr = lr_for_other, 
                                      num_diff_functions = numFunctions, 
                                      should_plot = False, 
                                      optimize_each_for_steps = stepsOfOptimization)
    
    lstm_losses = Evaluate_learned(model, 
                                   Quadratic_class, 
                                   seed = random_seed, 
                                   num_diff_functions = numFunctions, 
                                   should_plot = False, 
                                   optimize_each_for_steps = stepsOfOptimization)
    
    fig, ax = plt.subplots(figsize = (10, 7))
    
    ax.grid()
    ax.set_xlabel("Iterations")
    ax.set_ylabel(f"Average Loss of {numFunctions} functions")
        
    fig.suptitle("Adam vs. SGD vs. RMS Prop. vs. L2L")
    
    for loss in loss_optimizers:
        ax.semilogy(loss)
        
    ax.semilogy(lstm_losses)
    ax.legend(['sgd', 'adam', 'rmsprop', 'lstm'])
    plt.show()
    
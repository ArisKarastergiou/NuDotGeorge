#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse
import numpy as np
from matplotlib import pyplot as plt
import george
from george import kernels
import emcee
import scipy.optimize as op
import os
from os.path import basename
import corner

# In[2]:
# Define all functions

def get_gp(profile, t):
# Define the objective function (negative log-likelihood in this case).
    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y, quiet=True)

    kernel = 1.0 * kernels.ExpSquaredKernel(metric=10**3)
    y = profile
    gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
               white_noise=np.log(1**2), fit_white_noise=True)
# You need to compute the GP once before starting the optimization.
    gp.compute(t)

# Print the initial ln-likelihood.
    print(gp.log_likelihood(y))

# Run the optimization routine.
    p0 = gp.get_parameter_vector()
    results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")

# Update the kernel and print the final log-likelihood.
    gp.set_parameter_vector(results.x)
    return gp
# This next function is using the global variable residuals
def lnprob(p):
    # Trivial uniform prior.
    if np.any((-100 > p[1:]) + (p[1:] > 100)):
        return -np.inf
    
    # Update the kernel and compute the lnlikelihood.
    gp1.set_parameter_vector(p)
    return gp1.lnlikelihood(ydata, quiet=True)

                    

# In[5]:

# Read command line arguments
#------------------------------
parser = argparse.ArgumentParser(description='Fit a squared exponential model to a single 1D timeseries')
parser.add_argument('-f','--filename', help='File containing residuals', required=True)
#------------------------------
args = parser.parse_args()
filename = args.filename
samples = 50
# Load the data
ydata = np.loadtxt(filename, comments='#')
xdata = np.arange(len(ydata))
plt.figure()
plt.plot(xdata,ydata)
plt.xlabel('x')
plt.ylabel('y')
                                                                            
# Drive a simple GP
gp1 = get_gp(ydata, xdata)
print(gp1.log_likelihood(ydata), gp1.get_parameter_names(),gp1.get_parameter_vector())
noise = gp1.get_parameter_vector()[1]
print 'RMS = ', np.sqrt(np.exp(noise))
mumc, dummy = gp1.predict(ydata, xdata)
plt.plot(xdata, mumc)
plt.show()
# Set up the sampler.
nwalkers, ndim = 36, len(gp1)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

# Initialize the walkers.
p0 = gp1.get_parameter_vector() + 1e-2 * np.random.randn(nwalkers, ndim)
print("Running burn-in")
p0, _, _ = sampler.run_mcmc(p0, 50)

print("Running production chain")
sampler.run_mcmc(p0, 1000);
cornerplotdata = sampler.chain.reshape((-1,ndim))
fig=corner.corner(cornerplotdata,labels=(('logMean','logSigma','logAmplitude','logScale')))
plt.show()

#!/usr/bin/env python
# coding: utf-8

# In[1]:
import matplotlib.gridspec as gridspec
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

    k1 = 1e-5 * kernels.ExpSquaredKernel(metric=10.0)
    k2 = 1.0 * kernels.ExpSquaredKernel(metric=10)
    kernel = k1 #+ k2
    y = profile
    gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
               white_noise=np.log(1e-5), fit_white_noise=True)
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
#    scale = np.sqrt(np.exp(p[3]))
#    print scale
# Trivial uniform prior, but scale no less than 100 days
    if np.any((7.0 > p[3:]) + (p[1:] > 14.0)):# or scale < 1.0:
        return -np.inf

    
# Update the kernel and compute the lnlikelihood.
    gp1.set_parameter_vector(p)
    return gp1.lnlikelihood(residuals, quiet=True)



# In[5]:

# Read command line arguments
#------------------------------
parser = argparse.ArgumentParser(description='Pulsar nudot variability studies using GP\
s with George')
parser.add_argument('-f','--filename', help='File containing residuals', required=True)
parser.add_argument('-e','--parfile', help='ephemeris for nudot', required=True)
parser.add_argument('-p','--pulsar', help='Pulsar name', required=True)
parser.add_argument('-d','--diagnosticplots', help='make image plots', action='store_true',required = False)
#------------------------------
samples = 100
args = parser.parse_args()
filename = args.filename
parfile = args.parfile
filebase = basename(filename)
outfile = os.path.splitext(filebase)[0]
datfile = outfile + '.dat'
pulsar = args.pulsar

if not (os.path.exists('./{0}/'.format(pulsar))):
        os.mkdir('./{0}/'.format(pulsar))
        

# Load the data
data = np.loadtxt(filename, comments='#')
residuals = data[:,1]
dates = data[:,0]-data[0,0]
plt.figure()
plt.title(pulsar)
plt.plot(dates,residuals)
plt.xlabel('day')
plt.ylabel('residual [sec]')
plt.savefig(pulsar+'/Residuals.png')

# Load the parfile
# epoch and nudot
q = open(parfile)
for line in q:
    if line.startswith('F0'):
        f0_line = line.split()
        period = 1/float(f0_line[1])
    if line.startswith('F1'):
        f1_line = line.split()
        f1 = float(f1_line[1])
    if line.startswith('PEPOCH'):
        pepoch_line = line.split()
        epoch = float(pepoch_line[1])
q.close()
print 'The period and nudot from the ephemeris are:', period, f1
                                                                            
# Drive a simple GP
gp1 = get_gp(residuals,dates)
print(gp1.log_likelihood(residuals), gp1.get_parameter_names(),gp1.get_parameter_vector())
noise = gp1.get_parameter_vector()[1]

# In[13]:
# Set up the kernal of the double derivative with some dummy parameters

k1prime = 100 * kernels.ExpSquaredDoublePrimeKernel(10.0)
k2prime = 100 * kernels.ExpSquaredDoublePrimeKernel(10.0)
kernelprime = k1prime #+ k2prime

# Set up the sampler.
nwalkers, ndim = 36, len(gp1)
print 'GP length', len(gp1)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

# Initialize the walkers.
p0 = gp1.get_parameter_vector() + 1e-2 * np.random.randn(nwalkers, ndim)
print("Running burn-in")
runin_steps = 200
p0, _, _ = sampler.run_mcmc(p0, runin_steps)
sampler.reset()
print("Running production chain")
sampler_steps = 2000
sampler.run_mcmc(p0, sampler_steps)
labels=(('logMean','logSigma','logAmplitude','logScale'))
cornerplotdata = sampler.chain.reshape((-1,ndim))
fig=corner.corner(cornerplotdata,labels=labels)
plt.title(pulsar)
fig.savefig(pulsar+'/cornerplot.png')
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

# extract MCMC chains
samples_MC = sampler.chain
Lprob = sampler.lnprobability
print 'Lprob shape:',Lprob.shape
maxLprob = np.max(Lprob)
print 'maximum Lprob is: ', maxLprob
# Plot the chains
gs1 = gridspec.GridSpec(ndim+1,1) # ndim is number of parameters
gs1.update(left=0.1, right=0.98, bottom = 0.07, top = 0.98, hspace=0)
ax1 = plt.subplot(gs1[0,0])
plt.setp(ax1.get_xticklabels(), visible=False)
plt.title(pulsar)
plt.plot(Lprob.T, 'k-', alpha = 0.2)
plt.ylabel(r'$\ln P$')
for i in range(ndim):
    axc = plt.subplot(gs1[i+1,0], sharex = ax1)
    if i < (ndim):
        plt.setp(axc.get_xticklabels(), visible=False)
    plt.plot(samples_MC[:,:,i].T, 'k-', alpha = 0.2)
    plt.ylabel(labels[i]) # labels is a list of parameter names
    plt.xlim(0,sampler_steps+runin_steps)
    plt.xlabel('iteration number')
fig.savefig(pulsar+'/imperialwalkers.png')

# Produce samples from the MC
print 'Sample chain:', sampler.chain.shape

plt.figure()
big_array = np.zeros((samples,len(dates)))
residual_array = np.zeros((samples,len(dates)))
np.seterr(all='ignore')
for i in range(samples):
    # Choose a random walker and step.
    prob_test = 0 
    while prob_test < 1:
        w = np.random.randint(sampler.chain.shape[0])
        n = np.random.randint(sampler.chain.shape[1])
        if Lprob[w,n] > maxLprob - 5.0:
            prob_test = 1
    print i
    print 'current hyperparameters and lnprob:', sampler.chain[w,n], Lprob[w,n]
    newnoise=sampler.chain[w,n][1]
    newvector=sampler.chain[w,n][2:]
    newvector[1] = np.exp(newvector[1])
    #newvector[3] = np.exp(newvector[3])
    kernelprime.set_parameter_vector(newvector)
    gp1.set_parameter('mean:value',0.0)
    gp1.set_parameter('white_noise:value',newnoise) 
    mumc, dummy = gp1.predict(residuals, dates, kernel = kernelprime )
    resid, dummy = gp1.predict(residuals, dates)
    big_array[i] = f1 - mumc/period/86400.**2
    residual_array[i] = residuals - resid
    plt.plot(dates,big_array[i], "g", alpha=0.1)
    #plt.ylim((-1.4e-14,-0.6e-14))
plt.xlabel('day')
plt.ylabel('nudot')
axes = plt.gca()
ymin, ymax = axes.get_ylim()
plt.title(pulsar)
plt.savefig(pulsar+'/probabilityNuDot.png')

# In[15]:
# find the mean and stdev from the array

medians = np.median(big_array,0)
sigmas = np.std(big_array,0)

median_residuals = np.median(residual_array,0)
out = np.empty_like(data)
out[:,0] = dates
out[:,1] = median_residuals
np.savetxt(pulsar+'/residuals.dat', out)
out[:,0] = medians
out[:,1] = sigmas
np.savetxt(pulsar+'/nudots.dat', out)

# In[16]:


#plt.plot(dates,means)
plt.figure()
plt.errorbar(dates,medians,yerr=sigmas, fmt='o')
#plt.ylim((-1.5e-14,-0.5e-14))
plt.xlabel('day')
plt.ylabel('nudot')
axes = plt.gca()
axes.set_ylim(ymin,ymax)
plt.title(pulsar)
plt.savefig(pulsar+'/errorsNuDot.png')


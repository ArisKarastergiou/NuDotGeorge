#!/usr/bin/env python
# coding: utf-8

import argparse
import bilby
import george
import os

import numpy as np
import matplotlib.gridspec as gridspec
import scipy.optimize as op

from george import kernels
from matplotlib import pyplot as plt
from os.path import basename

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


def drive_gp(profile, t, logamplitude, scale, lognoise):
    kernel = np.exp(logamplitude) * kernels.ExpSquaredKernel(metric=scale)
    y = profile
    gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
               white_noise=lognoise, fit_white_noise=True)
# You need to compute the GP once before starting the optimization.
    gp.compute(t)
    return gp


def constant_function(x, a):
    return np.ones(len(x)) * a


def linear_function(x, a, b):
    return a * x + b


# Read command line arguments
#------------------------------
parser = argparse.ArgumentParser(description='Pulsar nudot variability studies using GP\
s with George')
parser.add_argument('-f','--filename', help='File containing residuals', required=True)
parser.add_argument('-e','--parfile', help='ephemeris for nudot', required=True)
parser.add_argument('-p','--pulsar', help='Pulsar name', required=True)
# parser.add_argument('-d','--diagnosticplots', help='make image plots', action='store_true',required = False)
#------------------------------
samples = 100
args = parser.parse_args()
filename = args.filename
parfile = args.parfile
filebase = basename(filename)
outfile = os.path.splitext(filebase)[0]
datfile = outfile + '.dat'
pulsar = args.pulsar

#if not (os.path.exists('./{0}/'.format(pulsar))):
#        os.mkdir('./{0}/'.format(pulsar))
        

# Load the data
data = np.genfromtxt(filename, delimiter=" ")
residuals = data[:,1]
errors = data[:,2] * 1e-6
dates = data[:,0]-data[0,0]
#plt.figure()
#plt.title(pulsar)
#plt.plot(dates, residuals, ".")
#plt.xlabel('day')
#plt.ylabel('residual [sec]')
#plt.savefig('./outdir/'+pulsar+'_Residuals.png')
##plt.show()
#plt.close()

# Load the parfile to extract epoch and nudot
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
print("The period and nudot from the ephemeris are:", period, f1)

# Set up the kernal of the double derivative with some dummy parameters
print("Setting up kernel")
k1 = 1e-5 * kernels.ExpSquaredKernel(10.0)
k2 = 1.0 * kernels.ExpSquaredKernel(10.0)
ktot = k1 #+ k2

ConstantMeanModel = bilby.core.likelihood.function_to_george_mean_model(constant_function)
mean_model = ConstantMeanModel(a=0)

likelihood = bilby.core.likelihood.GeorgeLikelihood(kernel=ktot, mean_model=mean_model, t=dates, y=residuals, yerr=errors)

priors = bilby.core.prior.PriorDict()
priors["mean:a"] = bilby.core.prior.Uniform(-20, 20, name="a", latex_label=r"$a$")
priors["kernel:k1:log_constant"] = bilby.core.prior.Uniform(-20, 20, name="log_A", latex_label=r"$\ln A$")
priors["kernel:k2:metric:log_M_0_0"] = bilby.core.prior.Uniform(-20, 20, name="log_M_0_0", latex_label=r"$\ln M_{00}$")
priors["white_noise:value"] = bilby.core.prior.Uniform(-20, 20, name="sigma", latex_label=r"$\sigma$")

if os.path.exists("outdir/{0}_result.json".format(pulsar)) == False:
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        outdir="./outdir",
        label=pulsar,
        npool=16,
        sampler="dynesty",
        sample="rslice",
        nlive=1024,
        evidence_tolerance=0.1,
        # importance_nested_sampling=True,
        resume=True,
    )

    result.plot_corner(dpi=150)
else:
    result = bilby.core.result.read_in_result("outdir/{0}_result.json".format(pulsar))

# extract samples
log_A_posts = result.posterior["kernel:k1:log_constant"].values
log_M00_posts = result.posterior["kernel:k2:metric:log_M_0_0"].values
white_noise = result.posterior["white_noise:value"].values

np.random.seed(0)
v1 = np.random.choice(log_A_posts,1000)
v2 = np.random.choice(log_M00_posts,1000)
v3 = np.random.choice(white_noise,1000)


nudot_arr = np.zeros((1000, len(dates)))
#ml_gp = drive_gp(residuals, dates, np.exp(v1), np.exp(v2), np.exp(v3))
ml_gp = drive_gp(residuals, dates, v1[0], np.exp(v2[0]), v3[0])
kernel_params = ml_gp.get_parameter_vector()
#repeat the same random numbers
for i in range(0, 1000):

    #mean (cannot set this way but just illustrating)    
        #kernel_params[0] = 0.0
    #log noise (also cannot set this way)
        #kernel_params[1] = v3
    # log amplitude
    kernel_params[2] = v1[i]
    # metric
    kernel_params[3] = np.exp(v2[i])

    # set up dderivative kernel with dummy parameters
    kerneldprime = 100 * kernels.ExpSquaredDoublePrimeKernel(10.0)
    # Set metric and amplitude for dderivative kernel
    kerneldprime.set_parameter_vector(kernel_params[2:])
    # Set mean to 0 and noise to v3
    ml_gp.set_parameter('mean:value',0.0)
    ml_gp.set_parameter('white_noise:value', v3[i])

    ml_mu, _ = ml_gp.predict(residuals, dates, return_var=True, kernel=kerneldprime)
    nudot_arr[i,:] = f1 - ml_mu/period/86400.**2

med = np.percentile(nudot_arr, 50., axis=0)
low = med - np.percentile(nudot_arr, 16., axis=0)
upp = np.percentile(nudot_arr, 84., axis=0) - med

median_of_medians = np.median(med)
logscale = np.floor(np.log10(np.abs(median_of_medians)))

mjds = data[:,0]

plt.errorbar(mjds, med*(10**(-logscale)), yerr=[upp*(10**(-logscale)), low*(10**(-logscale))], fmt=".", color="k")
plt.title(pulsar)
plt.xlabel("Time (MJD)")
plt.ylabel(r"$\dot{\nu} (\times 10^{%i})$ (Hz$^{-2}$)" %logscale)
plt.tight_layout()
plt.savefig("./outdir/{0}_nudot_timeseries.png".format(args.pulsar), dpi=200)
#plt.show()
#plt.close()

np.savetxt("./outdir/{0}_nudot.txt".format(args.pulsar), np.c_[mjds, med, low, upp].T)

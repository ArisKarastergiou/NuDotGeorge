#!/usr/bin/env python
# coding: utf-8

import argparse
import bilby
import george
from george.utils import multivariate_gaussian_samples
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
    kernel = k1 
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


def drive_gp(profile, t, logamplitude, logscale, lognoise):
    kernel = np.exp(logamplitude) * kernels.ExpSquaredKernel(metric=np.exp(logscale))
    print("AK: drive_gp: ", kernel.get_parameter_vector())
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
mjds = data[:,0]
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
#k2 = 1.0 * kernels.ExpSquaredKernel(10.0)
#ktot = k1 #+ k2

ConstantMeanModel = bilby.core.likelihood.function_to_george_mean_model(constant_function)
mean_model = ConstantMeanModel(a=0)

likelihood = bilby.core.likelihood.GeorgeLikelihood(kernel=k1, mean_model=mean_model, t=dates, y=residuals, yerr=errors)

priors = bilby.core.prior.PriorDict()
print("priors: ", priors)
priors["mean:a"] = bilby.core.prior.Uniform(-1., 1., name="a", latex_label=r"$a$")
priors["kernel:k1:log_constant"] = bilby.core.prior.Uniform(-20, 20, name="log_A", latex_label=r"$\ln A$")
priors["kernel:k2:metric:log_M_0_0"] = bilby.core.prior.Uniform(0, 20, name="log_M_0_0", latex_label=r"$\ln M_{00}$")
priors["white_noise:value"] = bilby.core.prior.Uniform(-25, -15, name="sigma", latex_label=r"$\sigma$")

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
mean_posts = result.posterior["mean:a"].values
log_A_posts = result.posterior["kernel:k1:log_constant"].values
log_M00_posts = result.posterior["kernel:k2:metric:log_M_0_0"].values
white_noise = result.posterior["white_noise:value"].values

# create array of gp parameter vectors (scale mean appropriately for nudot calculation)
#newvector_all = np.column_stack((mean_posts/86400.**2, white_noise-2.*np.log(86400.), log_A_posts-2.*np.log(86400.), log_M00_posts))
#newvector_all = np.column_stack((mean_posts, white_noise, log_A_posts, log_M00_posts))
newvector_all = np.column_stack((np.zeros(white_noise.shape), white_noise, log_A_posts, log_M00_posts))

np.random.seed(0)
idx = np.random.randint(newvector_all.shape[0], size=1000)

#pmean = np.random.choice(mean_posts/period/86400.**2,1000) 
#pmean = np.zeros(1000)
#logA = np.random.choice(log_A_posts,1000)
#logMetric = np.random.choice(log_M00_posts,1000)
#logNoise = np.random.choice(white_noise,1000)
#newvector = np.column_stack((pmean, logNoise, logA, logMetric))

newvector = newvector_all[idx,:]

nudot_arr = np.zeros((1000, len(dates)))
ml_gp = drive_gp(residuals, dates, newvector[0,2], newvector[0,3], newvector[0,1])
kernel_params = ml_gp.get_parameter_vector()
#kernel parameters are log AMP log metric
print("kernel params: ",k1.get_parameter_names())
print("kernel params: ",k1.get_parameter_vector())
print("gp kernel params: ",ml_gp.get_parameter_names())
#gp kernel parameters are log mean, log noise, log amp, metric (NOT LOG!!)
print("gp kernel params: ", kernel_params)
#repeat the same random numbers
for i in range(0, 1000):

    # Set gp parameters from samples
    #ml_gp.set_parameter('mean:value',0.0)
    #ml_gp.set_parameter('white_noise:value',logNoise[i])
    #ml_gp.set_parameter('kernel:k1:log_constant',logA[i])
    #ml_gp.set_parameter('kernel:k2:metric:log_M_0_0', logMetric[i])
    ml_gp.set_parameter_vector(newvector[i])
    kernel2prime = np.exp(newvector[i,2]) * kernels.ExpSquaredDoublePrimeKernel(np.exp(newvector[i,3]))
    kernel4prime = np.exp(newvector[i,2]) * kernels.ExpSquaredFourPrimeKernel(np.exp(newvector[i,3]))
    #ml_gp.compute(dates)
    #ml_mu = ml_gp.sample_conditional(residuals, dates, kernel=kernel2prime, k2=kernel4prime)
    ml_mu, cov, cov_minus = ml_gp.predict(residuals, dates, return_cov=True, kernel=kernel2prime, k2=kernel4prime)
    mycov = cov - cov_minus/period/86400.**2
    sample = multivariate_gaussian_samples(mycov, 1, mean=ml_mu)#/86400.**2)
    #print(sample.shape)
    #print(np.sqrt(np.diag(cov)))
    #print(ml_gp.get_parameter_vector(), kernel2prime.get_parameter_vector())
    nudot_arr[i,:] = f1 - sample/period/86400.**2

med = np.percentile(nudot_arr, 50., axis=0)
print(nudot_arr.shape,med.shape, np.std(nudot_arr[:,20]))
low = med - np.percentile(nudot_arr, 16., axis=0)
upp = np.percentile(nudot_arr, 84., axis=0) - med

median_of_medians = np.median(med)
logscale = np.floor(np.log10(np.abs(median_of_medians)))

#AK: metric to determine how well the nudot timeseries agree with a straight line.
yardstick = upp + low
parameter1 = (np.minimum(np.abs(med + upp - f1), np.abs(med - low - f1))/yardstick)**2
parameter2 = (np.minimum(np.abs(med + upp - median_of_medians), np.abs(med - low - median_of_medians))/yardstick)**2
metric1 = np.sqrt(np.sum(parameter1))
metric2 = np.sqrt(np.sum(parameter2))
with open('outdir/'+pulsar+'_nudot_flatness.txt', 'w') as f:
        f.write(pulsar+' '+str(metric1)+' '+str(metric2)+'\n')
        f.close()
###################
plt.figure()
plt.errorbar(mjds, med*(10**(-logscale)), yerr=[upp*(10**(-logscale)), low*(10**(-logscale))], fmt=".", color="k")
plt.title(pulsar)
plt.xlabel("Time (MJD)")
plt.ylabel(r"$\dot{\nu} (\times 10^{%i})$ (Hz$^{-2}$)" %logscale)
plt.tight_layout()
plt.savefig("./outdir/{0}_nudot_timeseries.png".format(args.pulsar), dpi=200)
#plt.show()
#plt.close()

np.savetxt("./outdir/{0}_nudot.txt".format(args.pulsar), np.c_[mjds, med, low, upp].T)

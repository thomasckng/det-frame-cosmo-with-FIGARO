import numpy as np
from figaro.utils import rejection_sampler
from figaro.cosmology import dVdz_approx_planck18, Planck18 as Omega
import os
import dill
import paths

np.random.seed(42)

# Mass distribution
from population_models.mass import plpeak # from https://github.com/thomasckng/pop_models_lvk

# Redshift distribution
def p_z(z):
    return dVdz_approx_planck18(z)/(1+z)

# Generate true detector frame mass samples
def generate_truth(n_single_event_draws):
    m = rejection_sampler(n_single_event_draws, plpeak, [0,200])
    z = rejection_sampler(n_single_event_draws, p_z, [0, 2])
    return np.array([m * (1 + z), z]).T

# Generate detector frame mass posterior samples
def generate_mz_posterior_samples(truth, sigma = 0.03, n_samples = 1000):
    log_obs_mean = np.log(truth) + np.random.normal(0, sigma, truth.shape) # shape = (n_events)
    return np.exp(np.random.normal(log_obs_mean, sigma, (n_samples, len(truth)))).T # shape = (n_events, n_samples)

# Load selection function
with open(paths.data/'selection_function.pkl', 'rb') as f: # selection_function.pkl is generated by generate_selection_function.py
    selfunc_interp = dill.load(f)
def selection_function(x):
    return selfunc_interp(x)

n_single_event_draws = 20_000
outdir = 'simulation'
if not os.path.exists(paths.data / outdir):
    os.makedirs(paths.data / outdir)
outdir = paths.data / outdir

samples = generate_truth(n_single_event_draws) # shape = (n_events, n_params)
samples[:,1] = Omega.LuminosityDistance(samples[:,1])
samples_single_event = samples[np.random.uniform(0,1,samples.shape[0]) <= selection_function(samples)] # shape = (n_events, n_params)
posterior = generate_mz_posterior_samples(samples_single_event[:,0]) # shape = (n_events, n_samples)

np.savetxt(outdir / 'true_samples.txt', samples[:,0])
np.savetxt(outdir / 'obs_samples.txt', samples_single_event[:,0])
np.savetxt(outdir / 'jsd_bounds.txt', [posterior.min(), posterior.max()])

if not os.path.exists(outdir / 'data'):
    os.makedirs(outdir / 'data')
for i,p in enumerate(posterior):
    np.savetxt(outdir / f'data/posterior_samples_{i+1}.txt', p)

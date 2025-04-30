import os
import paths
import numpy as np
from population_models.mass import powerlaw_smoothed as pl_mass
from population_models.redshift import powerlaw_redshift as pl_z
from scipy.stats import norm
from galleon.generator import Generator

np.random.seed(42)

simulated_truth = {'alpha': 3.5,
                   'mu': 35,
                   'sigma': 5,
                   'delta': 5,
                   'w': 0.2,
                   'm_min': 5,
                   'm_max': 90,
                   'kappa': 0}

# Mass distribution
def gaussian(x):
    return norm(simulated_truth['mu'], simulated_truth['sigma']).pdf(x)
def density_m(m):
    return (1-simulated_truth['w'])*pl_mass(m, simulated_truth['alpha'], simulated_truth['m_max'], simulated_truth['m_min'], simulated_truth['delta']) + simulated_truth['w']*gaussian(m)

# Redshift distribution
def density_z(z):
    return pl_z(z, simulated_truth['kappa'])

class Sampler(Generator):
    def mass_distribution(self, m):
        return density_m(m)
    def redshift_distribution(self, z):
        return density_z(z)

outdir = 'simulation'
if not os.path.exists(paths.data / outdir):
    os.makedirs(paths.data / outdir)
outdir = paths.data / outdir

Sampler().generate_mock_catalogue(300, 3000, 100_000, id = '10', out_folder = outdir)

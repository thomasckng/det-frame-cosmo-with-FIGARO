from figaro.load import load_density
from figaro.plot import plot_median_cr
from figaro import plot_settings
import numpy as np
import paths
from figaro.cosmology import Planck15
from figaro.cosmology import CosmologicalParameters
from population_models.redshift import powerlaw
from population_models.mass import powerlaw_smoothed as pl_mass
from scipy.stats import norm

label = 'simulation'
outdir = paths.data / label

simulated_truth = {'$H_0$': Planck15.h*100, '$\\alpha$': 3.5, '$\\mu$': 35, '$\\sigma$': 5, '$\\delta$': 5, '$w$': 0.2, '$m_\mathrm{min}$': 5, '$m_\mathrm{max}$': 90, '$\\kappa$': 0}

# Mass distribution
def gaussian(x):
    return norm(simulated_truth['mu'], simulated_truth['sigma']).pdf(x)
def density_m(m):
    return (1-simulated_truth['w'])*pl_mass(m, simulated_truth['alpha'], simulated_truth['m_max'], simulated_truth['m_min'], simulated_truth['delta']) + simulated_truth['w']*gaussian(m)

# Redshift distribution
zmax   = 2.3 # LVK paper
z_norm = np.linspace(0,zmax,1000)
dz     = z_norm[1]-z_norm[0]
def _unnorm_powerlaw_redshift(z, k, H0):
    reg_const = (1+zmax)/CosmologicalParameters(H0/100, 0.3065, 0.6935, -1).ComovingVolumeElement(zmax)
    return powerlaw(z, k)*CosmologicalParameters(H0/100, 0.3065, 0.6935, -1).ComovingVolumeElement(z)/(1+z) * reg_const
def powerlaw_redshift(z, k, H0):
    norm = np.sum(_unnorm_powerlaw_redshift(z_norm,k,H0)*dz)
    return _unnorm_powerlaw_redshift(z, k, H0)/norm
def density_z(z, H0):
    return powerlaw_redshift(z, simulated_truth['kappa'], H0)

draws = load_density(outdir / f'draws/draws_observed_{label}.json')
bounds = np.loadtxt(outdir / "jsd_bounds.txt")

mz = np.linspace(1,200,1000)
z = np.linspace(0.001,2,900)
m = np.einsum("i, j -> ij", mz, np.reciprocal(1+z)) # shape = (len(mz), len(z))

fig = plot_median_cr(draws, hierarchical=True, save=True, show=False, median_label='$p(m^z_1|\mathbf{\Theta})$')
ax = fig.axes[0]
colors = ['tab:green', 'tab:red', 'tab:orange']
for i, c in zip([40, 70, 100], colors):
    model_pdf = np.einsum("ij, j -> ij", density_m(m), density_z(z, i)) # shape = (len(mz), len(z))
    model_pdf = np.trapz(model_pdf, z, axis=1) # shape = (len(mz))
    ax.plot(mz, model_pdf/np.trapz(model_pdf, m), label=f"$H_0={i}$", c=c)
ax.axvspan(0, bounds[0], color='gray', alpha=0.2)
ax.axvspan(bounds[1], 200, color='gray', alpha=0.2)
ax.set_xlabel('$m^z_1\ [\mathrm{M}_\odot]$')
ax.set_ylabel('$\mathrm{Density}$')
ax.set_xlim(0, 200)
ax.set_ylim(0)
ax.legend()
fig.savefig(paths.figures / "simulation_comparison.pdf", bbox_inches='tight')

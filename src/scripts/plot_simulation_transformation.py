from matplotlib import pyplot as plt
import numpy as np
from figaro import plot_settings
import paths
from figaro.cosmology import Planck15
from figaro.cosmology import CosmologicalParameters
from population_models.redshift import powerlaw
from population_models.mass import powerlaw_smoothed as pl_mass
from scipy.stats import norm

simulated_truth = {'H_0': Planck15.h*100, 'alpha': 3.5, 'mu': 35, 'sigma': 5, 'delta': 5, 'w': 0.2, 'm_min': 5, 'm_max': 90, 'kappa': 0}

# Mass distribution
def gaussian(x):
    return norm(simulated_truth['mu'], simulated_truth['sigma']).pdf(x)
def density_m(m):
    return (1-simulated_truth['w'])*pl_mass(m, simulated_truth['alpha'], simulated_truth['m_max'], simulated_truth['m_min'], simulated_truth['delta']) + simulated_truth['w']*gaussian(m)

# Redshift distribution
zmax   = 2.3 # LVK paper
z_norm = np.linspace(0,zmax,1000)
dz     = z_norm[1]-z_norm[0]
def _unnorm_powerlaw_redshift(z, k, Omega):
    reg_const = (1+zmax)/Omega.ComovingVolumeElement(zmax)
    return powerlaw(z, k)*Omega.ComovingVolumeElement(z)/(1+z) * reg_const
def powerlaw_redshift(z, k, Omega):
    norm = np.sum(_unnorm_powerlaw_redshift(z_norm,k,Omega)*dz)
    return _unnorm_powerlaw_redshift(z, k, Omega)/norm
def density_z(z, H0):
    Omega = CosmologicalParameters(H0/100, 0.3065, 0.6935, -1)
    return powerlaw_redshift(z, simulated_truth['kappa'], Omega)


fig, (ax1, ax2) = plt.subplots(2, figsize=(7,6), gridspec_kw={'hspace': 0.3})
m = np.linspace(0,200,1000)
ax1.plot(m, density_m(m)/np.trapz(density_m(m), m), label = "$\mathrm{PL}+\mathrm{Peak}$", c = 'tab:blue')
ax1.set_xlabel('$m_1\ [\mathrm{M}_\odot]$')
ax1.set_ylabel('$p(m_1|\Lambda)$')
ax1.set_xlim(0,200)
ax1.set_ylim(0)
ax1.legend()

mz = np.linspace(1,200,1000)
z = np.linspace(0.001,2,1000)
m = np.einsum("i, j -> ij", mz, np.reciprocal(1+z)).flatten() # shape = (len(mz) * len(z))
pdf_m = density_m(m) # shape = (len(mz) * len(z))
colors = ['tab:green', 'tab:red', 'tab:orange']
for i, c in zip([40, 70, 100], colors):
    pdf_z = density_z(z, i)
    pdf_z = np.repeat(pdf_z / (1+z), len(mz)) # shape = (len(mz) * len(z))
    model_pdf = pdf_m * pdf_z # shape = (len(mz) * len(z))
    model_pdf = np.trapz(model_pdf.reshape(len(mz), len(z)), z, axis=1)
    model_pdf = model_pdf/np.trapz(model_pdf, mz)
    ax2.plot(mz, model_pdf, label=f"$H_0={i}$", c=c)
ax2.set_xlabel('$m^z_1\ [\mathrm{M}_\odot]$')
ax2.set_ylabel('$p(m^z_1|\Lambda,\Omega(H_0),\mathrm{det})$')
ax2.set_xlim(0,200)
ax2.set_ylim(0)
ax2.legend()
fig.savefig(paths.figures / "simulation_transformation.pdf", bbox_inches='tight')

from matplotlib import pyplot as plt
import numpy as np
from figaro.cosmology import dVdz_approx_planck18
from figaro import plot_settings
import paths

# Mass distribution
from population_models.mass import plpeak

# Redshift distribution
def p_z(z):
    return dVdz_approx_planck18(z)/(1+z)

m_grid = np.linspace(5, 50, 200)
z_grid = np.linspace(0.01, 2.0, 200)

# Compute 1D PDFs
plpeak_pdf = plpeak(m_grid)
pz_pdf = p_z(z_grid)
plpeak_pdf /= np.trapz(plpeak_pdf, m_grid)
pz_pdf /= np.trapz(pz_pdf, z_grid)

# Compute 2D uncorrelated PDF
def pdf_mz(m, z):
    return plpeak(m) * p_z(z)
M, Z = np.meshgrid(m_grid, z_grid, indexing='ij')
PDF_mz = pdf_mz(M, Z)
PDF_mz /= np.trapz(np.trapz(PDF_mz, z_grid, axis=1), m_grid)

fig = plt.figure(figsize=(7,7))
gs = fig.add_gridspec(2, 2, width_ratios=[4,1], height_ratios=[1,4], wspace=0.05, hspace=0.05)
ax_joint = fig.add_subplot(gs[1,0])
ax_marg_m = fig.add_subplot(gs[0,0], sharex=ax_joint)
ax_marg_z = fig.add_subplot(gs[1,1], sharey=ax_joint)

# 2D contour
c = ax_joint.contourf(m_grid, z_grid, PDF_mz.T, levels=np.logspace(-2, -1, 5), cmap='Blues')
ax_joint.set_xlabel(r'$m_1\ [M_\odot]$')
ax_joint.set_ylabel(r'$z$')

# 1D marginals
ax_marg_m.plot(m_grid, plpeak_pdf, color='k')
ax_marg_m.axis('off')
ax_marg_z.plot(pz_pdf, z_grid, color='k')
ax_marg_z.axis('off')

fig.savefig(paths.figures / 'simulation_before_transformation.pdf', bbox_inches='tight')

from matplotlib import pyplot as plt
import numpy as np
from figaro.cosmology import CosmologicalParameters
from figaro import plot_settings
import paths
import dill

# Mass distribution
from population_models.mass import plpeak

# Redshift distribution
def p_z(z, H0):
    Omega = CosmologicalParameters(H0/100., 0.315, 0.685, -1., 0., 0.)
    return Omega.ComovingVolumeElement(z)*(1+z)**(-2)/Omega.dDLdz(z)

# Load selection function
with open(paths.data / 'selection_function.pkl', 'rb') as f:
    selfunc_interp = dill.load(f)
def selection_function(grid):
    return selfunc_interp(grid)

# Set up grids
mz = np.linspace(1, 100, 200)
dL = np.linspace(10, 8000, 200)
H0_cases = [30, 70, 110]
colors = ['tab:blue', 'tab:orange', 'tab:green']

fig = plt.figure(figsize=(7,7))
gs = fig.add_gridspec(2, 2, width_ratios=[4,1], height_ratios=[1,4], wspace=0.05, hspace=0.05)
ax_joint = fig.add_subplot(gs[1,0])
ax_marg_mz = fig.add_subplot(gs[0,0], sharex=ax_joint)
ax_marg_dl = fig.add_subplot(gs[1,1], sharey=ax_joint)

for h0_val, color in zip(H0_cases, colors):
    # Calculate z from dL for this H0
    Omega = CosmologicalParameters(h0_val/100., 0.315, 0.685, -1., 0., 0.)
    z = Omega.Redshift(dL)  # shape = (len(dL),)
    m = np.einsum('i,j->ij', mz, 1/(1+z))  # shape = (len(mz), len(dL))
    model_pdf_m = plpeak(m)  # shape = (len(mz), len(dL))
    model_pdf_z = p_z(z, h0_val)  # shape = (len(dL),)
    pdf_mzdl = model_pdf_m * model_pdf_z  # shape = (len(mz), len(dL))
    # Apply selection effects
    grid = np.stack(np.meshgrid(mz, dL, indexing='ij'), axis=-1)  # shape = (len(mz), len(dL), 2)
    SE_grid = selection_function(grid)
    pdf_mzdl *= SE_grid
    # Normalize over mz and dL
    pdf_mzdl /= np.trapz(np.trapz(pdf_mzdl, dL, axis=1), mz)
    # 1D marginals
    pdf_mz = np.trapz(pdf_mzdl, dL, axis=1)
    pdf_dl = np.trapz(pdf_mzdl, mz, axis=0)
    pdf_mz /= np.trapz(pdf_mz, mz)
    pdf_dl /= np.trapz(pdf_dl, dL)
    # 2D contour
    c = ax_joint.contour(mz, dL, pdf_mzdl.T, levels=[1e-6, 5e-6], colors=color, linewidths=1, alpha=0.6)
    # 1D marginals
    ax_marg_mz.plot(mz, pdf_mz, color=color, label=f'$H_0={h0_val}$')
    ax_marg_dl.plot(pdf_dl, dL, color=color)

ax_joint.set_xlabel(r'$m^z_1\ [M_\odot]$')
ax_joint.set_ylabel(r'$d_L\ [\mathrm{Mpc}]$')
ax_marg_mz.axis('off')
ax_marg_dl.axis('off')
ax_marg_mz.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
fig.savefig(paths.figures / 'simulation_after_transformation.pdf', bbox_inches='tight')
plt.close(fig)

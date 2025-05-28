from figaro.load import load_density
from figaro.plot import plot_multidim
from figaro.cosmology import CosmologicalParameters
from figaro import plot_settings
import numpy as np
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

label = 'simulation'
outdir = paths.data / label

draws = load_density(outdir / f'draws/draws_observed_{label}.json')
obs_samples = np.loadtxt(outdir / 'obs_samples.txt')

bounds = np.loadtxt(outdir / 'jsd_bounds.txt')

fig = plot_multidim(draws, hierarchical=True, median_label='$p(m^z_1, d_L|\mathbf{\Theta})$', labels=['m^z_1', 'd_L'], units=['M_\odot', '\mathrm{Mpc}'], bounds=bounds)

# Set up grids
mz = np.linspace(bounds[0, 0], bounds[0, 1], 200)
dL = np.linspace(bounds[1, 0], bounds[1, 1], 200)
H0_cases = [30, 70, 110]
colors = ['tab:green', 'tab:red', 'tab:orange']

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
    c = fig.axes[2].contour(mz, dL, pdf_mzdl.T, levels=[1e-6, 5e-6], colors=color, linewidths=1, alpha=0.7)
    # 1D marginals
    fig.axes[0].plot(mz, pdf_mz, color=color, label=f'$H_0={h0_val}$')
    fig.axes[3].plot(dL, pdf_dl, color=color)

fig.axes[1].legend(*fig.axes[0].get_legend_handles_labels(), loc = 'center', fontsize=18)

fig.savefig(paths.figures / 'simulation_comparison.pdf', bbox_inches='tight')

from figaro.load import load_density
from figaro.plot import plot_multidim
from figaro import plot_settings
import numpy as np
from matplotlib import pyplot as plt
import paths

label = 'simulation'
outdir = paths.data / label

draws = load_density(outdir / f'draws/draws_observed_{label}.json')
obs_samples = np.loadtxt(outdir / 'obs_samples.txt')

fig = plot_multidim(draws, hierarchical=True, median_label='$p(m^z_1, d_L|\mathbf{\Theta})$', labels=['m^z_1', 'd_L'], units=['M_\odot', '\mathrm{Mpc}'], bounds=np.loadtxt(outdir / 'jsd_bounds.txt'))
fig.axes[0].hist(obs_samples[:, 0], bins = int(np.sqrt(len(obs_samples))), histtype = 'step', density = True, label = '$T_t$', color = 'red')
fig.axes[3].hist(obs_samples[:, 1], bins = int(np.sqrt(len(obs_samples))), histtype = 'step', density = True, color = 'red')

fig.axes[1].legend(*fig.axes[0].get_legend_handles_labels(), loc = 'center')

fig.savefig(paths.figures / 'simulation_reconstruction.pdf', bbox_inches='tight')

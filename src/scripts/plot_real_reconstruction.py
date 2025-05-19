from figaro.load import load_density
from figaro.plot import plot_multidim
from figaro import plot_settings
import numpy as np
from matplotlib import pyplot as plt
import paths

label = 'real'
outdir = paths.data / label

draws = load_density(outdir / f'draws/draws_observed_{label}.json')
samples_med = np.loadtxt(outdir / 'samples_med.txt')

fig = plot_multidim(draws, hierarchical=True, median_label='$p(m^z_1, d_L|\mathbf{\Theta})$', labels=['m^z_1', 'd_L'], units=['M_\odot', '\mathrm{Mpc}'], bounds=np.loadtxt(outdir / 'jsd_bounds.txt'))
fig.axes[0].hist(samples_med[:, 0], bins = int(np.sqrt(len(samples_med))), histtype = 'step', density = True, label = '$\mathrm{median}(Y_t)$', color = 'red')
fig.axes[3].hist(samples_med[:, 1], bins = int(np.sqrt(len(samples_med))), histtype = 'step', density = True, color = 'red')

fig.axes[1].legend(*fig.axes[0].get_legend_handles_labels(), loc = 'center')

fig.savefig(paths.figures / 'real_reconstruction.pdf', bbox_inches='tight')

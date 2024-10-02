from figaro.load import load_density
from figaro.plot import plot_median_cr
from figaro import plot_settings
import numpy as np
from matplotlib import pyplot as plt
import paths

label = 'real'
outdir = paths.data / label

draws = load_density(outdir / f'draws/draws_observed_{label}.json')
samples_med = np.loadtxt(outdir / 'samples_med.txt')

fig, ax  = plt.subplots()
fig = plot_median_cr(draws, hierarchical=True, save=True, show=False, fig=fig, median_label='$p(m^z_1|\mathbf{\Theta})$')
ax.hist(samples_med, bins = int(np.sqrt(len(samples_med))), histtype = 'step', density = True, label = '$\mathrm{median}(Y_t)$', color = 'tab:red')
ax.set_xlim(0,200)
ax.set_ylim(0)
ax.set_xlabel('$m^z_1\ [\mathrm{M}_\odot]$')
ax.set_ylabel('$\mathrm{Density}$')
ax.legend()
fig.savefig(paths.figures / 'real_reconstruction.pdf', bbox_inches='tight')

from figaro.load import load_density
from figaro.plot import plot_median_cr
from figaro import plot_settings
import numpy as np
import paths

label = 'simulation'
outdir = paths.data / 'simulation'

draws = load_density(outdir / f'draws/draws_observed_{label}.json')
obs_samples = np.loadtxt(outdir / 'obs_samples.txt')

fig = plot_median_cr(draws, hierarchical=True, save=True, show=False)
ax = fig.axes[0]
ax.hist(obs_samples, bins = int(np.sqrt(len(obs_samples))), histtype = 'step', density = True, label = '$\mathrm{median}(Y_t)$', color = 'tab:red')
ax.set_xlim(0,120)
ax.set_ylim(0)
ax.set_xlabel('$m^z_1\ [\mathrm{M}_\odot]$')
ax.set_ylabel('$\mathrm{Density}$')
ax.legend()
fig.savefig(paths.figures / 'simulation_reconstruction.pdf', bbox_inches='tight')

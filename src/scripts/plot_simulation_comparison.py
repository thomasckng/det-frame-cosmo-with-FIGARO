from figaro.load import load_density
from figaro.plot import plot_median_cr
from figaro import plot_settings
import numpy as np
import paths

label = 'simulation'
outdir = paths.data / label

draws = load_density(outdir / f'draws/draws_observed_{label}.json')
mz, H0, z, m, model_pdf = np.load(paths.data / 'grid.npz').values()
bounds = np.loadtxt(outdir / "jsd_bounds.txt")

fig = plot_median_cr(draws, hierarchical=True, save=True, show=False, median_label='$p(m^z_1|\mathbf{\Theta})$')
ax = fig.axes[0]
colors = ['tab:green', 'tab:red', 'tab:orange']
for i, c in zip([np.argmin(abs(H0-40)), np.argmin(abs(H0-70)), np.argmin(abs(H0-100))], colors):
    ax.plot(mz, model_pdf[:,i]/np.trapz(model_pdf[:,i], mz), label=f"$H_0={H0[i]:.0f}$", c=c)
ax.axvspan(0, bounds[0], color='gray', alpha=0.2)
ax.axvspan(bounds[1], 200, color='gray', alpha=0.2)
ax.set_xlabel('$m^z_1\ [\mathrm{M}_\odot]$')
ax.set_ylabel('$\mathrm{Density}$')
ax.set_xlim(0, 200)
ax.set_ylim(0)
ax.legend()
fig.savefig(paths.figures / "simulation_comparison.pdf", bbox_inches='tight')

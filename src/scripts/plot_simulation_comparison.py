from figaro.load import load_density
from figaro.plot import plot_median_cr
from figaro import plot_settings
import numpy as np
import paths

label = 'simulation'
outdir = paths.data / label

draws = load_density(outdir / f'draws/draws_observed_{label}.json')

mz, H0, z, m, model_pdf = np.load(outdir / '../grid.npz').values()

bounds = np.loadtxt(outdir / "jsd_bounds.txt")

fig = plot_median_cr(draws, save=True, show=False)
ax = fig.axes[0]

colors = ['tab:green', 'tab:red', 'tab:orange']
for i, c in zip([np.argmin(abs(H0-40)), np.argmin(abs(H0-70)), np.argmin(abs(H0-100))], colors):
    ax.plot(mz, model_pdf[:,i]/np.trapz(model_pdf[:,i], mz), label=f"$H_0={H0[i]:.0f}$", c=c)
ax.legend()
ax.set_xlabel('$m^z_1\ [\mathrm{M}_\odot]$')
ax.set_ylabel('Density')
ax.set_xlim(bounds[0], bounds[1])
ax.set_ylim(0)

fig.savefig(outdir+"/comparison.pdf", bbox_inches='tight')

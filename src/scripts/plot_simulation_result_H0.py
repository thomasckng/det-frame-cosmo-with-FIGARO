from figaro import plot_settings
from figaro.cosmology import Planck18
import numpy as np
import matplotlib.pyplot as plt
import paths

H0_samples = np.loadtxt(paths.data / 'simulation' / 'H0s.txt')
true_H0 = Planck18.h*100

fig, ax = plt.subplots()
ax.hist(H0_samples, bins = int(np.sqrt(len(H0_samples))), histtype = 'step', density = True, color = 'tab:blue')
percs = np.percentile(H0_samples, [5, 16, 50, 84, 95])
ax.axvline(true_H0, lw = 0.7, ls = '--', c = 'red', label = f'$H_0={true_H0:.1f}\ (\mathrm{{Simulated}})$')
ax.axvline(percs[2], c = 'steelblue', lw=0.7, label = f'$H_0={percs[2]:.1f}^{{+{percs[3]-percs[2]:.1f}}}_{{-{percs[2]-percs[1]:.1f}}}$')
ax.axvspan(percs[1], percs[3], alpha=0.35, color='mediumturquoise')
ax.axvspan(percs[0], percs[4], alpha=0.2, color='darkturquoise')
ax.legend()
ax.set_xlabel('$H_0\ [\mathrm{km/s/Mpc}]$')
ax.set_ylabel('$\mathrm{Density}$')
ax.set_xlim(percs[2]-30, percs[2]+30)
fig.savefig(paths.figures / "simulation_result_H0.pdf", bbox_inches='tight')

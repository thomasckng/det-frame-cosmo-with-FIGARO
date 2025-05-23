from figaro import plot_settings
from figaro.cosmology import Planck18
import numpy as np
import matplotlib.pyplot as plt
import paths

fig_jsd, ax_jsd = plt.subplots()
colors = [['tab:blue', 'steelblue', 'mediumturquoise', 'darkturquoise'], ['tab:red', 'salmon', 'lightcoral', 'indianred']]

for i, mass_dist in enumerate(['PLP', 'PL']):
    result = np.load(paths.data / f'simulation/H0/{mass_dist}.npz')
    H0_samples = result['samples']
    jsd_samples = result['jsd']
    true_H0 = Planck18.h*100

    # Plot H0
    fig, ax = plt.subplots()
    ax.hist(H0_samples, bins = int(np.sqrt(len(H0_samples))), histtype = 'step', density = True, color = colors[i][0])
    percs = np.percentile(H0_samples, [5, 15.87, 50, 84.13, 95])
    ax.axvline(true_H0, lw = 0.7, ls = '--', c = 'red', label = f'$H_0={true_H0:.1f}\ (\mathrm{{Simulated}})$')
    ax.axvline(percs[2], c = colors[i][1], lw=0.7, label = f'$H_0={percs[2]:.1f}^{{+{percs[3]-percs[2]:.1f}}}_{{-{percs[2]-percs[1]:.1f}}}$')
    ax.axvspan(percs[1], percs[3], alpha=0.35, color=colors[i][2])
    ax.axvspan(percs[0], percs[4], alpha=0.2, color=colors[i][3])
    ax.legend(loc='upper left')
    ax.set_xlabel('$H_0\ [\mathrm{km/s/Mpc}]$')
    ax.set_ylabel('$\mathrm{Density}$')
    ax.set_xlim(percs[2]-2*(percs[2]-percs[0]), percs[2]+2*(percs[4]-percs[2]))
    fig.savefig(paths.figures / f"simulation_result_H0_{mass_dist}.pdf", bbox_inches='tight')
    fig.clf()

    # Plot JSD
    percs = np.percentile(jsd_samples, [5, 15.87, 50, 84.13, 95])
    ax_jsd.hist(jsd_samples, bins = int(np.sqrt(len(jsd_samples))), histtype = 'step', density = True, color = colors[i][0], label = f'$\mathrm{{{mass_dist}}}\,(d_\mathrm{{JS}} = {percs[2]:.3f}^{{+{percs[3]-percs[2]:.3f}}}_{{-{percs[2]-percs[1]:.3f}}})$')
    ax_jsd.axvline(percs[2], c = colors[i][1], lw=0.7)
    ax_jsd.axvspan(percs[1], percs[3], alpha=0.35, color=colors[i][2])
    ax_jsd.axvspan(percs[0], percs[4], alpha=0.2, color=colors[i][3])

ax_jsd.legend()
ax_jsd.set_xlabel('$d_\mathrm{JS}$')
ax_jsd.set_ylabel('$\mathrm{Density}$')
fig_jsd.savefig(paths.figures / f"simulation_result_H0_jsd.pdf", bbox_inches='tight')

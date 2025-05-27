from figaro import plot_settings
from figaro.cosmology import Planck18
import numpy as np
import matplotlib.pyplot as plt
import paths

fig_jsd, ax_jsd = plt.subplots()
colors = [['tab:blue', 'steelblue', 'mediumturquoise', 'darkturquoise'], ['tab:red'], ['tab:green']]

for i, mu in enumerate([35.0, 45.0, 55.0]):
    result = np.load(paths.data / f'simulation/H0/PLP_{mu}.npz')
    H0_samples = result['samples']
    jsd_samples = result['jsd']
    true_H0 = Planck18.h*100

    # Plot H0
    fig, ax = plt.subplots()
    ax.hist(H0_samples, bins = int(np.sqrt(len(H0_samples))), histtype = 'step', density = True, color = colors[0][0])
    percs = np.percentile(H0_samples, [5, 15.87, 50, 84.13, 95])
    ax.axvline(true_H0, lw = 0.7, ls = '--', c = 'red', label = f'$H_0={true_H0:.1f}\ (\mathrm{{Simulated}})$')
    ax.axvline(percs[2], c = colors[0][1], lw=0.7, label = f'$H_0={percs[2]:.1f}^{{+{percs[3]-percs[2]:.1f}}}_{{-{percs[2]-percs[1]:.1f}}}$')
    ax.axvspan(percs[1], percs[3], alpha=0.35, color=colors[0][2])
    ax.axvspan(percs[0], percs[4], alpha=0.2, color=colors[0][3])
    ax.legend(loc='upper left')
    ax.set_xlabel('$H_0\ [\mathrm{km/s/Mpc}]$')
    ax.set_ylabel('$\mathrm{Density}$')
    ax.set_xlim(percs[2]-2*(percs[2]-percs[0]), percs[2]+2*(percs[4]-percs[2]))
    fig.savefig(paths.figures / f"simulation_result_H0_PLP_{mu}.pdf", bbox_inches='tight')
    fig.clf()

    # Plot JSD
    ax_jsd.hist(jsd_samples, bins = int(np.sqrt(len(jsd_samples))), histtype = 'step', density = True, color = colors[i][0], label = f'$\mu={int(mu)}$')

ax_jsd.legend()
ax_jsd.set_xlabel('$d_\mathrm{JS}$')
ax_jsd.set_ylabel('$\mathrm{Density}$')
fig_jsd.savefig(paths.figures / f"simulation_result_H0_jsd_mu.pdf", bbox_inches='tight')

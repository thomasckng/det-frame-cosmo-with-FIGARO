from figaro import plot_settings
from figaro.cosmology import Planck18
import numpy as np
import matplotlib.pyplot as plt
import paths

H0_samples = np.loadtxt(paths.data / 'simulation' / 'H0s.txt')
true_H0 = Planck18.h*100

plt.hist(H0_samples)
percs = np.percentile(H0_samples, [5, 16, 50, 84, 95])
plt.axvline(true_H0, lw = 0.7, ls = '--', c = 'orangered', label = '$\\mathrm{Simulated}$')
plt.axvline(percs[2], c = 'steelblue', lw=0.7, label = '$H_0 = '+str(f'{percs[2]:.1f}')+'^{+'+str(f'{percs[3]-percs[2]:.1f}')+'}_{-'+str(f'{percs[2]-percs[1]:.1f}')+'}$')
plt.axvspan(percs[1], percs[3], alpha=0.25, color='mediumturquoise')
plt.axvspan(percs[0], percs[4], alpha=0.25, color='darkturquoise')
plt.legend()
plt.xlabel('$H_0\ [\mathrm{km/s/Mpc}]$')
plt.ylabel('$\mathrm{Density}$')
plt.xlim(true_H0-20, true_H0+20)
plt.savefig(paths.figures / "simulation_result_H0.pdf", bbox_inches='tight')

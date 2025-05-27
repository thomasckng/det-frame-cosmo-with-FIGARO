from figaro import plot_settings
import numpy as np
import matplotlib.pyplot as plt
import paths

fig_jsd, ax_jsd = plt.subplots()

result = np.load(paths.data / f'simulation/H0/PLP_35.0.npz')
jsd_samples = result['jsd']

ax_jsd.hist(jsd_samples, bins = int(np.sqrt(len(jsd_samples))), histtype = 'step', density = True, color = 'tab:blue', label = f'$N=309$')

result = np.load(paths.data / f'simulation_less/H0/PLP_35.0.npz')
jsd_samples = result['jsd']

ax_jsd.hist(jsd_samples, bins = int(np.sqrt(len(jsd_samples))), histtype = 'step', density = True, color = 'tab:red', label = f'$N=89$')

ax_jsd.legend()
ax_jsd.set_xlabel('$d_\mathrm{JS}$')
ax_jsd.set_ylabel('$\mathrm{Density}$')
fig_jsd.savefig(paths.figures / f"simulation_result_H0_jsd_N.pdf", bbox_inches='tight')

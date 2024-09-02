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

fig, (ax1, ax2) = plt.subplots(2, figsize=(7,6), gridspec_kw={'hspace': 0.3})
fig = plot_median_cr(draws, hierarchical=True, save=True, show=False, fig=fig, median_label='$p(m^z_1|\mathbf{\Theta})$')
ax1.hist(samples_med, bins = int(np.sqrt(len(samples_med))), histtype = 'step', density = True, label = '$\mathrm{median}(Y_t)$', color = 'tab:red')
ax1.set_xlim(0,200)
ax1.set_ylim(0)
ax1.set_xlabel('$m^z_1\ [\mathrm{M}_\odot]$')
ax1.set_ylabel('$\mathrm{Density}$')
ax1.legend()
m_z = np.linspace(0,200,1000)
ax2.plot(m_z, draws[0].pdf(m_z), alpha = 0.5, color = 'tab:blue', label = '$p(m^z_1|\Theta_i)$')
for i, draw in enumerate(draws[1:10]):
    ax2.plot(m_z, draw.pdf(m_z), alpha = 0.5, color = 'tab:blue')
ax2.hist(samples_med, bins = int(np.sqrt(len(samples_med))), histtype = 'step', density = True, label = '$\mathrm{median}(Y_t)$', color = 'tab:red')
ax2.set_xlim(0,200)
ax2.set_ylim(0)
ax2.set_xlabel('$m^z_1\ [\mathrm{M}_\odot]$')
ax2.set_ylabel('$\mathrm{Density}$')
ax2.legend()
fig.savefig(paths.figures / 'real_reconstruction.pdf', bbox_inches='tight')

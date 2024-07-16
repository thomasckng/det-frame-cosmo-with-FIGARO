from matplotlib import pyplot as plt
import numpy as np
from figaro import plot_settings
import paths

label = 'simulation'
outdir = paths.data / label

# Mass distribution
from population_models.mass import plpeak
mz, H0, z, m, model_pdf = np.load(outdir / '../grid.npz').values()

fig, (ax1, ax2) = plt.subplots(2, figsize=(7,6), gridspec_kw={'hspace': 0.3})
m = np.linspace(0,120,1000)
ax1.plot(m, plpeak(m)/np.trapz(plpeak(m), m), label = "$\mathrm{PL}+\mathrm{Peak}$", c = 'tab:blue')
ax1.set_xlabel('$m_1\ [\mathrm{M}_\odot]$')
ax1.set_ylabel('$p(m_1|\Lambda)$')
ax1.set_xlim(0,120)
ax1.set_ylim(0)
ax1.legend()
colors = ['tab:green', 'tab:red', 'tab:orange']
for i, c in zip([np.argmin(abs(H0-40)), np.argmin(abs(H0-70)), np.argmin(abs(H0-100))], colors):
    ax2.plot(mz, model_pdf[:,i]/np.trapz(model_pdf[:,i], mz), label=f"$H_0={H0[i]:.0f}$", c=c)
ax2.set_xlabel('$m^z_1\ [\mathrm{M}_\odot]$')
ax2.set_ylabel('$p(m^z_1|\Lambda, \Omega(H_0), \mathrm{det})$')
ax2.set_xlim(0,120)
ax2.set_ylim(0)
ax2.legend()
fig.savefig(paths.figures / "simulation_transformation.pdf", bbox_inches='tight')

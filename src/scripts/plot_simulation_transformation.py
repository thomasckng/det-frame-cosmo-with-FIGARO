from matplotlib import pyplot as plt
import numpy as np
import paths

label = 'simulation'
outdir = paths.data / label

mz, H0, z, m, model_pdf = np.load(outdir / '../grid.npz').values()

# Mass distribution
from population_models.mass import plpeak

fig, (ax1, ax2) = plt.subplots(2)

ax1.plot(m, plpeak(m)/np.trapz(plpeak(m), m), label = "PL+Peak", c = 'tab:blue')
ax1.xlabel('$m_1\ [\mathrm{M}_\odot]$')
ax1.ylabel('$p(m_1|\Lambda)$')
ax1.legend()
ax1.xlim(0,120)
ax1.ylim(0)

colors = ['tab:green', 'tab:red', 'tab:orange']
for i, c in zip([np.argmin(abs(H0-40)), np.argmin(abs(H0-70)), np.argmin(abs(H0-100))], colors):
    ax2.plot(mz, model_pdf[:,i]/np.trapz(model_pdf[:,i], mz), label=f"$H_0={H0[i]:.0f}$", c=c)
ax2.legend()
ax2.xlabel('$m^z_1\ [\mathrm{M}_\odot]$')
ax2.ylabel('$p(m^z_1|\Lambda, \Omega(H_0), \mathrm{det})$')
ax2.xlim(0,120)
ax2.ylim(0)

fig.savefig(outdir / "simulation_transformation.pdf", bbox_inches='tight')

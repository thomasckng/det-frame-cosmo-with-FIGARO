import numpy as np
from corner import corner
from figaro import plot_settings
import paths

f = np.load(paths.data / 'real/multi' / '9_CMA-ES.npz')
result = f['result']
parameters = ["$H_0$", "$\\alpha$", "$\\mu$", "$\\sigma$", "$w$", "$\\delta$", "$m_\mathrm{min}$", "$m_\mathrm{max}$", "$\\kappa$"]

fig = corner(result, labels=parameters, color='steelblue', truth_color='red', levels = [0.3935, 0.90], plot_datapoints=False, plot_density=False, smooth=1, fill_contours=True, contourf_kwargs={'colors': ['white', 'darkturquoise', 'mediumturquoise'], 'alpha': [1, 0.2, 0.5]}, contour_kwargs={'linewidths': 1}, show_titles=True)
fig.savefig(paths.figures / "real_result_multi.pdf", bbox_inches='tight')

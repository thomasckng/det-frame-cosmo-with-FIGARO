import numpy as np
import matplotlib.pyplot as plt
from corner import corner
from figaro.cosmology import Planck18
from figaro import plot_settings
import paths

f = np.load(paths.data / 'simulation/multi' / '4b_Powell.npz')
result = f['result']
parameters = ["$H_0$", "$\\alpha$", "$\\mu$", "$\\sigma$"]

fig = corner(result, labels=parameters, color='steelblue', truth_color='red', levels = [0.3935, 0.9], plot_datapoints=False, plot_density=False, smooth=1, fill_contours=True, contourf_kwargs={'colors': ['white', 'darkturquoise', 'mediumturquoise'], 'alpha': [1, 0.2, 0.5]}, contour_kwargs={'linewidths': 1})
fig.savefig(paths.figures / "real_result_multi.pdf", bbox_inches='tight')

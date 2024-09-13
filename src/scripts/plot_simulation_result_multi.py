import numpy as np
import matplotlib.pyplot as plt
from corner import corner
from figaro.cosmology import Planck18
from figaro import plot_settings
import paths

f = np.load(paths.data / 'simulation/multi' / '4a_Powell.npz')
result = f['result']
simulated_truth = {'$H_0$': Planck18.h*100, '$\\alpha$': 3.5, '$\\mu$': 35, '$\\sigma$': 5, '$\\delta$': 5, '$w$': 0.2, '$m_\mathrm{min}$': 5, '$m_\mathrm{max}$': 90}
parameters = ["$H_0$", "$\\mu$", "$\\sigma$", "$\\delta$"]

fig = corner(result, labels=parameters, truths=[simulated_truth[param] for param in parameters], color='steelblue', truth_color='red', levels = [0.3935, 0.9], plot_datapoints=False, plot_density=False, smooth=1, fill_contours=True, contourf_kwargs={'colors': ['white', 'darkturquoise', 'mediumturquoise'], 'alpha': [1, 0.2, 0.5]}, contour_kwargs={'linewidths': 1})
fig.savefig(paths.figures / "simulation_result_multi.pdf", bbox_inches='tight')

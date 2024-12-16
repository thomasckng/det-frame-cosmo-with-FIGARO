import numpy as np
from corner import corner
from figaro import plot_settings
import paths

f = np.load(paths.data / 'real/multi' / '4c_Powell.npz')
result = f['result']
result = result[(result[:, 3] > -5) & (result[:, 3] < 15)] # plot limits
parameters = ["$H_0$", "$\\mu$", "$\\sigma$", "$\\kappa$"]

fig = corner(result, labels=parameters, color='steelblue', truth_color='red', levels = [0.3935, 0.90], plot_datapoints=False, plot_density=False, smooth=1, fill_contours=True, contourf_kwargs={'colors': ['white', 'darkturquoise', 'mediumturquoise'], 'alpha': [1, 0.2, 0.5]}, contour_kwargs={'linewidths': 1})
fig.savefig(paths.figures / "real_result_multi.pdf", bbox_inches='tight')

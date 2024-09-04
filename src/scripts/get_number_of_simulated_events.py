import paths
import numpy as np

n_true_samples = len(np.loadtxt(paths.data/'simulation/true_samples.txt'))
n_obs_samples = len(np.loadtxt(paths.data/'simulation/obs_samples.txt'))

with open(paths.output/'n_true_samples.txt', 'w') as f:
    f.write(str(n_true_samples))

with open(paths.output/'n_obs_samples.txt', 'w') as f:
    f.write(str(n_obs_samples))

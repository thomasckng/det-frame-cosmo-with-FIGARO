import numpy as np
import paths

f = np.load(paths.data / 'real/multi' / '4d_Powell.npz')
result = f['result']
result = result[(result[:, 0] > 40) & (result[:, 0] < 100)]
H0 = result[:, 0]
med = np.median(H0)
low = np.percentile(H0, 10)
high = np.percentile(H0, 90)

with open(paths.output/'real_H0.txt', 'w') as f:
    f.write(f'$H_0 = {med:.2f}^{{+{high - med:.2f}}}_{{-{med - low:.2f}}} \, \mathrm{{km/s/Mpc}}$')

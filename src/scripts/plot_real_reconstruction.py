from figaro.load import load_density
from figaro.plot import joyplot
from figaro import plot_settings
import numpy as np
from matplotlib import pyplot as plt
import paths
from tqdm import tqdm

label = 'real'
outdir = paths.data / label

draws = load_density(outdir / f'draws/draws_intrinsic_{label}.json')

DL = np.linspace(10,5000,10)
M  = np.linspace(0,150,500)

dd = np.array([[d.condition([D],[1]).pdf(M) for d in draws] for D in tqdm(DL)])

joyplot(dd, M, DL, save = True, out_folder = paths.figures, name = 'real_joyplot', xlabel = '\\mathrm{M}_1^z\ [\\mathrm{M}_\\odot]', ylabel = '\\mathrm{D}_L\ [\\mathrm{Mpc}]', credible_regions = True)

import numpy as np
from figaro.cosmology import CosmologicalParameters
from scipy.spatial.distance import jensenshannon as scipy_jsd
from figaro.load import load_density
import sys
from tqdm import tqdm
import dill
import paths

# Mass distribution
from population_models.mass import plpeak # from from https://github.com/sterinaldi/cbc_pdet

# Redshift distribution
def p_z(z, H0):
    return CosmologicalParameters(H0/100., 0.315, 0.685, -1., 0., 0.).ComovingVolumeElement(z)/(1+z)

label = sys.argv[1]
outdir = paths.data / label

print("Preparing model pdfs...")
grid_label = "grid"
try:
    mz, H0, z, m, model_pdf = np.load(outdir / f"../{grid_label}.npz").values()
except:
    mz = np.linspace(1,200,900)
    H0 = np.linspace(5,150,1000)
    z = np.linspace(0.001,2,800)
    m = np.einsum("i, j -> ij", mz, np.reciprocal(1+z)) # shape = (len(mz), len(z))

    # Calculate source-frame population model pdf for each H0
    model_pdf = np.einsum("ij, kj -> ijk", plpeak(m), [p_z(z, i) for i in H0]) # shape = (len(mz), len(z), len(H0))

    # Load selection function
    with open(outdir / '../selection_function.pkl', 'rb') as f: # selection_function.pkl is generated by generate_selection_function.py
        selfunc_interp = dill.load(f)
    def selection_function(x):
        return selfunc_interp(x)
    
    grid = [np.transpose(np.meshgrid(mz, CosmologicalParameters(i/100., 0.315, 0.685, -1., 0., 0.).LuminosityDistance(z))) for i in H0] # shape = (len(H0), len(mz), len(z), 2)
    SE_grid = np.array([selection_function(grid[i]) for i in tqdm(range(len(H0)), desc = 'selection function grid')]) # shape = (len(H0), len(mz), len(z))
    model_pdf = np.einsum("ijk, kij -> ijk", model_pdf, SE_grid) # shape = (len(mz), len(z), len(H0))

    model_pdf = np.trapz(model_pdf, z, axis=1) # shape = (len(mz), len(H0))

    np.savez(outdir / f"../{grid_label}.npz", mz=mz, H0=H0, z=z, m=m, model_pdf=model_pdf)

print("Reading bounds and draws...")
draws = load_density(outdir / f"draws/draws_observed_{label}.json") # follow README.md to generate draws

bounds = np.loadtxt(outdir / "jsd_bounds.txt") # jsd_bounds.txt is generated by simulate_posterior_samples.py

print("Preparing H0 inference...")
# Mask out mz where there is no sample
_mask = [mz[k] <= bounds[1] and mz[k] >= bounds[0] for k in range(len(mz))]
mz_short = mz[_mask]
model_pdf_short = model_pdf[_mask]

figaro_pdf = np.array([draw.pdf(mz_short) for draw in draws])# shape (n_draws, len(mz_short))

print("Inferring H0...")
# Compute JSD between (reconstructed observed distributions for each DPGMM draw) and (model mz distributions for each H0)
jsd = np.array([scipy_jsd(model_pdf_short, np.full((len(H0), len(mz_short)), figaro_pdf[j]).T) for j in tqdm(range(len(figaro_pdf)), desc='JSD')])
# Find H0 that minimizes JSD for each DPGMM draw
H0_samples = H0[np.argmin(jsd, axis=1)]

print("Saving results...")
np.savetxt(outdir / "jsds.txt", jsd)
np.savetxt(outdir / "H0s.txt", H0_samples)

print("Done!")
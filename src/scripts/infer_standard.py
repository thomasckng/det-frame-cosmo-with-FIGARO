import raynest.model
import numpy as np
from figaro.cosmology import CosmologicalParameters
from population_models.mass import plpeak
import dill
import os
import sys
import paths
sys.executable = '/users/chi-kit.ng/.conda/envs/population/bin/python3'

label = sys.argv[2]
outdir = paths.data / label

def read_data():
    data = []
    for file in os.listdir(outdir/"data"):
        if file.endswith(".txt"):
            data.append(np.loadtxt(outdir/"data"/file))
    return np.array(data)

mz = np.linspace(1,200,900)
z = np.linspace(0.001,2,800)
m = np.einsum("i, j -> ij", mz, np.reciprocal(1+z)) # shape = (len(mz), len(z))

# Redshift distribution
def p_z(z, H0, kappa=0):
    # Fixed parameters are from the result of Planck 2018
    return CosmologicalParameters(H0/100., 0.315, 0.685, -1., 0., 0.).ComovingVolumeElement(z)*(1+z)**(kappa-1)

param = sys.argv[1]

bounds_dict = {
    "H0": [1, 350],
    "alpha": [1.01, 15],
    "mu": [0.01, 70],
    "sigma": [0.01, 60],
    "w": [0, 1],
    "delta": [0.01, 20],
    "mmin": [1, 10],
    "mmax": [70, 150],
    "kappa": [-100, 100]
}

if label == "simulation":
    if param == "1":
        names = ["H0"]
        def p_m_p_z(x):
            return np.einsum("ij, j -> ij", plpeak(m), p_z(z, x["H0"]))
    elif param == "2a":
        names = ["H0", "alpha"]
        def p_m_p_z(x):
            return np.einsum("ij, j -> ij", plpeak(m, alpha=x["alpha"]), p_z(z, x["H0"]))
    elif param == "2b":
        names = ["H0", "mu"]
        def p_m_p_z(x):
            return np.einsum("ij, j -> ij", plpeak(m, mu=x["mu"]), p_z(z, x["H0"]))
    elif param == "4a":
        names = ["H0", "mu", "sigma", "delta"]
        def p_m_p_z(x):
            return np.einsum("ij, j -> ij", plpeak(m, mu=x["mu"], sigma=x["sigma"], delta=x["delta"]), p_z(z, x["H0"]))
    elif param == "4b":
        names = ["H0", "alpha", "mu", "sigma"]
        def p_m_p_z(x):
            return np.einsum("ij, j -> ij", plpeak(m, alpha=x["alpha"], mu=x["mu"], sigma=x["sigma"]), p_z(z, x["H0"]))
    elif param == "5a":
        names = ["H0", "alpha", "mu", "sigma", "w"]
        def p_m_p_z(x):
            return np.einsum("ij, j -> ij", plpeak(m, alpha=x["alpha"], mu=x["mu"], sigma=x["sigma"], w=x["w"]), p_z(z, x["H0"]))
    elif param == "5b":
        names = ["H0", "alpha", "mu", "mmin", "w"]
        def p_m_p_z(x):
            return np.einsum("ij, j -> ij", plpeak(m, alpha=x["alpha"], mu=x["mu"], mmin=x["mmin"], w=x["w"]), p_z(z, x["H0"]))
    elif param == "6a":
        names = ["H0", "alpha", "mu", "sigma", "w", "delta"]
        def p_m_p_z(x):
            return np.einsum("ij, j -> ij", plpeak(m, alpha=x["alpha"], mu=x["mu"], sigma=x["sigma"], w=x["w"], delta=x["delta"]), p_z(z, x["H0"]))
    elif param == "8":
        names = ["H0", "alpha", "mu", "sigma", "w", "delta", "mmin", "mmax"]
        def p_m_p_z(x):
            return np.einsum("ij, j -> ij", plpeak(m, alpha=x["alpha"], mu=x["mu"], sigma=x["sigma"], w=x["w"], delta=x["delta"], mmin=x["mmin"], mmax=x["mmax"]), p_z(z, x["H0"]))
    else:
        print("Invalid argument!")
        sys.exit(1)

elif label == "real":
    # Fixed parameters are the median values from the result of "Constraints on the Cosmic Expansion History from GWTC–3" (SNR > 10 & w0flatLCDM)
    fixed_params = {
        "alpha": 4.2487241670754035,
        "mu": 31.825703461630482,
        "sigma": 3.7904971258458042,
        "w": 0.024059947239759398,
        "delta": 4.8961636235644015,
        "mmin": 5.0808821331157095,
        "mmax": 109.03299036617125
    }
    if param == "3a":
        names = ["H0", "alpha", "kappa"]
        def p_m_p_z(x):
            return np.einsum("ij, j -> ij", plpeak(m, alpha=x["alpha"], mu=fixed_params["mu"], sigma=fixed_params["sigma"], w=fixed_params["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"]), p_z(z, x["H0"], kappa=x["kappa"]))
    elif param == "3b":
        names = ["H0", "mu", "kappa"]
        def p_m_p_z(x):
            return np.einsum("ij, j -> ij", plpeak(m, mu=x["mu"], alpha=fixed_params["alpha"], sigma=fixed_params["sigma"], w=fixed_params["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"]), p_z(z, x["H0"], kappa=x["kappa"]))
    elif param == "4c":
        names = ["H0", "mu", "sigma", "kappa"]
        def p_m_p_z(x):
            return np.einsum("ij, j -> ij", plpeak(m, alpha=fixed_params["alpha"], mu=x["mu"], sigma=x["sigma"], w=fixed_params["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"]), p_z(z, x["H0"], kappa=x["kappa"]))
    elif param == "4d":
        names = ["H0", "alpha", "mu", "kappa"]
        def p_m_p_z(x):
            return np.einsum("ij, j -> ij", plpeak(m, alpha=x["alpha"], mu=x["mu"], sigma=fixed_params["sigma"], w=fixed_params["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"]), p_z(z, x["H0"], kappa=x["kappa"]))
    elif param == "5c":
        names = ["H0", "alpha", "mu", "sigma", "kappa"]
        def p_m_p_z(x):
            return np.einsum("ij, j -> ij", plpeak(m, alpha=x["alpha"], mu=x["mu"], sigma=x["sigma"], w=fixed_params["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"]), p_z(z, x["H0"], kappa=x["kappa"]))
    elif param == "6b":
        names = ["H0", "alpha", "mu", "sigma", "w", "kappa"]
        def p_m_p_z(x):
            return np.einsum("ij, j -> ij", plpeak(m, alpha=x["alpha"], mu=x["mu"], sigma=x["sigma"], w=x["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"]), p_z(z, x["H0"], kappa=x["kappa"]))
    elif param == "9":
        names = ["H0", "alpha", "mu", "sigma", "w", "delta", "mmin", "mmax", "kappa"]
        def p_m_p_z(x):
            return np.einsum("ij, j -> ij", plpeak(m, alpha=x["alpha"], mu=x["mu"], sigma=x["sigma"], w=x["w"], delta=x["delta"], mmin=x["mmin"], mmax=x["mmax"]), p_z(z, x["H0"], kappa=x["kappa"]))
    else:
        print("Invalid argument!")
        sys.exit(1)
else:
    print("Invalid argument!")
    sys.exit(1)

# Load selection function
with open(paths.data/'selection_function.pkl', 'rb') as f: # selection_function.pkl is generated by generate_selection_function.py
    selfunc_interp = dill.load(f)
def selection_function(x):
    return selfunc_interp(x)

def log_likelihood(M_z, x, N_event, N_mz):
    model_pdf = p_m_p_z(x) # shape = (len(mz), len(z))
    log_likelihood = np.sum(np.log(np.sum(np.interp(M_z, mz, np.trapz(model_pdf, z, axis=1)), axis=1))) - np.log(N_mz) * N_event

    grid = np.transpose(np.meshgrid(mz, CosmologicalParameters(x["H0"]/100., 0.315, 0.685, -1., 0., 0.).LuminosityDistance(z))) # shape = (len(mz), len(z), 2)
    SE_grid = selection_function(grid) # shape = (len(mz), len(z))
    model_pdf = np.einsum("ij, ij -> ij", model_pdf, SE_grid) # shape = (len(mz), len(z))
    model_pdf = np.trapz(model_pdf, z, axis=1) # shape = (len(mz))
    log_likelihood -= np.log(np.trapz(model_pdf, mz)) * N_event
    return log_likelihood

class Inference(raynest.model.Model):

    def __init__(self):
        super(Inference,self).__init__()
        self.Mz = read_data()
        self.N_event = len(self.Mz)
        self.N_mz = len(self.Mz[0])
        self.names = names
        self.bounds = [bounds_dict[name] for name in self.names]

    def log_prior(self, x):
        return super(Inference,self).log_prior(x)

    def log_likelihood(self, x):
        return log_likelihood(self.Mz, x, self.N_event, self.N_mz)

if __name__ == '__main__':

    W = Inference()
    work = raynest.raynest(W, verbose = 2, output = outdir/'inference', nnest = int(sys.argv[3])//10, nensemble = int(sys.argv[3]), nlive = 100)
    work.run()
    post = work.posterior_samples.ravel()
    np.save(outdir/f'multi/{param}_standard.npy', post)

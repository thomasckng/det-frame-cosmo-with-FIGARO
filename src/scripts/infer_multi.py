import sys
import os
import paths

label = sys.argv[3]
outdir = paths.data / label

if len(sys.argv) != 5:
    print("Invalid number of arguments!")
    sys.exit(1)

param = sys.argv[1]
method = sys.argv[2]

if not os.path.exists(outdir/f'multi/{param}_{method}.npz'):
    import numpy as np
    from numpy.random import uniform as uni
    from scipy.spatial.distance import jensenshannon
    from figaro.load import load_density
    from figaro.cosmology import CosmologicalParameters
    from multiprocessing import Pool
    from population_models.mass import plpeak
    import dill

    # Redshift distribution
    def p_z(z, H0, kappa=0):
        # Fixed parameters are from the result of Planck 2018
        Omega = CosmologicalParameters(H0/100., 0.315, 0.685, -1., 0., 0.)
        return Omega.ComovingVolumeElement(z)*(1+z)**(kappa-2)/Omega.dDLdz(z) # Jacobian for m, z to mz, dL is included. Jacobian = (1+z)**(-1)/Omega.dDLdz(z)

    bounds_dict = {
        "H0": (1, 350),
        "alpha": (1.01, 15),
        "mu": (0.01, 70),
        "sigma": (0.01, 60),
        "w": (0, 1),
        "delta": (0.01, 20),
        "mmin": (1, 10),
        "mmax": (70, 150),
        "kappa": (-10, 10)
    }
    if label == "simulation":
        fixed_params = {
            "alpha": 3.5,
            "mmin": 5,
            "mmax": 90,
            "delta": 5,
            "mu": 35,
            "sigma": 5,
            "w": 0.2,
            "kappa": 0
        }
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
    else:
        print("Invalid label!")
        sys.exit(1)

    if param == "2a":
        param_list = ['H0', 'alpha']
        def p_m(m, x):
            return plpeak(m, alpha=x[param_list.index("alpha")], mu=fixed_params["mu"], sigma=fixed_params["sigma"], w=fixed_params["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"])
    elif param == "2b":
        param_list = ['H0', 'mu']
        def p_m(m, x):
            return plpeak(m, alpha=fixed_params["alpha"], mu=x[param_list.index("mu")], sigma=fixed_params["sigma"], w=fixed_params["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"])
    elif param == "3a":
        param_list = ['H0', 'alpha', 'kappa']
        def p_m(m, x):
            return plpeak(m, alpha=x[param_list.index("alpha")], mu=fixed_params["mu"], sigma=fixed_params["sigma"], w=fixed_params["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"])
    elif param == "3b":
        param_list = ['H0', 'mu', 'kappa']
        def p_m(m, x):
            return plpeak(m, alpha=fixed_params["alpha"], mu=x[param_list.index("mu")], sigma=fixed_params["sigma"], w=fixed_params["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"])
    elif param == "4a":
        param_list = ['H0', 'alpha', 'mu', 'sigma']
        def p_m(m, x):
            return plpeak(m, alpha=x[param_list.index("alpha")], mu=x[param_list.index("mu")], sigma=x[param_list.index("sigma")], w=fixed_params["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"])
    elif param == "4b":
        param_list = ['H0', 'mu', 'sigma', 'kappa']
        def p_m(m, x):
            return plpeak(m, alpha=fixed_params["alpha"], mu=x[param_list.index("mu")], sigma=x[param_list.index("sigma")], w=fixed_params["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"])
    elif param == "4c":
        param_list = ['H0', 'alpha', 'mu', 'kappa']
        def p_m(m, x):
            return plpeak(m, alpha=x[param_list.index("alpha")], mu=x[param_list.index("mu")], sigma=fixed_params["sigma"], w=fixed_params["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"])
    elif param == "5a":
        param_list = ['H0', 'alpha', 'mu', 'sigma', 'w']
        def p_m(m, x):
            return plpeak(m, alpha=x[param_list.index("alpha")], mu=x[param_list.index("mu")], sigma=x[param_list.index("sigma")], w=x[param_list.index("w")], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"])
    elif param == "5b":
        param_list = ['H0', 'alpha', 'mu', 'sigma', 'kappa']
        def p_m(m, x):
            return plpeak(m, alpha=x[param_list.index("alpha")], mu=x[param_list.index("mu")], sigma=x[param_list.index("sigma")], w=fixed_params["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"])
    elif param == "6":
        param_list = ['H0', 'alpha', 'mu', 'sigma', 'w', 'kappa']
        def p_m(m, x):
            return plpeak(m, alpha=x[param_list.index("alpha")], mu=x[param_list.index("mu")], sigma=x[param_list.index("sigma")], w=x[param_list.index("w")], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"])
    elif param == "9":
        param_list = ['H0', 'alpha', 'mu', 'sigma', 'w', 'delta', 'mmin', 'mmax', 'kappa']
        def p_m(m, x):
            return plpeak(m, alpha=x[param_list.index("alpha")], mu=x[param_list.index("mu")], sigma=x[param_list.index("sigma")], w=x[param_list.index("w")], delta=x[param_list.index("delta")], mmin=x[param_list.index("mmin")], mmax=x[param_list.index("mmax")])
    else:
        print("Invalid parameter!")
        sys.exit(1)
    
    bounds = [bounds_dict[p] for p in param_list]

    
    print("Loading posterior draws and bounds...")
    draws = load_density(outdir/f"draws/draws_observed_{label}.json")

    jsd_bounds = np.loadtxt(outdir/f"jsd_bounds.txt")

    print("Preparing inference grid and PDFs...")
    # Initialize mz and dL using jsd_bounds
    mz = np.linspace(jsd_bounds[0][0], jsd_bounds[0][1], 900)
    dL = np.linspace(jsd_bounds[1][0], jsd_bounds[1][1], 80)
    # For each H0, convert dL to z
    def get_z_from_dL(H0, dL):
        return CosmologicalParameters(H0/100., 0.315, 0.685, -1., 0., 0.).Redshift(dL)

    grid = np.transpose(np.meshgrid(mz, dL))  # shape = (len(mz), len(dL), 2)
    pdf_figaro = np.array([draw.pdf(grid) for draw in draws])  # shape = (n_draws, len(mz), len(dL))

    print("Defining model PDF and selection function...")
    # Model PDF construction
    if "kappa" in param_list:
        def model_pdf_func(x):
            # x: parameter vector, x[0] is H0
            z = get_z_from_dL(x[param_list.index("H0")], dL) # shape = (len(dL),)
            m = np.einsum("i, j -> ij", mz, np.reciprocal(1+z))  # shape = (len(mz), len(dL))
            model_pdf_m = p_m(m, x) # shape = (len(mz), len(dL))
            model_pdf_z = p_z(z, x[param_list.index("H0")], x[param_list.index("kappa")]) # shape = (len(mz), len(dL))
            return np.einsum("ij, j -> ij", model_pdf_m, model_pdf_z)  # shape = (len(mz), len(dL))
    else:
        def model_pdf_func(x):
            # x: parameter vector, x[0] is H0
            z = get_z_from_dL(x[param_list.index("H0")], dL) # shape = (len(dL),)
            m = np.einsum("i, j -> ij", mz, np.reciprocal(1+z)) # shape = (len(mz), len(dL))
            model_pdf_m = p_m(m, x) # shape = (len(mz), len(dL))
            model_pdf_z = p_z(z, x[param_list.index("H0")], fixed_params["kappa"]) # shape = (len(mz), len(dL))
            return np.einsum("ij, j -> ij", model_pdf_m, model_pdf_z) # shape = (len(mz), len(dL))

    print("Loading and applying selection function...")
    # Load selection function
    with open(paths.data/'selection_function.pkl', 'rb') as f: # selection_function.pkl is generated by generate_selection_function.py
        selfunc_interp = dill.load(f)
    def selection_function(x):
        return selfunc_interp(x)

    print("Defining JSD objective function...")
    # JSD function
    def jsd(x, i):
        model_pdf = model_pdf_func(x)  # shape = (len(mz), len(dL))
        SE_grid = selection_function(grid)  # shape = (len(mz), len(dL))
        model_pdf = np.einsum("ij, ij -> ij", model_pdf, SE_grid)  # shape = (len(mz), len(dL))
        return jensenshannon(model_pdf.ravel(), pdf_figaro[i].ravel())

    if method in ["Powell", "TNC"]:

        from scipy.optimize import minimize as scipy_minimize

        def minimize(i):
            x0 = [uni(*bounds[j]) for j in range(len(bounds))]
            return scipy_minimize(jsd, x0=x0, bounds=bounds, args=(i,), method=method).x
    elif method == "CMA-ES":

        import cma

        def minimize(i):
            x0 = [uni(*bounds[j]) for j in range(len(bounds))]
            return cma.fmin2(jsd, x0, 1, {'bounds': np.array(bounds).T.tolist(), 'CMA_stds': np.array(bounds).T[1]/4}, args=(i,))[0]
    else:
        print("Invalid method!")
        sys.exit(1)

    def minimize_and_save(i):
        result = minimize(i)
        np.save(outdir/f'checkpoints/{param}_{method}_{str(i)}', result)
        return result


    remaining = list(range(len(pdf_figaro)))
    if not os.path.exists(outdir/'checkpoints'):
        os.makedirs(outdir/'checkpoints')
    for i in range(len(pdf_figaro)):
        if os.path.exists(outdir/f'checkpoints/{param}_{method}_{str(i)}.npy'):
            remaining.remove(i)
    print(f"Remaining number of draws: {str(len(remaining))}")

    if len(remaining) > 0:
        print("Starting parallel inference for remaining draws...")
        n_pool = int(sys.argv[4])
        with Pool(n_pool) as p:
            p.map(minimize_and_save, remaining)

    print("Collecting and saving results...")
    result = []
    for i in range(len(pdf_figaro)):
        if os.path.exists(outdir/f'checkpoints/{param}_{method}_{str(i)}.npy'):
            try:
                result.append(np.load(outdir/f'checkpoints/{param}_{method}_{str(i)}.npy'))
            except EOFError:
                result.append(minimize_and_save(i))
    result = np.array(result)

    print("Saving final results to disk...")
    if not os.path.exists(outdir/'multi'):
        os.makedirs(outdir/'multi')
    np.savez(outdir/f"multi/{param}_{method}.npz", result=result, pdf_figaro=pdf_figaro)

print("Cleaning up checkpoints...")
for filename in os.listdir(outdir/'checkpoints'):
    if filename.startswith(f"{param}_{method}_"):
        os.remove(outdir/f"checkpoints/{filename}")

print("All done!")

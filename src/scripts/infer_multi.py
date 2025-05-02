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
        return CosmologicalParameters(H0/100., 0.315, 0.685, -1., 0., 0.).ComovingVolumeElement(z)*(1+z)**(kappa-1)

    bounds_dict = {
        "H0": (1, 350),
        "alpha": (1.01, 15),
        "mu": (0.01, 70),
        "sigma": (0.01, 60),
        "w": (0, 1),
        "delta": (0.01, 20),
        "mmin": (1, 10),
        "mmax": (70, 150),
        "kappa": (-100, 100)
    }
    if label == "simulation":
        if param == "2a":
            bounds = bounds_dict["H0"], bounds_dict["alpha"]
            def p_m(m, x):
                return plpeak(m, alpha=x[1])
        elif param == "2b":
            bounds = bounds_dict["H0"], bounds_dict["mu"]
            def p_m(m, x):
                return plpeak(m, mu=x[1])
        elif param == "4a":
            bounds = bounds_dict["H0"], bounds_dict["mu"], bounds_dict["sigma"], bounds_dict["delta"]
            def p_m(m, x):
                return plpeak(m, mu=x[1], sigma=x[2], delta=x[3])
        elif param == "4b":
            bounds = bounds_dict["H0"], bounds_dict["alpha"], bounds_dict["mu"], bounds_dict["sigma"]
            def p_m(m, x):
                return plpeak(m, alpha=x[1], mu=x[2], sigma=x[3])
        elif param == "5a":
            bounds = bounds_dict["H0"], bounds_dict["alpha"], bounds_dict["mu"], bounds_dict["sigma"], bounds_dict["w"]
            def p_m(m, x):
                return plpeak(m, alpha=x[1], mu=x[2], sigma=x[3], w=x[4])
        elif param == "5b":
            bounds = bounds_dict["H0"], bounds_dict["alpha"], bounds_dict["mu"], bounds_dict["mmin"], bounds_dict["w"]
            def p_m(m, x):
                return plpeak(m, alpha=x[1], mu=x[2], mmin=x[3], w=x[4])
        elif param == "6a":
            bounds = bounds_dict["H0"], bounds_dict["alpha"], bounds_dict["mu"], bounds_dict["sigma"], bounds_dict["w"], bounds_dict["delta"]
            def p_m(m, x):
                return plpeak(m, alpha=x[1], mu=x[2], sigma=x[3], w=x[4], delta=x[5])
        elif param == "8":
            bounds = bounds_dict["H0"], bounds_dict["alpha"], bounds_dict["mu"], bounds_dict["sigma"], bounds_dict["w"], bounds_dict["delta"], bounds_dict["mmin"], bounds_dict["mmax"]
            def p_m(m, x):
                return plpeak(m, alpha=x[1], mu=x[2], sigma=x[3], w=x[4], delta=x[5], mmin=x[6], mmax=x[7])
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
            bounds = bounds_dict["H0"], bounds_dict["alpha"], bounds_dict["kappa"]
            def p_m(m, x):
                return plpeak(m, alpha=x[1], mu=fixed_params["mu"], sigma=fixed_params["sigma"], w=fixed_params["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"])
        elif param == "3b":
            bounds = bounds_dict["H0"], bounds_dict["mu"], bounds_dict["kappa"]
            def p_m(m, x):
                return plpeak(m, mu=x[1], alpha=fixed_params["alpha"], sigma=fixed_params["sigma"], w=fixed_params["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"])
        elif param == "4c":
            bounds = bounds_dict["H0"], bounds_dict["mu"], bounds_dict["sigma"], bounds_dict["kappa"]
            def p_m(m, x):
                return plpeak(m, alpha=fixed_params["alpha"], mu=x[1], sigma=x[2], w=fixed_params["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"])
        elif param == "4d":
            bounds = bounds_dict["H0"], bounds_dict["alpha"], bounds_dict["mu"], bounds_dict["kappa"]
            def p_m(m, x):
                return plpeak(m, alpha=x[1], mu=x[2], sigma=fixed_params["sigma"], w=fixed_params["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"])
        elif param == "5c":
            bounds = bounds_dict["H0"], bounds_dict["alpha"], bounds_dict["mu"], bounds_dict["sigma"], bounds_dict["kappa"]
            def p_m(m, x):
                return plpeak(m, alpha=x[1], mu=x[2], sigma=x[3], w=fixed_params["w"], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"])
        elif param == "6b":
            bounds = bounds_dict["H0"], bounds_dict["alpha"], bounds_dict["mu"], bounds_dict["sigma"], bounds_dict["w"], bounds_dict["kappa"]
            def p_m(m, x):
                return plpeak(m, alpha=x[1], mu=x[2], sigma=x[3], w=x[4], delta=fixed_params["delta"], mmin=fixed_params["mmin"], mmax=fixed_params["mmax"])
        elif param == "9":
            bounds = bounds_dict["H0"], bounds_dict["alpha"], bounds_dict["mu"], bounds_dict["sigma"], bounds_dict["w"], bounds_dict["delta"], bounds_dict["mmin"], bounds_dict["mmax"], bounds_dict["kappa"]
            def p_m(m, x):
                return plpeak(m, alpha=x[1], mu=x[2], sigma=x[3], w=x[4], delta=x[5], mmin=x[6], mmax=x[7])
        else:
            print("Invalid argument!")
            sys.exit(1)
    else:
        print("Invalid argument!")
        sys.exit(1)

    
    print("Reading bounds and draws...")
    draws = load_density(outdir/f"draws/draws_observed_{label}.json")

    jsd_bounds = np.loadtxt(outdir/f"jsd_bounds.txt")


    print("Preparing inference...")
    # Define dL grid
    mz = np.linspace(1,200,900)
    dL = np.linspace(10, 25000, 80)
    # For each H0, convert dL to z
    def get_z_from_dL(H0, dL):
        return CosmologicalParameters(H0/100., 0.315, 0.685, -1., 0., 0.).Redshift(dL)

    # Mask out mz and dL where there is no sample
    _mz_mask = [mz[k] <= jsd_bounds[0][1] and mz[k] >= jsd_bounds[0][0] for k in range(len(mz))]
    mz_short = mz[_mz_mask]
    _dL_mask = [dL[k] <= jsd_bounds[1][1] and dL[k] >= jsd_bounds[1][0] for k in range(len(dL))]
    dL_short = dL[_dL_mask]

    grid = np.transpose(np.meshgrid(mz_short, dL_short))  # shape = (len(mz_short), len(dL_short), 2)
    pdf_figaro = np.array([draw.pdf(grid) for draw in draws])  # shape = (n_draws, len(mz_short), len(dL_short))

    # Model PDF construction
    if label == "simulation":
        def model_pdf_func(x):
            # x: parameter vector, x[0] is H0
            z = get_z_from_dL(x[0], dL_short)  # shape = (len(dL_short),)
            m = np.einsum("i, j -> ij", mz_short, np.reciprocal(1+z))  # shape = (len(mz_short), len(dL_short))
            model_pdf_m = p_m(m, x) # shape = (len(mz_short), len(dL_short))
            model_pdf_z = p_z(z, x[0]) # shape = (len(mz_short), len(dL_short))
            return np.einsum("ij, j -> ij", model_pdf_m, model_pdf_z)  # shape = (len(mz_short), len(dL_short))
    elif label == "real":
        def model_pdf_func(x):
            # x: parameter vector, x[0] is H0
            z = get_z_from_dL(x[0], dL_short)  # shape = (len(dL_short),)
            m = np.einsum("i, j -> ij", mz_short, np.reciprocal(1+z))  # shape = (len(mz_short), len(dL_short))
            model_pdf_m = p_m(m, x) # shape = (len(mz_short), len(dL_short))
            model_pdf_z = p_z(z, x[0], x[-1]) # shape = (len(mz_short), len(dL_short))
            return np.einsum("ij, j -> ij", model_pdf_m, model_pdf_z)  # shape = (len(mz_short), len(dL_short))

    # Load selection function
    with open(paths.data/'selection_function.pkl', 'rb') as f: # selection_function.pkl is generated by generate_selection_function.py
        selfunc_interp = dill.load(f)
    def selection_function(x):
        return selfunc_interp(x)

    # JSD function
    def jsd(x, i):
        model_pdf = model_pdf_func(x)  # shape = (len(mz_short), len(dL_short))
        SE_grid = selection_function(grid)  # shape = (len(mz_short), len(dL_short))
        model_pdf = np.einsum("ij, ij -> ij", model_pdf, SE_grid)  # shape = (len(mz_short), len(dL_short))
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
        print("Invalid argument!")
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
        print("Starting inference...")
        n_pool = int(sys.argv[4])
        with Pool(n_pool) as p:
            p.map(minimize_and_save, remaining)

    print("Collecting results...")
    result = []
    for i in range(len(pdf_figaro)):
        if os.path.exists(outdir/f'checkpoints/{param}_{method}_{str(i)}.npy'):
            try:
                result.append(np.load(outdir/f'checkpoints/{param}_{method}_{str(i)}.npy'))
            except EOFError:
                result.append(minimize_and_save(i))
    result = np.array(result)

    print("Saving results...")
    if not os.path.exists(outdir/'multi'):
        os.makedirs(outdir/'multi')
    np.savez(outdir/f"multi/{param}_{method}.npz", result=result, pdf_figaro=pdf_figaro)

print("Removing checkpoints...")
for filename in os.listdir(outdir/'checkpoints'):
    if filename.startswith(f"{param}_{method}_"):
        os.remove(outdir/f"checkpoints/{filename}")

print("Done!")

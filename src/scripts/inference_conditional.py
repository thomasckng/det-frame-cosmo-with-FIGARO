import sys
import os
import paths

label = sys.argv[3]
outdir = paths.data / label

if len(sys.argv) != 7:
    print("Invalid number of arguments!")
    sys.exit(1)

param = sys.argv[1]
method = sys.argv[2]

mz = float(sys.argv[6])

if not os.path.exists(outdir/f'multi/{param}_{method}_{mz}.npz'):
    import numpy as np
    from numpy.random import uniform as uni
    from scipy.spatial.distance import jensenshannon
    from figaro.load import load_density, load_data
    from figaro.cosmology import CosmologicalParameters
    from multiprocessing import Pool
    from population_models.mass import powerlaw_smoothed as pl_mass
    from population_models.redshift import powerlaw_redshift # Does not depend on H0 even though H0 is a parameter
    from scipy.stats import norm

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
        # Fixed parameters are the simulated truth
        fixed_params = {'alpha': 3.5,
                        'mu': 35,
                        'sigma': 5,
                        'delta': 5,
                        'w': 0.2,
                        'mmin': 5,
                        'mmax': 90,
                        'kappa': 0}
        # fixed_params = {'alpha': 3.5,
        #                 'mu': 35,
        #                 'sigma': 4,
        #                 'delta': 5,
        #                 'w': 0.1,
        #                 'mmin': 4,
        #                 'mmax': 80,
        #                 'kappa': 2}

    # elif label == "real":
    #     # Fixed parameters are the median values from the result of "Constraints on the Cosmic Expansion History from GWTC–3" (SNR > 10 & w0flatLCDM)
    #     fixed_params = {
    #         "alpha": 4.2487241670754035,
    #         "mu": 31.825703461630482,
    #         "sigma": 3.7904971258458042,
    #         "w": 0.024059947239759398,
    #         "delta": 4.8961636235644015,
    #         "mmin": 5.0808821331157095,
    #         "mmax": 109.03299036617125
    #     }

    if param == "1":
        if label == "real":
            print("Invalid parameter! Only works for simulation.")
            sys.exit(1)
        param_list = ['H0']
        bounds = [bounds_dict[p] for p in param_list]
        def p_m_p_z(x, m, z):
            pdf_m = (1 - fixed_params['w']) * pl_mass(m,
                                                      fixed_params['alpha'],
                                                      fixed_params['mmax'],
                                                      fixed_params['mmin'],
                                                      fixed_params['delta']) + fixed_params['w'] * norm(fixed_params['mu'],
                                                                                                        fixed_params['sigma']).pdf(m)
            pdf_z = powerlaw_redshift(z,
                                      fixed_params['kappa'])
            return pdf_m * pdf_z # shape = len(dL)
        
    elif param == "4a":
        if label == "real":
            print("Invalid parameter! Only works for simulation.")
            sys.exit(1)
        param_list = ['H0', 'alpha', 'mu', 'sigma']
        bounds = [bounds_dict[p] for p in param_list]
        
        def p_m_p_z(x, m, z):
            pdf_m = (1 - fixed_params['w']) * pl_mass(m,
                                                      x[param_list.index('alpha')],
                                                      fixed_params['mmax'],
                                                      fixed_params['mmin'],
                                                      fixed_params['delta']) + fixed_params['w'] * norm(x[param_list.index('mu')],
                                                                                                        x[param_list.index('sigma')]).pdf(m)
            pdf_z = powerlaw_redshift(z,
                                      fixed_params['kappa'])
            return pdf_m * pdf_z # shape = len(dL)

    elif param == "9":
        param_list = ['H0', 'alpha', 'mu', 'sigma', 'w', 'delta', 'mmin', 'mmax', 'kappa']
        bounds = [bounds_dict[p] for p in param_list]

        def p_m_p_z(x, m, z):
            pdf_m = (1 - x[param_list.index('w')]) * pl_mass(m,
                                                        x[param_list.index('alpha')],
                                                        x[param_list.index('mmax')],
                                                        x[param_list.index('mmin')],
                                                        x[param_list.index('delta')]) + x[param_list.index('w')] * norm(x[param_list.index('mu')],
                                                                                                                x[param_list.index('sigma')]).pdf(m)
            pdf_z = powerlaw_redshift(z,
                                      x[param_list.index('kappa')])
            return pdf_m * pdf_z # shape = len(dL)

    else:
        print("Invalid argument!")
        sys.exit(1)


    print("Reading data...")
    posteriors = load_data(outdir/sys.argv[5], par=['m1_detect', 'luminosity_distance'])[0]
    dL_median = [np.median(posteriors[i][:, 1]) for i in range(len(posteriors))]
    dL_min = np.min(dL_median)
    dL_max = np.max(dL_median)
    # dL_min = [np.min(posteriors[i][:, 1]) for i in range(len(posteriors))]
    # dL_max = [np.max(posteriors[i][:, 1]) for i in range(len(posteriors))]
    dL = np.linspace(np.min(dL_min), np.max(dL_max), 150)

    draws = load_density(outdir/f"draws_intrinsic/draws_intrinsic_{label}.json")
    # draws = draws[:1000] # downsample to 1000 draws


    def jsd(x, i):
        Omega = CosmologicalParameters(x[param_list.index('H0')]/100, 0.3065, 0.6935, -1) # shape = len(dL)
        z = Omega.Redshift(dL) # shape = len(dL)
        m = mz / (1+z) # shape = len(dL)
        ddLdz = Omega.dDLdz(z) # shape = len(dL)
        model_pdf = p_m_p_z(x, m, z) / (1+z) / ddLdz # shape = len(dL)

        return jensenshannon(model_pdf, pdf_figaro[i])
    

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
        
    elif method == "Grid":
        if param != "1":
            print("Invalid parameter! Grid search only works for H0.")
            sys.exit(1)

        H0 = np.linspace(bounds[0][0], bounds[0][1], 100)
        Omega = [CosmologicalParameters(h0/100, 0.3065, 0.6935, -1) for h0 in H0]
        z = np.array([Omega[i].Redshift(dL) for i in range(len(Omega))]) # shape = (len(Omega), len(dL))
        m = np.array([(mz / (1+z[i])) for i in range(len(Omega))]) # shape = (len(Omega), len(dL))
        ddLdz = np.array([Omega[i].dDLdz(z[i]) for i in range(len(Omega))]) # shape = (len(Omega), len(dL))
        denominator = np.array([(1+z[i]) * ddLdz[i] for i in range(len(Omega))]) # shape = (len(Omega), len(dL))

        def minimize(i):
            jsd_list = [jensenshannon(p_m_p_z(None, m[j], z[j]) / denominator[j], pdf_figaro[i]) for j in range(len(Omega))] # shape = len(Omega)
            return H0[np.argmin(jsd_list)]

    else:
        print("Invalid argument!")
        sys.exit(1)

    def minimize_and_save(i):
        result = minimize(i)
        np.save(outdir/f'checkpoints/{param}_{method}_{mz}_{str(i)}', result)
        return result

    print("Preparing inference...")
    pdf_figaro = np.array([draw.condition([mz], [0]).pdf(dL) for draw in draws]) # shape = (len(draws), len(dL))

    remaining = list(range(len(pdf_figaro)))
    if not os.path.exists(outdir/'checkpoints'):
        os.makedirs(outdir/'checkpoints')
    for i in range(len(pdf_figaro)):
        if os.path.exists(outdir/f'checkpoints/{param}_{method}_{mz}_{str(i)}.npy'):
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
        if os.path.exists(outdir/f'checkpoints/{param}_{method}_{mz}_{str(i)}.npy'):
            try:
                result.append(np.load(outdir/f'checkpoints/{param}_{method}_{mz}_{str(i)}.npy'))
            except EOFError:
                result.append(minimize_and_save(i))
    result = np.array(result)

    print("Saving results...")
    if not os.path.exists(outdir/'multi'):
        os.makedirs(outdir/'multi')
    np.savez(outdir/f"multi/{param}_{method}_{mz}.npz", result=result, pdf_figaro=pdf_figaro)

print("Removing checkpoints...")
if os.path.exists(outdir/'checkpoints'):
    for filename in os.listdir(outdir/'checkpoints'):
        if filename.startswith(f"{param}_{method}_{mz}_"):
            os.remove(outdir/f"checkpoints/{filename}")

print("Done!")

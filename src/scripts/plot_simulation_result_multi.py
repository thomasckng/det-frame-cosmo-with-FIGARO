import numpy as np
from corner import corner
from figaro.cosmology import Planck18
from figaro import plot_settings
import paths

param_dict = {
    '2a': ['$H_0$', '$\\alpha$'],
    '2b': ['$H_0$', '$\\mu$'],
    '3a': ['$H_0$', '$\\alpha$', '$\\kappa$'],
    '3b': ['$H_0$', '$\\mu$', '$\\kappa$'],
    '4a': ['$H_0$', '$\\alpha$', '$\\mu$', '$\\sigma$'],
    '4b': ['$H_0$', '$\\mu$', '$\\sigma$', '$\\kappa$'],
    '4c': ['$H_0$', '$\\alpha$', '$\\mu$', '$\\kappa$'],
    '5a': ['$H_0$', '$\\alpha$', '$\\mu$', '$\\sigma$', '$w$'],
    '5b': ['$H_0$', '$\\alpha$', '$\\mu$', '$\\sigma$', '$\\kappa$'],
    '6': ['$H_0$', '$\\alpha$', '$\\mu$', '$\\sigma$', '$w$', '$\\kappa$'],
    '9': ['$H_0$', '$\\alpha$', '$\\mu$', '$\\sigma$', '$w$', '$\\delta$', '$m_\mathrm{min}$', '$m_\mathrm{max}$', '$\\kappa$'],
}

for key, parameters in param_dict.items():
    try:
        f = np.load(paths.data / 'simulation/multi' / f'{key}_Powell.npz')
        result = f['result']
        simulated_truth = {'$H_0$': Planck18.h*100, '$\\alpha$': 3.5, '$\\mu$': 35, '$\\sigma$': 5, '$\\delta$': 5, '$w$': 0.2, '$m_\mathrm{min}$': 5, '$m_\mathrm{max}$': 90, '$\\kappa$': 0}

        fig = corner(result,
                    labels=parameters,
                    range= [0.8]*len(parameters),
                    truths=[simulated_truth[param] for param in parameters],
                    color='steelblue',
                    truth_color='red',
                    levels = [0.3935, 0.9],
                    plot_datapoints=False,
                    plot_density=False,
                    smooth=1,
                    fill_contours=True,
                    contourf_kwargs={'colors': ['white', 'darkturquoise', 'mediumturquoise'], 'alpha': [1, 0.2, 0.5]},
                    contour_kwargs={'linewidths': 1}, show_titles=True)
        fig.savefig(paths.figures / f'simulation_result_{key}.pdf', bbox_inches='tight')
        fig.clf()
        f.close()
        print(f"Plot saved for key: {key}.")
    except FileNotFoundError:
        print(f"File not found for key: {key}. Skipping...")
    except Exception as e:
        print(f"An error occurred for key: {key}. Error: {e}")
        try:
            f.close()
        except:
            pass
    print()

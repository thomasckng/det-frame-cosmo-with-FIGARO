import numpy as np
from corner import corner
# from figaro.cosmology import Planck18
from figaro import plot_settings
import matplotlib.pyplot as plt
import paths

param_dict = {
    # '2a': ['$H_0$', '$\\alpha$'],
    # '2b': ['$H_0$', '$\\mu$'],
    '3a': ['$H_0$', '$\\alpha$', '$\\kappa$'],
    # '3b': ['$H_0$', '$\\mu$', '$\\kappa$'],
    # '4a': ['$H_0$', '$\\alpha$', '$\\mu$', '$\\sigma$'],
    # '4b': ['$H_0$', '$\\mu$', '$\\sigma$', '$\\kappa$'],
    # '4c': ['$H_0$', '$\\alpha$', '$\\mu$', '$\\kappa$'],
    # '5a': ['$H_0$', '$\\alpha$', '$\\mu$', '$\\sigma$', '$w$'],
    # '5b': ['$H_0$', '$\\alpha$', '$\\mu$', '$\\sigma$', '$\\kappa$'],
    '5c': ['$H_0$', '$\\alpha_1$', '$\\alpha_2$', '$b$', '$\\kappa$'],
    '6': ['$H_0$', '$\\alpha$', '$\\mu$', '$\\sigma$', '$w$', '$\\kappa$'],
    # '9': ['$H_0$', '$\\alpha$', '$\\mu$', '$\\sigma$', '$w$', '$\\delta$', '$m_\mathrm{min}$', '$m_\mathrm{max}$', '$\\kappa$'],
}

# expected = {'$H_0$': Planck18.h*100, "$\\alpha$": 4.2487241670754035, "$\\mu$": 31.825703461630482, "$\\sigma$": 3.7904971258458042, "$w$": 0.024059947239759398, "$\\delta$": 4.8961636235644015, "$m_\mathrm{min}$": 5.0808821331157095, "$m_\mathrm{max}$": 109.03299036617125, "$\\kappa$": None}
bounds = {'$H_0$': (40, 100), "$\\alpha$": (2, 8), "$\\mu$": (10, 60), "$\\sigma$": (0, 5), "$w$": (0, 1), "$\\delta$": (0, 10), "$m_\mathrm{min}$": (0, 10), "$m_\mathrm{max}$": (70, 100), "$\\kappa$": (-1, 5)}

fig_jsd, ax_jsd = plt.subplots()
colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
for mass_dist in ['PLP', 'PL', 'BPL']:
    for key, parameters in param_dict.items():
        try:
            f = np.load(paths.data / 'real/multi' / f'{key}_Powell.npz')
            result = f['result']
            jsd_samples = f['jsd']

            fig = corner(result,
                        labels=parameters,
                        # range= [0.8]*len(parameters),
                        range=[bounds[param] for param in parameters],
                        # truths=[expected[param] for param in parameters],
                        # truth_color='red',
                        color='steelblue',
                        levels = [0.3935, 0.9],
                        plot_datapoints=False,
                        plot_density=False,
                        smooth=1,
                        fill_contours=True,
                        contourf_kwargs={'colors': ['white', 'darkturquoise', 'mediumturquoise'], 'alpha': [1, 0.2, 0.5]},
                        contour_kwargs={'linewidths': 1},
                        show_titles=True,
                        title_kwargs={'fontsize': 20},
                        )
            fig.savefig(paths.figures / f'real_result_{key}.pdf', bbox_inches='tight')
            fig.clf()
            f.close()
            print(f"Plot saved for key: {key}.")

            ax_jsd.hist(jsd_samples, bins = int(np.sqrt(len(jsd_samples))), histtype = 'step', density = True, color= colors.pop(0), label = f'$\mathrm{{{mass_dist}}}$')
        except FileNotFoundError:
            print(f"File not found for key: {key}. Skipping...")
        except Exception as e:
            print(f"An error occurred for key: {key}. Error: {e}")
            try:
                f.close()
            except:
                pass
        print()

ax_jsd.legend()
ax_jsd.set_xlabel('$d_\mathrm{JS}$')
ax_jsd.set_ylabel('$\mathrm{Density}$')
fig_jsd.savefig(paths.figures / f"real_result_jsd.pdf", bbox_inches='tight')

import h5py
import pandas as pd
import numpy as np
import os
import fnmatch
import sys
import paths
import tqdm

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

n_samples = 10000
seed = 12345

events = [\
'GW150914',
'GW151012',
'GW151226',
'GW170104',
'GW170608',
'GW170729',
'GW170809',
'GW170814',
# 'GW170817', # BNS
'GW170818',
'GW170823',
'GW190408_181802',
'GW190412_053044',
'GW190413_134308',
'GW190421_213856',
# 'GW190425_081805', # BNS
'GW190503_185404',
'GW190512_180714',
'GW190513_205428',
'GW190517_055101',
'GW190519_153544',
'GW190521_030229',
'GW190521_074359',
'GW190527_092055',
'GW190602_175927',
'GW190620_030421',
'GW190630_185205',
'GW190701_203306',
'GW190706_222641',
'GW190707_093326',
'GW190708_232457',
'GW190720_000836',
'GW190727_060333',
'GW190728_064510',
'GW190803_022701',
'GW190814_211039',
'GW190828_063405',
'GW190828_065509',
'GW190910_112807',
'GW190915_235702',
'GW190924_021846',
'GW190925_232845',
'GW190929_012149',
'GW190930_133541',
'GW191105_143521',
'GW191109_010717',
'GW191127_050227',
'GW191129_134029',
'GW191204_171526',
'GW191215_223052',
'GW191216_213338',
'GW191222_033537',
'GW191230_180458',
# 'GW200105_162426', # NSBH
'GW200112_155838',
# 'GW200115_042309', # NSBH
'GW200128_022011',
'GW200129_065458',
'GW200202_154313',
'GW200208_130117',
'GW200209_085452',
'GW200219_094415',
'GW200224_222234',
'GW200225_060421',
'GW200302_015811',
'GW200311_115853',
'GW200316_215756',
'GW190413_052954',
# 'GW190426_152155', # NSBH
'GW190719_215514',
'GW190725_174728',
'GW190731_140936',
'GW190805_211137',
# 'GW190917_114630', # NSBH
'GW191103_012549',
'GW200216_220804',
]

mass_med_min, mass_med_max = 0, 0
mass_min_min, mass_max_max = 0, 0
for event in tqdm.tqdm(events):
    h5_paths = find('*' + str(event) + '*' + 'nocosmo.h5', sys.argv[1])
    if len(h5_paths) == 0:
        print(f"No file found for {event}")
    elif len(h5_paths) > 1:
        print(f"Multiple files found for {event}")
    else:
        path_h5 = h5_paths[0]
        with h5py.File(path_h5, 'r') as f:
            f = f['C01:IMRPhenomXPHM']
            result_lvk = pd.DataFrame.from_records(f["posterior_samples"][()])
            result_lvk = result_lvk.sample(n=n_samples, random_state=seed)
        mass_med_min = min(mass_med_min, np.median(result_lvk['mass_1']))
        mass_med_max = max(mass_med_max, np.median(result_lvk['mass_1']))
        mass_min_min = min(mass_min_min, min(result_lvk['mass_1']))
        mass_max_max = max(mass_max_max, max(result_lvk['mass_1']))
        np.savetxt(paths.data/f'real/data/{event}.txt', result_lvk['mass_1'])
np.savetxt(paths.data/'jsd_bounds.txt', [mass_med_min, mass_med_max])
print(f"Mass sample bounds: {mass_min_min}, {mass_max_max}")

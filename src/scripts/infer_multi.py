import numpy as np
from numpy.random import uniform as uni
from scipy.spatial.distance import jensenshannon as scipy_jsd
from figaro.load import load_density
from figaro.cosmology import CosmologicalParameters
from multiprocessing import Pool
from selection_function import selection_function
from population_models.mass import plpeak
import sys
import os

# Redshift distribution
def p_z(z, H0):
    return CosmologicalParameters(H0/100., 0.315, 0.685, -1., 0., 0.).ComovingVolumeElement(z)/(1+z)

if len(sys.argv) < 5:
    print("Invalid number of arguments!")
    sys.exit(1)

if sys.argv[1] == "2a":
    x0 = [uni(10,200), uni(1.01,5)]
    bounds = ((10,200), (1.01,5))
    def plp(m, x):
        return plpeak(m, alpha=x[0])
elif sys.argv[1] == "2b":
    x0 = [uni(10,200), uni(10,50)]
    bounds = ((10,200), (10,50))
    def plp(m, x):
        return plpeak(m, mu=x[0])
elif sys.argv[1] == "4a":
    x0 = [uni(10,200), uni(10,50), uni(0.01,10), uni(0.01,15)]
    bounds = ((10,200), (10,50), (0.01,10), (0.01,15))
    def plp(m, x):
        return plpeak(m, mu=x[0], sigma=x[1], delta=x[2])
elif sys.argv[1] == "4b":
    x0 = [uni(10,200), uni(1.01,5), uni(10,50), uni(0.01,10)]
    bounds = ((10,200), (1.01,5), (10,50), (0.01,10))
    def plp(m, x):
        return plpeak(m, alpha=x[0], mu=x[1], sigma=x[2])
elif sys.argv[1] == "5a":
    x0 = [uni(10,200), uni(1.01,5), uni(10,50), uni(0.01,10), uni(0,1)]
    bounds = ((10,200), (1.01,5), (10,50), (0.01,10), (0,1))
    def plp(m, x):
        return plpeak(m, alpha=x[0], mu=x[1], sigma=x[2], w=x[3])
elif sys.argv[1] == "5b":
    x0 = [uni(10,200), uni(1.01,5), uni(10,50), uni(1,10), uni(0,1)]
    bounds = ((10,200), (1.01,5), (10,50), (1,10), (0,1))
    def plp(m, x):
        return plpeak(m, alpha=x[0], mu=x[1], mmin=x[2], w=x[3])
elif sys.argv[1] == "6":
    x0 = [uni(10,200), uni(1.01,5), uni(10,50), uni(0.01,10), uni(0,1), uni(0.01,15)]
    bounds = ((10,200), (1.01,5), (10,50), (0.01,10), (0,1), (0.01,15))
    def plp(m, x):
        return plpeak(m, alpha=x[0], mu=x[1], sigma=x[2], w=x[3], delta=x[4])
else:
    print("Invalid argument!")
    sys.exit(1)

def jsd(x, i):
    model_pdf = np.einsum("ij, j -> ij", plp(m, x[1:]), p_z(z, x[0])) # shape = (len(mz), len(z))
    grid = np.transpose(np.meshgrid(mz, CosmologicalParameters(x[0]/100., 0.315, 0.685, -1., 0., 0.).LuminosityDistance(z))) # shape = (len(mz), len(z), 2)
    SE_grid = selection_function(grid) # shape = (len(mz), len(z))
    model_pdf = np.einsum("ij, ij -> ij", model_pdf, SE_grid) # shape = (len(mz), len(z))
    model_pdf = np.trapz(model_pdf, z, axis=1) # shape = (len(mz))
    model_pdf_short = model_pdf[_mask]

    return scipy_jsd(model_pdf_short, pdf_figaro[i])


if sys.argv[2] in ["Powell", "TNC"]:

    method = sys.argv[2]

    from scipy.optimize import minimize as scipy_minimize

    def minimize(i):
        return scipy_minimize(jsd, x0=x0, bounds=bounds, args=(i,), method=method).x
    
elif sys.argv[2] == "CMA-ES":

    import cma

    def minimize(i):
        return cma.fmin2(jsd, x0, 1, {'bounds': np.array(bounds).T.tolist(), 'CMA_stds': np.array(bounds).T[1]/4}, args=(i,))[0]


else:
    print("Invalid argument!")
    sys.exit(1)

def minimize_and_save(i):
    np.save(outdir+'/checkpoints/'+sys.argv[1]+'_'+sys.argv[2]+'_'+str(i), minimize(i))
    return i

n_pool = int(sys.argv[4])

label = sys.argv[3]
outdir = os.path.dirname(os.path.realpath(__file__)) + "/" + label

mz = np.linspace(1,200,900)
H0 = np.linspace(5,150,1000)
z = np.linspace(0.001,2,800)
m = np.einsum("i, j -> ij", mz, np.reciprocal(1+z)) # shape = (len(mz), len(z))

print("Reading bounds and draws...")
draws = load_density(outdir+"/draws/draws_observed_"+label+".json")

jsd_bounds = np.loadtxt(outdir+"/jsd_bounds.txt")

print("Preparing inference...")
# Mask out mz where there is no sample
_mask = [mz[k] <= jsd_bounds[1] and mz[k] >= jsd_bounds[0] for k in range(len(mz))]
mz_short = mz[_mask]

pdf_figaro = np.array([draw.pdf(mz_short) for draw in draws])# shape (n_draws, len(mz_short))

if not os.path.exists(outdir+'multi/'+sys.argv[1]+'_'+sys.argv[2]+'.npz'):
    remaining = list(range(len(pdf_figaro)))
    if not os.path.exists(outdir+'/checkpoints'):
        os.makedirs(outdir+'/checkpoints')
    for i in range(len(pdf_figaro)):
        if os.path.exists(outdir+'/checkpoints/'+sys.argv[1]+'_'+sys.argv[2]+'_'+str(i)+'.npy'):
            remaining.remove(i)
    print("Remaining number of draws: "+str(len(remaining)))

    print("Starting inference...")
    with Pool(n_pool) as p:
        p.map(minimize_and_save, remaining)

    print("Collecting results...")
    result = []
    for i in range(len(pdf_figaro)):
        if os.path.exists(outdir+'/checkpoints/'+sys.argv[1]+'_'+sys.argv[2]+'_'+str(i)+'.npy'):
            result.append(np.load(outdir+'/checkpoints/'+sys.argv[1]+'_'+sys.argv[2]+'_'+str(i)+'.npy'))
    result = np.array(result)

    print("Saving results...")
    if not os.path.exists(outdir+'/multi'):
        os.makedirs(outdir+'/multi')
    np.savez(outdir+"/multi/"+sys.argv[1]+"_"+sys.argv[2]+".npz", result=result, pdf_figaro=pdf_figaro)

print("Removing checkpoints...")
for i in range(len(pdf_figaro)):
    if os.path.exists(outdir+'/checkpoints/'+sys.argv[1]+'_'+sys.argv[2]+'_'+str(i)+'.npy'):
        os.remove(outdir+'/checkpoints/'+sys.argv[1]+'_'+sys.argv[2]+'_'+str(i)+'.npy')

print("Done!")
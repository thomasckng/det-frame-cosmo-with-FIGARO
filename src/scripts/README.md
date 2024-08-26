After installing [FIGARO](https://github.com/sterinaldi/FIGARO), reconstruction of the simulation can be done using the following command:
```
figaro-par-hierarchical --config reconstruction_options_simulation.ini
```
Use `figaro-hierarchical` instead of `figaro-par-hierarchical` if you want to run the reconstruction without parallelization.

To get samples from LVK PE results, run the following python script:
```
python get_lvk_samples.py outdir
```
where `outdir` is the directory where the PE results in the form of `*.h5` files are stored.

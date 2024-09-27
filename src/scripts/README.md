To generate results in the paper, first install the required packages listed in [`environment.yml`](/environment.yml).

Then, run the following command to generate the selection function for the simulation:
```
python generate_selection_function.py
```

Next, run the following command to generate the simulated data:
```
python generate_simulated_data.py
```

After installing [FIGARO](https://github.com/sterinaldi/FIGARO), reconstruction of the simulation can be done using the following command:
```
figaro-par-hierarchical --config reconstruction_options_simulation.ini
```
Use `figaro-hierarchical` instead of `figaro-par-hierarchical` if you want to run the reconstruction without parallelization.

To infer $H_0$ from the simulation results, run the following command:
```
python infer_H0.py simulation
```

To infer multiple parameters instead, run the following command:
```
python infer_multi.py <method> <parameters> simulation <n_parallel>
```
where `<method>` is the method to be used for inference, `<parameters>` is the index for which parameters to infer, and `<n_parallel>` is the number of parallel processes to use.
Check the script for more details.

For running the analysis on the LVK data, first store all single-event PE results of [GWTC-2.1](https://zenodo.org/records/6513631) and [GWTC-3](https://zenodo.org/records/8177023) in a directory.
Then, to get samples from LVK PE results, run the following command:
```
python get_lvk_samples.py outdir
```
where `outdir` is the path to the directory containing the PE results in the format of `*.h5`.

For the reconstruction and inference, run the above commands with the appropriate configuration files and options.

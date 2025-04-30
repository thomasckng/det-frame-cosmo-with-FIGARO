To generate results in the paper, first install the required packages listed in [`environment.yml`](/environment.yml).

Next, run the following command to generate the simulated data:
```
python generate_posterior_samples.py
```

After installing [FIGARO](https://github.com/sterinaldi/FIGARO), reconstruction of the simulation can be done using the following command:
```
figaro-par-hierarchical --config reconstruction_options_simulation.ini
```
Use `figaro-hierarchical` instead of `figaro-par-hierarchical` if you want to run the reconstruction without parallelization.

To infer $H_0$ from the simulation results, run the following command:
```
python inference.py 1 Grid simulation <n_parallel> MDC_10
```

To infer multiple parameters instead, run the following command:
```
python inference.py <method> <parameters> simulation <n_parallel> MDC_10
```
where `<method>` is the method to be used for inference, `<parameters>` is the index for which parameters to infer, and `<n_parallel>` is the number of parallel processes to use.
Check the script for more details.

For running the analysis on the LVK data, first download all single-event PE results and O3 search sensitivity estimates by running `download_lvk_result.sh`.

For the reconstruction and inference, run the above commands with the appropriate configuration files and options.

rule get_number_of_simulated_events:
    input:
        "src/data/simulation/true_samples.txt",
        "src/data/simulation/obs_samples.txt"
    output:
        "src/tex/output/n_true_samples.txt",
        "src/tex/output/n_obs_samples.txt"
    script:
        "src/scripts/get_number_of_simulated_events.py"

rule get_number_of_simulated_events:
    input:
        "src/data/simulation/true_samples.txt",
        "src/data/simulation/obs_samples.txt"
    output:
        "src/tex/output/n_true_samples.txt",
        "src/tex/output/n_obs_samples.txt"
    script:
        "src/scripts/get_number_of_simulated_events.py"

rule compute_real_H0:
    input:
        "src/data/real/multi/4d_Powell.npz"
    output:
        "src/tex/output/real_H0.txt"
    script:
        "src/scripts/get_real_H0.py"

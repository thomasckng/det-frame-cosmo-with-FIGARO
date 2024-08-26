rule compute_Gaussian_result:
    input:
        "src/data/simulation/true_samples.txt",
        "src/data/simulation/obs_samples.txt"
    output:
        "src/tex/output/n_true_samples.txt",
        "src/tex/output/n_obs_samples.txt"
    script:
        "src/scripts/get_n_samples.py"

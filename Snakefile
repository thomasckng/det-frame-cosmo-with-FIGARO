rule compute_real_H0:
    input:
        "src/data/real/multi/4c_Powell.npz"
    output:
        "src/tex/output/real_H0.txt"
    script:
        "src/scripts/get_real_H0.py"

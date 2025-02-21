import numpy as np

def compute_rademacher_bound(n):
    # Define the neural network architecture
    input_dim = 2
    hidden_dim = 32
    output_dim = 3

    params_L1 = input_dim * hidden_dim + hidden_dim
    params_L2 = hidden_dim * hidden_dim + hidden_dim
    params_Lout = hidden_dim * output_dim + output_dim

    m = params_L1 + params_L2 + params_Lout  # the total number of parameters

    # Assume that the Spectral Norm is 2 (by experience)
    L = 2 * 2  # 2-layer ReLU

    # Calculate Rademacher complexity
    Rademacher_bound = (L * np.sqrt(m) * np.log(n)) / np.sqrt(n)

    return Rademacher_bound

# Rademacher complexity with different sample sizes n
for n in [10, 100, 1000, 10000]:
    print(f"n = {n}, Estimated Rademacher Complexity: {compute_rademacher_bound(n):.2f}")

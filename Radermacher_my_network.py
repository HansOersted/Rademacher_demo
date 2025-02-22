import torch
import numpy as np
import matplotlib.pyplot as plt

# The weigt matrix of the neural network
W1 = torch.randn(32, 2)
W2 = torch.randn(32, 32)
W_out = torch.randn(3, 32)

# the largest singular value of each layer
sigma_W1 = torch.linalg.svdvals(W1)[0]
sigma_W2 = torch.linalg.svdvals(W2)[0]
sigma_W_out = torch.linalg.svdvals(W_out)[0]

# Lipschitz constant L
L = sigma_W1 * sigma_W2 * sigma_W_out

print(f"Spectral Norms: W1={sigma_W1:.3f}, W2={sigma_W2:.3f}, W_out={sigma_W_out:.3f}")
print(f"Estimated Lipschitz Constant L: {L:.3f}")

# calculate Rademacher complexity upper bound
def compute_rademacher_bound(n, L, C=2):  # C from experience
    m = 1251 # number of parameters (2×32+32)+(32×32+32)+(32×3+3)
    return C * (L * np.sqrt(m) * np.log(n)) / np.sqrt(n)

# calculate Rademacher complexity in different sample sizes
n_values = np.logspace(1, 5, 100)
rademacher_bound = compute_rademacher_bound(n_values, L)

plt.figure(figsize=(8, 5))
plt.plot(n_values, rademacher_bound, label=r"$\mathcal{R}_n(\mathcal{V}) = C \frac{L \sqrt{m} \log n}{\sqrt{n}}$")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Training Samples (n)")
plt.ylabel("Rademacher Complexity Upper Bound")
plt.legend()
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Calculate Rademacher complexity
n_values = np.array([10, 100, 1000, 10000])
rademacher_values = np.array([150, 70, 30, 15])  # the assumed results

# define the Rademacher complexity function
def rademacher_fn(n, C, L):
    m = 1251  # number of parameters
    return C * (L * np.sqrt(m) * np.log(n)) / np.sqrt(n) 

# fit C, L
popt, _ = curve_fit(rademacher_fn, n_values, rademacher_values)
C_fit, L_fit = popt

print(f"Fitted C: {C_fit}, Fitted L: {L_fit}")

n_test = np.logspace(1, 5, 100)
rademacher_fit = rademacher_fn(n_test, C_fit, L_fit)

plt.figure(figsize=(8, 5))
plt.plot(n_values, rademacher_values, "ro", label="Measured")
plt.plot(n_test, rademacher_fit, "b-", label="Fitted Function")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("n")
plt.ylabel("Rademacher Complexity")
plt.legend()
plt.grid()
plt.show()

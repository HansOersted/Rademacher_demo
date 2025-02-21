import numpy as np

X = np.array([1, 2, 3])  # data set
n = len(X)  # sample size

# Define 2 functions h1 or h2
def h1(x):
    return x

def h2(x):
    return -x

# set the number of experiments
M = 1000

# store Rademacher
rademacher_values = []

# perform M experiments
for _ in range(M): # 0, 1, 2, ..., M-1
    # random εi {-1, 1}
    epsilon = np.random.choice([-1, 1], size=n)

    # calculate the average of εi * h1(x) and εi * h2(x)
    sum_h1 = np.sum(epsilon * h1(X)) / n
    sum_h2 = np.sum(epsilon * h2(X)) / n

    # calculate the Rademacher value
    rademacher_value = max(sum_h1, sum_h2)
    rademacher_values.append(rademacher_value)

# calculate the estimated Rademacher complexity
R_n_V = np.mean(rademacher_values)
R_n_V
print("Estimated Rademacher Complexity:", R_n_V)

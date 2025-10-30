import numpy as np

from clinometer import estimate_slope

# Number of samples
N = 100
# Independent variable
x = np.linspace(0, 10, N)
# Dependent variable (noisy)
y = np.sin(x) + 0.05 * np.random.randn(N)
# True gradient
dy_dx_true = np.cos(x)

# Acceptance threshold
# This value should be proportional to the expected level of noise.
# Higher values give more robust gradients at the expense of a lower
# sensitivity (smearing in time/space).
e = 0.1

# The maximum size of the search window.
Nmax = 10

# Fit a straight line (poly=0) or a 2nd-order polynomial (poly=1).
# Usually a straight line is good enough
poly = 0

# Estimate the noisy gradient
# This returns a NumPy array with the following rows:
# [0] = estimated slope
# [1] = window size
# [2] = |Δy| (should be close to e)
# [3] = convergence flag
result = estimate_slope(y, x, e=e, Nmax=Nmax, poly=poly)
dy_dx = result[0]
print(dy_dx)
print(dy_dx_true)

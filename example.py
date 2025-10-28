import numpy as np

from clinometer import estimate_slope

N = 100
x = np.linspace(0, 10, N)
y = np.sin(x) + 0.05 * np.random.randn(N)
dy_dx_true = np.cos(x)

dy_dx = estimate_slope(y, x, e=0.1, Nmax=10, poly=0)
print(dy_dx[0])
print(dy_dx_true)

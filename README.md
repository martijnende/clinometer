# Clinometer: noisy gradient estimation

There are many situations in which one might want to know the derivative (gradient) of some measurements, for example to convert a displacement measurement to velocity. When the measured data are noisy (which is always the case), most conventional gradient estimators, like [finite-differences](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html) or [Savitzky-Golay](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html), tend to fail, especially when the gradient itself spans orders of magnitude.

_Clinometer_ is a small Python/Cython package that estimates the gradient around a given data point using a growing window method: for a given data point $x_i$, it defines a window $2w$ that encompases all data from $x_{i-w}$ to $x_{i+w}$. If the absolute difference $x_{i+w} - x_{i-w}$ exceeds a user-defined threshold $e$, the gradient is estimated through linear regression on the data contained in this window. If the the threshold is not exceeded, $w$ is incremented until the threshold is met.

What this means in practice, is that when the gradient is small compared to the noise, a large window is used to estimate the gradient, whereas if the gradient is large compared to the noise, a small window is used. Hence, the estimated gradient is guaranteed to be significant.

Note that the sampling of the independent variable (usually time or space) does not need to be regular; _clinometer_ automatically deals with changes in the recording rate.

## Installation

Pull the repository to your local system and enter the directory:
```bash
git clone https://github.com/martijnende/clinometer
cd clinometer
```
Activate the desired Python environment (e.g. with conda) and install the package:
```bash
pip install -e .
```
or compile it manually:
```bash
python setup.py build_ext --inplace
```

## Example usage

```python
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
```
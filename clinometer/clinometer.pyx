#cython: boundscheck=False
#cython: wraparound=False
#cython: profile=False
#cython: cdivision=True
#cython: infer_types=True

""" If you want to profile this algorithm in more detail, set #cython: profile=True above  """

import numpy as np
from libc.math cimport fabs

# ============================================================
#  find_window_size()
# ============================================================
def find_window_size(double[:] data,
                     unsigned int n,
                     unsigned int i,
                     double e,
                     unsigned int N,
                     unsigned int Nmax):
    """
    Expands or shrinks the window size for each data point 
    until the difference in y-data ( |y[right] - y[left]| ) 
    matches the provided error. The window size is then used by
    the next data point as a starting point. If the window has 
    to grow outside the imposed limit of Nmax (i.e.
    |y[n+Nmax] - y[n-Nmax]| < error), or if the algorithm gets
    stuck in an infinite loop (e.g. with discontinuous data), 
    the function breaks and flags that datapoint as 
    insignificant (converged = 0).
    """
    cdef unsigned int converged = 1
    cdef unsigned int leftlim, leftlim2, rightlim, rightlim2
    cdef unsigned int counter = 0
    cdef unsigned int stepsize = 1
    cdef int[::1] out = np.zeros(4, dtype=np.int32)

    # Looping until the difference > error
    while True:
        # Infinite loop check
        counter += 1
        if counter > 2 * Nmax:
            converged = 0
            break

        # Adjust window bounds
        leftlim = n - i
        leftlim2 = n - i - stepsize
        rightlim = n + i
        rightlim2 = n + i + stepsize

        # Bounds checking
        if i > n:
            leftlim = 0
            rightlim = rightlim - n + i
        if i + stepsize > n:
            leftlim2 = 0
            rightlim2 = rightlim2 - n + i + stepsize
        if n + i >= N - 1:
            rightlim = N - 1
            leftlim = leftlim - n - i + N - 1
        if n + i + stepsize >= N - 1:
            rightlim2 = N - 1
            leftlim2 = leftlim2 - n - i + N - 1

        # Growing or shrinking window
        if fabs(data[leftlim] - data[rightlim]) < e:
            # Is the window too small or just right?
            if fabs(data[leftlim2] - data[rightlim2]) >= e:
                break

            # |y[n+Nmax] - y[n-Nmax]| < error → stop
            if i > Nmax:
                converged = 0
                break

            # Increase window size
            i += stepsize
        else:
            # Is the window too large or just right?
            if fabs(data[leftlim2] - data[rightlim2]) <= e:
                break

            # Need at least 3 points
            if i <= stepsize:
                break

            # Decrease window size
            i -= stepsize

    # Store results
    out[0] = i
    out[1] = leftlim2
    out[2] = rightlim2
    out[3] = converged
    return np.asarray(out)

# ============================================================
#  scan_regression()
# ============================================================

def scan_regression(double[:, :] r_params, int poly):
    """
    Performs OLS regression to estimate the gradient in each 
    data point using the regression parameters that were 
    gathered in the main function. This exploits the property 
    that sum[a -> b] = sum[0 -> b] - sum[0 -> a-1]. The entire 
    operation is very similar to a scan operation common in
    GPU/parallel programming.
    """
    cdef Py_ssize_t n
    cdef Py_ssize_t N = r_params.shape[0]
    cdef double[::1] gradient = np.empty(N, dtype=np.float64)

    cdef int leftlim, rightlim, l
    cdef double sumx, sumy, xx, xxx, xxxx, xy, xxy, a, b, x
    cdef object K, B, A, X

    for n in range(N):
        leftlim = <int>r_params[n, 4]
        rightlim = <int>r_params[n, 5]
        l = rightlim - leftlim + 1
        x = r_params[n, 9]

        # Collect regression parameters
        if leftlim > 0:
            sumx = r_params[rightlim, 0] - r_params[leftlim - 1, 0]
            sumy = r_params[rightlim, 1] - r_params[leftlim - 1, 1]
            xx = r_params[rightlim, 2] - r_params[leftlim - 1, 2]
            xy = r_params[rightlim, 3] - r_params[leftlim - 1, 3]
            xxx = r_params[rightlim, 6] - r_params[leftlim - 1, 6]
            xxxx = r_params[rightlim, 7] - r_params[leftlim - 1, 7]
            xxy = r_params[rightlim, 8] - r_params[leftlim - 1, 8]
        else:
            sumx = r_params[rightlim, 0]
            sumy = r_params[rightlim, 1]
            xx = r_params[rightlim, 2]
            xy = r_params[rightlim, 3]
            xxx = r_params[rightlim, 6]
            xxxx = r_params[rightlim, 7]
            xxy = r_params[rightlim, 8]

        if poly == 0:
            a = l * xy - (sumx * sumy)
            b = l * xx - (sumx * sumx)
            gradient[n] = 0.0 if b == 0.0 else a / b
        else:
            # Small matrices, so np.linalg.inv is acceptable
            K = np.linalg.inv(np.array([[l, sumx, xx],
                                        [sumx, xx, xxx],
                                        [xx, xxx, xxxx]], dtype=np.float64))
            B = np.array([sumy, xy, xxy], dtype=np.float64)
            A = np.dot(K, B)
            X = np.array([0.0, 1.0, 2.0 * x], dtype=np.float64)
            gradient[n] = np.dot(A, X)

    return np.asarray(gradient)


# ============================================================
#  estimate_slope()
# ============================================================

def estimate_slope(double[:] data,
                   double[:] time,
                   double e,
                   unsigned int Nmax,
                   int poly):
    """
    Estimate the local slope of `data` over `time` using an adaptive window.

    Parameters
    ----------
    data : 1D array of float64
        The dependent variable (y).
    time : 1D array of float64
        The independent variable (x).
    e : float
        Error tolerance for window expansion.
    Nmax : int
        Maximum allowed half-window size.
    poly : {0, 1}
        0 for linear regression, 1 for quadratic. 

    Returns
    -------
    np.ndarray (4, N)
        Rows:
        [0] = estimated slope
        [1] = window size
        [2] = |Δy|
        [3] = convergence flag (1 or 0)
    """
    cdef Py_ssize_t n, j
    cdef unsigned int i = 1, leftlim, rightlim, converged, diff
    cdef unsigned int N = data.shape[0]
    cdef double x, y

    cdef int[::1] win
    cdef double[::1] gradient
    cdef double[:, :] r_params = np.zeros((N, 10), dtype=np.float64)
    cdef double[:, :] out = np.zeros((N, 4), dtype=np.float64)

    # Loop over all data points
    for n in range(N):
        # Window properties
        win = find_window_size(data, n, i, e, N, Nmax)
        i = win[0]
        leftlim = win[1]
        rightlim = win[2]
        converged = win[3]

        # Cumulative sums for regression
        x = time[n]
        y = data[n]
        j = n - 1 if n > 0 else 0

        r_params[n, 0] = r_params[j, 0] + x
        r_params[n, 1] = r_params[j, 1] + y
        r_params[n, 2] = r_params[j, 2] + x * x
        r_params[n, 3] = r_params[j, 3] + x * y

        if poly:
            r_params[n, 6] = r_params[j, 6] + x ** 3
            r_params[n, 7] = r_params[j, 7] + x ** 4
            r_params[n, 8] = r_params[j, 8] + x * x * y
            r_params[n, 9] = x

        r_params[n, 4] = leftlim
        r_params[n, 5] = rightlim

        # Store window size, delta y, and convergence flag
        diff = rightlim - leftlim + 1
        out[n, 1] = diff
        out[n, 2] = fabs(data[leftlim] - data[rightlim])
        out[n, 3] = converged

    # Regression scan
    gradient = scan_regression(r_params, poly)
    out[:, 0] = gradient[:]
    return np.asarray(out).T
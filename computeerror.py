import numpy as np
from scipy.interpolate import interp1d
from .linesourceexact import getsol
from scipy.integrate.quadpack import dblquad


def make2d(n, type="middle"):
    # Creates the reference solution on a 2d grid of sice nxn. No ghostcells

    tmp = getsol()
    r = np.append(tmp[:, 0], 4)
    rho = np.append(tmp[:, 1], 0)
    interpolant = interp1d(r, rho, assume_sorted=True)

    def f(x, y):
        xyradius = np.sqrt(x ** 2 + y ** 2)
        z = interpolant(xyradius)
        return z

    rho2d = np.zeros((n, n))
    x = np.linspace(-1.5, 1.5, n + 1)  # edges
    y = np.linspace(-1.5, 1.5, n + 1)  # edges

    for i in range(n):
        for j in range(n):
            xleft, xright, yleft, yright = x[i], x[i + 1], y[j], y[j + 1]
            if type == "interp":
                f1 = f(xleft, yleft)
                f2 = f(xleft, yright)
                f3 = f(xright, yleft)
                f4 = f(xright, yright)
                value = (f1 + f2 + f3 + f4) / 4
            if type == "middle":
                value = f((xright + xleft) / 2, (yright + yleft) / 2)

            if type == "integral":
                value, error = dblquad(f, xleft, xright,
                                       lambda x: yleft, lambda x: yright,
                                       epsabs=1e-3)
                value /= ((xright - xleft) * (yright - yleft))
            rho2d[i, j] = value
    return rho2d

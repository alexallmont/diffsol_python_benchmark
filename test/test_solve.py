from pybamm_diffsol import PyDiffsol
import numpy as np


def test_solve():
    model = PyDiffsol(
        """
        in = [r, k]
        r { 1 } k { 1 }
        u_i { y = 0.1 }
        F_i { (r * y) * (1 - (y / k)) }
    """
    )
    t_eval = np.array([0.0, 1.0])
    t_interp = np.linspace(0.0, 1.0, 100)
    k = 1.0
    r = 1.0
    y0 = 0.1
    y = model.solve(np.array([r, k]), t_interp, t_eval)
    soln = k / (1.0 + (k - y0) * np.exp(-r * t_interp) / y0)
    np.testing.assert_allclose(y[0], soln, rtol=1e-5)


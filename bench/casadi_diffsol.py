import casadi
import timeit
from diffsol_python_benchmark import PyDiffsolDense
import numpy as np

x = casadi.MX.sym("x", 2)
# Two states

# Expression for ODE right-hand side
z = 1 - x[1] ** 2
rhs = casadi.vertcat(z * x[0] - x[1], x[0])

ode = {}  # ODE declaration
ode["x"] = x  # states
ode["ode"] = rhs  # right-hand side

# Construct a Function that integrates over 4s
F = casadi.integrator("F", "cvodes", ode, 0, 4, {"abstol": 1e-6, "reltol": 1e-4})

def casadi_bench():
    F(x0=[0, 1])

n = 1000
casadi_time = timeit.timeit(casadi_bench, number=n) / n
print("Casadi time: ", casadi_time)

model = PyDiffsolDense(
    """
    u_i { x0 = 0.0, x1 = 1.0 }
    F_i { (1 - x1*x1) * x0 - x1, x0 }
    """, 1e-4, 1e-6
)
t_eval = np.array([0.0, 4.0])
t_interp = np.array([0.0, 4.0])

def diffsol_bench():
    model.solve(np.array([]), t_interp, t_eval)

diffsol_time = timeit.timeit(diffsol_bench, number=n) / n
print("Diffsol time: ", diffsol_time)
print("Speedup: ", casadi_time / diffsol_time)

casadi_soln = F(x0=[0, 1])["xf"]
diffsol_soln = model.solve(np.array([]), t_interp, t_eval)[:, -1]
print("Casadi solution: ", casadi_soln)
print("Diffsol solution: ", diffsol_soln)

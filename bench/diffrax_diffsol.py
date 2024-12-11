import timeit
from diffsol_python_benchmark import PyDiffsolDense
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


class Robertson(eqx.Module):
    k1: float
    k2: float
    k3: float

    def __call__(self, t, y, args):
        f0 = -self.k1 * y[0] + self.k3 * y[1] * y[2]
        f1 = self.k1 * y[0] - self.k2 * y[1] ** 2 - self.k3 * y[1] * y[2]
        f2 = self.k2 * y[1] ** 2
        return jnp.stack([f0, f1, f2])

@jax.jit
def main():
    k1 = 0.04
    k2 = 3e7
    k3 = 1e4
    robertson = Robertson(k1, k2, k3)
    terms = diffrax.ODETerm(robertson)
    t0 = 0.0
    t1 = 100.0
    y0 = jnp.array([1.0, 0.0, 0.0])
    dt0 = 0.0002
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=jnp.array([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]))
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    return sol

main()

def diffrax_bench():
    main()

n = 1000
diffrax_time = timeit.timeit(diffrax_bench, number=n) / n
print("Diffrax time: ", diffrax_time)

model = PyDiffsolDense(
    """
    k1 { 0.04 }
    k2 { 30000000 }
    k3 { 10000 }
    u_i {
        x = 1,
        y = 0,
        z = 0,
    }
    F_i {
        -k1 * x + k3 * y * z,
        k1 * x - k2 * y * y - k3 * y * z,
        k2 * y * y,
    }
    """, 1e-8, 1e-8
)
t_eval = np.array([0.0, 100.0])
t_interp = np.array([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])

def diffsol_bench():
    model.solve(np.array([]), t_interp, t_eval)

diffsol_time = timeit.timeit(diffsol_bench, number=n) / n
print("Diffsol time: ", diffsol_time)
print("Speedup: ", diffrax_time / diffsol_time)

diffrax_soln = main().ys[-1]
diffsol_soln = model.solve(np.array([]), t_interp, t_eval)[:, -1]
print("Diffrax solution: ", diffrax_soln)
print("Diffsol solution: ", diffsol_soln)


import timeit
import casadi
import diffsol_python_benchmark.diffrax_models as diffrax_models
import diffsol_python_benchmark.casadi_models as casadi_models
import diffsol_python_benchmark.diffsol_models as diffsol_models
from diffsol_python_benchmark import PyDiffsolDense, PyDiffsolSparse
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
import numpy as np
import os

os.environ[
    "XLA_FLAGS"
] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_force_host_platform_device_count=1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREAD"] = "1"


ngroups = [1, 10, 20, 50, 100, 1000, 10000]
tols = [1e-4]
t_interp = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]

## setup diffrax benchmark
jax.config.update("jax_enable_x64", True)

diffrax_fns = []

for ng in ngroups:
    for tol in tols:

        class RobertsonOde(eqx.Module):
            ngroups: int

            def __call__(self, t, y, args):
                k1 = 0.04
                k2 = 30000000.0
                k3 = 10000.0

                xs = slice(0, self.ngroups)
                ys = slice(self.ngroups, 2 * self.ngroups)
                zs = slice(2 * self.ngroups, 3 * self.ngroups)
                f0 = -k1 * y[xs] + k3 * y[ys] * y[zs]
                f1 = k1 * y[xs] - k2 * y[ys] ** 2 - k3 * y[ys] * y[zs]
                f2 = k2 * y[ys] ** 2
                return jnp.vstack([f0, f1, f2]).flatten()



        #diffrax
        @jax.jit
        def main():
            robertson = RobertsonOde(ngroups=ng)
            terms = diffrax.ODETerm(robertson)
            t0 = 0.0
            t1 = 100.0
            y0 = jnp.zeros(3 * ng)
            jax.lax.dynamic_update_slice(y0, jnp.array([1.0] * ng), (0,))
            dt0 = 0.0002
            solver = diffrax.Kvaerno5()
            saveat = diffrax.SaveAt(ts=jnp.array(t_interp))
            stepsize_controller = diffrax.PIDController(rtol=tol, atol=tol)
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
            return sol.ys[-1]

        main()

        # casadi
        robertson = casadi_models.robertson_ode(ngroups=ng)

        # Construct a Function that integrates over 4s
        F = casadi.integrator(
            "F", "cvodes", robertson, 0, 100.0, {"abstol": tol, "reltol": tol}
        )

        x0 = np.zeros(3 * ng)
        x0[:ng] = 1.0

        def casadi_bench():
            F(x0=x0)["xf"]

        # diffsol
        robertson = diffsol_models.robertson_ode(ngroups=ng)
        if ng < 20:
            model = PyDiffsolDense(robertson, tol, tol)
        else:
            model = PyDiffsolSparse(robertson, tol, tol)
        t_eval = np.array([0.0, 100.0])
        t_interp_np = np.array(t_interp)

        def diffsol_bench():
            model.solve(np.array([]), t_interp_np, t_eval)[:, -1]

        run_diffrax = ng <= 100
        n = 1000 // (int(0.01 * ng) + 1)
        print("ngroups: ", ng)
        print("tol: ", tol) 
        print("n: ", n)
        if run_diffrax:
            diffrax_time = timeit.timeit(main, number=n) / n
            print("Diffrax time: ", diffrax_time)
        casadi_time = timeit.timeit(casadi_bench, number=n) / n
        print("Casadi time: ", casadi_time)
        diffsol_time = timeit.timeit(diffsol_bench, number=n) / n
        print("Diffsol time: ", diffsol_time)
        print("Speedup over casadi: ", casadi_time / diffsol_time)
        if run_diffrax:
            print("Speedup over diffrax: ", diffrax_time / diffsol_time)
        print()

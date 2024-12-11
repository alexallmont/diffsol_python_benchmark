# DiffSol Python Benchmark

This package provides a set of benchmarks to compare the performance of different solvers in Python against the [DiffSol](https://github.com/martinjrobins/diffsol) solver.

## Installation

To install the package from PyPI, run the following command:

```bash
pip install diffsol_python_benchmark
```

## Usage

### Casadi

TODO

### Scipy

TODO

### Diffrax

TODO

### PyBaMM

This is a demonstration of how to use [DiffSol](https://github.com/martinjrobins/diffsol) to solve a [PyBaMM](https://github.com/pybamm-team/PyBaMM/) model.

Use the following code to solve an PyBaMM SPM model using DiffSol and PyBaMM and compare the timings. You can change the outputs and inputs to any other parameters/variables in the model.

Note that this is only a demonstration and is only tested with the SPM model. It may not work with other models.

```python
from pybamm_diffsol import PybammDiffsol, Pybamm2Diffsl
import numpy as np
import timeit
import pybamm

outputs = ["Voltage [V]"]
inputs = ["Current function [A]"]

# read model from spm.ds file to a string
model_str = Pybamm2Diffsl(pybamm.lithium_ion.SPM()).to_str(inputs, outputs)
model = PybammDiffsol(model_str)
t_eval = np.array([0.0, 3600.0])
t_interp = np.linspace(0.0, 3600.0, 100)
params = np.array([1.0])
n = 1000


def diffsol_bench():
    model.solve(params, t_interp, t_eval)


diffsol_time = timeit.timeit(diffsol_bench, number=n) / n
print("Diffsol time: ", diffsol_time)

# solver pybamm spm model
spm = pybamm.lithium_ion.SPM()
t_eval = np.array([0.0, 3600.0])
t_interp = np.linspace(0.0, 3600.0, 100)
params = spm.default_parameter_values
for inpt in inputs:
    params[inpt] = "[input]"
geometry = spm.default_geometry

params.process_model(spm)
params.process_geometry(geometry)
mesh = pybamm.Mesh(geometry, spm.default_submesh_types, spm.default_var_pts)
disc = pybamm.Discretisation(mesh, spm.default_spatial_methods)
disc.process_model(spm)
solver = pybamm.IDAKLUSolver()
inputs = {inpt: 1.0 for inpt in inputs}
solver.solve(spm, t_eval=t_eval, inputs=inputs)


def pybamm_bench():
    sol = solver.solve(spm, t_eval=t_eval, inputs=inputs)
    # force evalulation of the outputs
    for output in outputs:
        sol[output].data[0]


pybamm_time = timeit.timeit(pybamm_bench, number=n) / n
print("Pybamm time: ", pybamm_time)
print("Speedup: ", pybamm_time / diffsol_time)
```


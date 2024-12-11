use diffsol::{error::DiffsolError, matrix::MatrixRef, vector::{Vector, VectorRef}, BdfState, DefaultDenseMatrix, LinearSolver, Matrix, NonLinearOp, OdeBuilder, OdeEquationsImplicit, OdeSolverMethod, OdeSolverProblem, OdeSolverState, OdeSolverStopReason, Op};
use numpy::{ndarray::{s, Array2}, IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};

type MS = diffsol::SparseColMat<f64>;
type LSS = diffsol::FaerSparseLU<f64>;
type MD = nalgebra::DMatrix<f64>;
type LSD = diffsol::NalgebraLU<f64>;

#[cfg(not(feature = "diffsol-llvm"))]
type CG = diffsol::CraneliftModule;
#[cfg(feature = "diffsol-llvm")]
type CG = diffsol::LlvmModule;

type EqnD = diffsol::DiffSl<MD, CG>;
type EqnS = diffsol::DiffSl<MS, CG>;


struct PyDiffsolError(DiffsolError);

impl From<PyDiffsolError> for PyErr {
    fn from(error: PyDiffsolError) -> Self {
        PyValueError::new_err(error.0.to_string())
    }
}

impl From<DiffsolError> for PyDiffsolError {
    fn from(other: DiffsolError) -> Self {
        Self(other)
    }
}

fn solve<'py, Eqn, M, LS>(problem: &mut OdeSolverProblem<Eqn>, py: Python<'py>, params: PyReadonlyArray1<'py, f64>, t_interp: PyReadonlyArray1<'py, f64>, t_eval: PyReadonlyArray1<'py, f64>, y0: Option<PyReadonlyArray1<'py, f64>>, y0dot: Option<PyReadonlyArray1<'py, f64>>) -> Result<Bound<'py, PyArray2<f64>>, PyDiffsolError> 
where 
    Eqn: OdeEquationsImplicit<M=M, V=M::V, T=M::T>, 
    M: Matrix<T=f64,V: DefaultDenseMatrix>, 
    LS: LinearSolver<M>,
    for<'b> &'b M::V: VectorRef<Eqn::V>,
    for<'b> &'b M: MatrixRef<Eqn::M>,
{
        let t_interp = t_interp.as_array();
        let t_eval = t_eval.as_array();
        let params = params.as_array();
        let mut fparams = M::V::zeros(params.len());
        fparams.copy_from_slice(params.as_slice().unwrap());
        problem.eqn.set_params(&fparams);

        let mut solver = if let Some(y0) = y0 {
            let y0dot = y0dot.unwrap();
            let mut state = BdfState::new_without_initialise(problem)?;
            state.as_mut().y.copy_from_slice(y0.as_slice().unwrap());
            state.as_mut().dy.copy_from_slice(y0dot.as_slice().unwrap());
            state.set_step_size(problem, 1);
            problem.bdf_solver::<LS>(state)?
        } else {
            problem.bdf::<LS>()?
        };

        // check t_interp is sorted
        if !t_interp.iter().is_sorted() {
            return Err(DiffsolError::Other("t_interp must be sorted".to_string()).into());
        }
        // check t_eval is sorted
        if !t_eval.iter().is_sorted() {
            return Err(DiffsolError::Other("t_eval must be sorted".to_string()).into());
        }

        // first element of t_interp and t_eval must be equal to the initial time of the solver
        if t_eval[0] != solver.state().t {
            return Err(DiffsolError::Other("t_eval[0] must be equal to the initial time of the solver".to_string()).into());
        }
        if t_interp[0] != solver.state().t {
            return Err(DiffsolError::Other("t_interp[0] must be equal to the initial time of the solver".to_string()).into());
        }

        // last element of t_interp equal to or less than the last element of t_eval
        if t_interp[t_interp.len() - 1] > t_eval[t_eval.len() - 1] {
            return Err(DiffsolError::Other("last element of t_interp must be equal to or less than the last element of t_eval".to_string()).into());
        }

        let mut out = problem.eqn.out().unwrap().call(solver.state().y, solver.state().t);
        let mut sol = Array2::zeros((out.len(), t_interp.len())); 
        sol.slice_mut(s![.., 0]).iter_mut().zip(out.as_slice().iter()).for_each(|(a, b)| *a = *b);
        let mut next_t_interp = t_interp[1];
        let mut next_t_interp_idx = 1;
        let mut root_or_no_more_t_interp = false;
        for &t in t_eval.iter().skip(1) {
            solver.set_stop_time(t)?;
            let mut finished = false;
            while !finished {
                let curr_t = match solver.step() {
                    Ok(OdeSolverStopReason::InternalTimestep) => solver.state().t,
                    Ok(OdeSolverStopReason::RootFound(root_t)) => {
                        finished = true;
                        root_or_no_more_t_interp = true;
                        root_t
                    },
                    Ok(OdeSolverStopReason::TstopReached) => {
                        finished = true;
                        t
                    },
                    Err(_) => panic!("unexpected solver error"),
                };
                while curr_t >= next_t_interp {
                    let y = solver.interpolate(next_t_interp).unwrap();
                    problem.eqn.out().unwrap().call_inplace(&y, next_t_interp, &mut out);
                    sol.slice_mut(s![.., next_t_interp_idx]).iter_mut().zip(out.as_slice().iter()).for_each(|(a, b)| *a = *b);
                    next_t_interp_idx += 1;
                    if next_t_interp_idx == t_interp.len() {
                        root_or_no_more_t_interp = true;
                        finished = true;
                        break;
                    }
                    next_t_interp = t_interp[next_t_interp_idx];
                }
            }
            if root_or_no_more_t_interp {
                break;
            }
        }
        let sol = if next_t_interp_idx < t_interp.len() {
            sol.slice(s![.., ..next_t_interp_idx]).to_owned()
        } else {
            sol
        };
        Ok(sol.into_pyarray(py))
    }

unsafe impl Send for PyDiffsolDense {}
unsafe impl Sync for PyDiffsolDense {}

#[pyclass]
struct PyDiffsolDense {
    problem: OdeSolverProblem<EqnD>,
}

#[pymethods]
impl PyDiffsolDense {
    #[new]
    fn new(code: &str, rtol: f64, atol: f64) -> Result<Self, PyDiffsolError> {
        let diffsl = EqnD::compile(code)?;
        let nparams = diffsl.nparams();
        let dummy_params = vec![0.0; nparams];
        let problem = OdeBuilder::<MD>::new()
            .p(dummy_params)
            .rtol(rtol)
            .atol([atol])
            .build_from_eqn(diffsl)?;
        Ok(Self { problem })
    }

    #[pyo3(signature = (params, t_interp, t_eval, y0=None, y0dot=None))]
    fn solve<'py>(&mut self, py: Python<'py>, params: PyReadonlyArray1<'py, f64>, t_interp: PyReadonlyArray1<'py, f64>, t_eval: PyReadonlyArray1<'py, f64>, y0: Option<PyReadonlyArray1<'py, f64>>, y0dot: Option<PyReadonlyArray1<'py, f64>>) -> Result<Bound<'py, PyArray2<f64>>, PyDiffsolError> {
        solve::<_, _, LSD>(&mut self.problem, py, params, t_interp, t_eval, y0, y0dot)
    }
}

unsafe impl Send for PyDiffsolSparse {}
unsafe impl Sync for PyDiffsolSparse {}

#[pyclass]
struct PyDiffsolSparse {
    problem: OdeSolverProblem<EqnS>,
}

#[pymethods]
impl PyDiffsolSparse {
    #[new]
    fn new(code: &str, rtol: f64, atol: f64) -> Result<Self, PyDiffsolError> {
        let diffsl = EqnS::compile(code)?;
        let nparams = diffsl.nparams();
        let dummy_params = vec![0.0; nparams];
        let problem = OdeBuilder::<MS>::new()
            .p(dummy_params)
            .rtol(rtol)
            .atol([atol])
            .build_from_eqn(diffsl)?;
        Ok(Self { problem })
    }

    #[pyo3(signature = (params, t_interp, t_eval, y0=None, y0dot=None))]
    fn solve<'py>(&mut self, py: Python<'py>, params: PyReadonlyArray1<'py, f64>, t_interp: PyReadonlyArray1<'py, f64>, t_eval: PyReadonlyArray1<'py, f64>, y0: Option<PyReadonlyArray1<'py, f64>>, y0dot: Option<PyReadonlyArray1<'py, f64>>) -> Result<Bound<'py, PyArray2<f64>>, PyDiffsolError> {
        solve::<_, _, LSS>(&mut self.problem, py, params, t_interp, t_eval, y0, y0dot)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn diffsol_python_benchmark(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDiffsolDense>()?;
    m.add_class::<PyDiffsolSparse>()?;
    Ok(())
}

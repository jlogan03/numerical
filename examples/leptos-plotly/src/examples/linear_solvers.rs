use crate::plot_helpers::{LineSeries, build_line_plot};
use crate::plotly_support::use_plotly_chart;
use crate::timing::measure_average_until;
use faer::sparse::linalg::lu::LuSymbolicParams;
use faer::sparse::{SparseColMat, Triplet};
use faer::{Par, Spec};
use leptos::prelude::*;
use numerical::sparse::{BiCGSTAB, DiagonalPrecond, SparseLu};
use plotly::Plot;

/// Interactive sparse direct-versus-iterative solver comparison.
#[component]
pub fn LinearSolverComparisonPage() -> impl IntoView {
    let (dimension, set_dimension) = signal(120_usize);
    let (diagonal_shift, set_diagonal_shift) = signal(0.25_f64);
    let (tol_log10, set_tol_log10) = signal(-6.0_f64);
    let (max_iterations, set_max_iterations) = signal(180_usize);
    let (warm_start, set_warm_start) = signal(false);
    let (preconditioner, set_preconditioner) = signal(BicgPreconditioner::None);

    let inputs = move || SparseSolverInputs {
        dimension: dimension.get(),
        diagonal_shift: diagonal_shift.get(),
        tolerance: 10.0_f64.powf(tol_log10.get()),
        max_iterations: max_iterations.get(),
        warm_start: warm_start.get(),
        preconditioner: preconditioner.get(),
    };
    let demo = Memo::new(move |_| run_sparse_solver_demo(inputs()));

    use_plotly_chart("linear-solvers-error-plot", move || {
        build_solver_plot(demo.get(), SolverPlot::StateError)
    });
    use_plotly_chart("linear-solvers-residual-plot", move || {
        build_solver_plot(demo.get(), SolverPlot::ResidualHistory)
    });

    let summary = move || sparse_solver_summary(demo.get());

    view! {
        <div class="page">
            <header class="page-header">
                <p class="eyebrow">"Linear Algebra"</p>
                <h1>"Sparse Solver Comparison"</h1>
                <p>
                    "The same shifted tridiagonal system is solved with sparse LU and with BiCGSTAB."
                    " The direct path gives a machine-precision baseline, while the iterative path"
                    " trades time for residual accuracy according to the selected tolerance."
                </p>
            </header>

            <div class="control-layout">
                <aside class="control-card">
                    <section>
                        <h2>"System"</h2>
                        <p class="section-copy">
                            "The planted system is a diagonally shifted one-dimensional Laplacian."
                            " Larger shifts improve conditioning, while looser BiCGSTAB tolerances"
                            " reduce iteration count."
                        </p>

                        <div class="control-row">
                            <label for="linear-solvers-dimension">"Dimension"</label>
                            <output>{move || dimension.get().to_string()}</output>
                            <input
                                id="linear-solvers-dimension"
                                type="range"
                                min="20"
                                max="260"
                                step="10"
                                prop:value=move || dimension.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<usize>() {
                                        set_dimension.set(value.clamp(20, 260));
                                    }
                                }
                            />
                        </div>

                        <div class="control-row">
                            <label for="linear-solvers-shift">"Diagonal shift"</label>
                            <output>{move || format!("{:.2}", diagonal_shift.get())}</output>
                            <input
                                id="linear-solvers-shift"
                                type="range"
                                min="0.05"
                                max="2.00"
                                step="0.05"
                                prop:value=move || diagonal_shift.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_diagonal_shift.set(value.clamp(0.05, 2.0));
                                    }
                                }
                            />
                        </div>

                        <div class="control-row">
                            <label for="linear-solvers-tolerance">"BiCGSTAB tolerance"</label>
                            <output>{move || format!("{:.1e}", 10.0_f64.powf(tol_log10.get()))}</output>
                            <input
                                id="linear-solvers-tolerance"
                                type="range"
                                min="-10"
                                max="-3"
                                step="0.25"
                                prop:value=move || tol_log10.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_tol_log10.set(value.clamp(-10.0, -3.0));
                                    }
                                }
                            />
                        </div>

                        <div class="control-row">
                            <label for="linear-solvers-max-iters">"Max iterations"</label>
                            <output>{move || max_iterations.get().to_string()}</output>
                            <input
                                id="linear-solvers-max-iters"
                                type="range"
                                min="20"
                                max="400"
                                step="10"
                                prop:value=move || max_iterations.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<usize>() {
                                        set_max_iterations.set(value.clamp(20, 400));
                                    }
                                }
                            />
                        </div>

                        <div class="control-row checkbox-row">
                            <label for="linear-solvers-warm-start">"Warm-start BiCGSTAB"</label>
                            <input
                                id="linear-solvers-warm-start"
                                type="checkbox"
                                prop:checked=move || warm_start.get()
                                on:change=move |ev| set_warm_start.set(event_target_checked(&ev))
                            />
                        </div>

                        <div class="control-row">
                            <label for="linear-solvers-preconditioner">"Preconditioner"</label>
                            <select
                                id="linear-solvers-preconditioner"
                                on:change=move |ev| {
                                    set_preconditioner
                                        .set(BicgPreconditioner::from_form_value(&event_target_value(&ev)));
                                }
                            >
                                <option
                                    value="none"
                                    selected=move || preconditioner.get() == BicgPreconditioner::None
                                >
                                    "None"
                                </option>
                                <option
                                    value="diagonal"
                                    selected=move || preconditioner.get() == BicgPreconditioner::Diagonal
                                >
                                    "Diagonal (Jacobi)"
                                </option>
                            </select>
                        </div>
                    </section>

                    <section>
                        <h2>"Interpretation"</h2>
                        <p class="section-copy">
                            "Sparse LU is a direct solve, so its accuracy is mostly independent of the"
                            " iterative tolerance slider. BiCGSTAB exposes the classic Krylov tradeoff:"
                            " tighter tolerance means more iterations and lower residual, especially on"
                            " the more weakly shifted systems. The warm-start toggle seeds the Krylov"
                            " solve with a diagonal-inverse guess instead of the zero vector, while"
                            " the preconditioner menu controls the right-preconditioning strategy."
                        </p>
                    </section>

                    <section>
                        <h2>"Run summary"</h2>
                        <p class="section-copy">{summary}</p>
                    </section>
                </aside>

                <div class="plots-grid wide">
                    <article class="plot-card">
                        <div class="plot-header">
                            <div>
                                <h2>"Statewise solution error"</h2>
                                <p>"Pointwise log10 absolute error against a planted dense truth vector."</p>
                            </div>
                        </div>
                        <div id="linear-solvers-error-plot" class="plot-surface"></div>
                    </article>

                    <article class="plot-card">
                        <div class="plot-header">
                            <div>
                                <h2>"Residual history"</h2>
                                <p>"BiCGSTAB residual norm per iteration, alongside the target tolerance and LU residual."</p>
                            </div>
                        </div>
                        <div id="linear-solvers-residual-plot" class="plot-surface"></div>
                    </article>
                </div>
            </div>
        </div>
    }
}

#[derive(Clone, Copy)]
enum SolverPlot {
    StateError,
    ResidualHistory,
}

#[derive(Clone, Copy)]
struct SparseSolverInputs {
    dimension: usize,
    diagonal_shift: f64,
    tolerance: f64,
    max_iterations: usize,
    warm_start: bool,
    preconditioner: BicgPreconditioner,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BicgPreconditioner {
    None,
    Diagonal,
}

impl BicgPreconditioner {
    fn from_form_value(value: &str) -> Self {
        match value {
            "diagonal" => Self::Diagonal,
            _ => Self::None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Diagonal => "diagonal",
        }
    }
}

#[derive(Clone, PartialEq)]
struct SparseSolverDemo {
    state_index: Vec<f64>,
    lu_log10_error: Vec<f64>,
    bicg_log10_error: Vec<f64>,
    iteration_index: Vec<f64>,
    bicg_log10_residual_history: Vec<f64>,
    tolerance_log10_line: Vec<f64>,
    lu_residual_log10_line: Vec<f64>,
    lu_ms: f64,
    bicg_ms: f64,
    lu_relative_error: f64,
    bicg_relative_error: f64,
    lu_residual_norm: f64,
    bicg_residual_norm: f64,
    bicg_iterations: usize,
    bicg_converged: bool,
    bicg_warm_started: bool,
    bicg_preconditioner: BicgPreconditioner,
    lu_repetitions: usize,
    bicg_repetitions: usize,
}

fn build_solver_plot(result: Result<SparseSolverDemo, String>, which: SolverPlot) -> Plot {
    match result {
        Ok(demo) => match which {
            SolverPlot::StateError => build_line_plot(
                "Sparse solve error",
                "state index",
                "log10 |x_hat - x_true|",
                false,
                vec![
                    LineSeries::lines("LU", demo.state_index.clone(), demo.lu_log10_error),
                    LineSeries::lines("BiCGSTAB", demo.state_index, demo.bicg_log10_error),
                ],
            ),
            SolverPlot::ResidualHistory => build_line_plot(
                "BiCGSTAB residual history",
                "iteration",
                "log10 residual norm",
                false,
                vec![
                    LineSeries::lines(
                        "BiCGSTAB residual",
                        demo.iteration_index.clone(),
                        demo.bicg_log10_residual_history,
                    ),
                    LineSeries::lines(
                        "target tolerance",
                        demo.iteration_index.clone(),
                        demo.tolerance_log10_line,
                    ),
                    LineSeries::lines(
                        "LU residual",
                        demo.iteration_index,
                        demo.lu_residual_log10_line,
                    ),
                ],
            ),
        },
        Err(message) => build_line_plot(&message, "", "", false, Vec::new()),
    }
}

fn sparse_solver_summary(result: Result<SparseSolverDemo, String>) -> String {
    match result {
        Ok(demo) => format!(
            "Sparse LU averaged {:.1} us over {} solves with relative solution error {:.2e} and residual {:.2e}. BiCGSTAB averaged {:.1} us over {} solves after {} iterations{}{} using {} preconditioning with relative solution error {:.2e} and residual {:.2e}.",
            millis_to_micros(demo.lu_ms),
            demo.lu_repetitions,
            demo.lu_relative_error,
            demo.lu_residual_norm,
            millis_to_micros(demo.bicg_ms),
            demo.bicg_repetitions,
            demo.bicg_iterations,
            if demo.bicg_converged {
                String::new()
            } else {
                String::from(" (did not hit the requested tolerance)")
            },
            if demo.bicg_warm_started {
                String::from("; warm-started")
            } else {
                String::new()
            },
            demo.bicg_preconditioner.label(),
            demo.bicg_relative_error,
            demo.bicg_residual_norm,
        ),
        Err(message) => format!("Sparse solver comparison failed: {message}"),
    }
}

fn run_sparse_solver_demo(inputs: SparseSolverInputs) -> Result<SparseSolverDemo, String> {
    let n = inputs.dimension.clamp(20, 260);
    let shift = inputs.diagonal_shift.clamp(0.05, 2.0);
    let tolerance = inputs.tolerance.clamp(1.0e-12, 1.0e-3);
    let max_iterations = inputs.max_iterations.clamp(1, 400);
    let matrix = shifted_laplacian_matrix(n, shift).map_err(|err| err.to_string())?;
    let x_true = planted_solution(n);
    let b = apply_shifted_laplacian(&x_true, shift);
    let bicg_initial_guess = if inputs.warm_start {
        diagonal_inverse_guess(&b, shift)
    } else {
        vec![0.0; n]
    };

    let (lu_solution, lu_ms, lu_repetitions) =
        measure_average_until(25.0, || solve_with_lu(&matrix, &b));
    let lu_solution = lu_solution?;
    let lu_solution_vec = lu_solution;

    let (bicg_result, bicg_ms, bicg_repetitions) = measure_average_until(25.0, || {
        solve_with_bicg(
            &matrix,
            &bicg_initial_guess,
            &b,
            tolerance,
            max_iterations,
            inputs.preconditioner,
        )
    });
    let bicg_result = bicg_result?;
    let bicg_solution_vec = bicg_result.solution;

    let state_index = (0..n).map(|i| (i + 1) as f64).collect::<Vec<_>>();
    let lu_error = pointwise_log10_error(&lu_solution_vec, &x_true);
    let bicg_error = pointwise_log10_error(&bicg_solution_vec, &x_true);
    let lu_residual = residual_norm(&lu_solution_vec, &b, shift);
    let bicg_residual = residual_norm(&bicg_solution_vec, &b, shift);
    let lu_relative_error = relative_error(&lu_solution_vec, &x_true);
    let bicg_relative_error = relative_error(&bicg_solution_vec, &x_true);
    let iteration_index = (0..bicg_result.residual_history.len())
        .map(|i| i as f64)
        .collect::<Vec<_>>();
    let bicg_log10_residual_history = bicg_result
        .residual_history
        .iter()
        .map(|&value| safe_log10(value))
        .collect::<Vec<_>>();
    let tolerance_log10_line = vec![safe_log10(tolerance); iteration_index.len()];
    let lu_residual_log10_line = vec![safe_log10(lu_residual); iteration_index.len()];

    Ok(SparseSolverDemo {
        state_index,
        lu_log10_error: lu_error,
        bicg_log10_error: bicg_error,
        iteration_index,
        bicg_log10_residual_history,
        tolerance_log10_line,
        lu_residual_log10_line,
        lu_ms,
        bicg_ms,
        lu_relative_error,
        bicg_relative_error,
        lu_residual_norm: lu_residual,
        bicg_residual_norm: bicg_residual,
        bicg_iterations: bicg_result.iterations,
        bicg_converged: bicg_result.converged,
        bicg_warm_started: inputs.warm_start,
        bicg_preconditioner: inputs.preconditioner,
        lu_repetitions,
        bicg_repetitions,
    })
}

struct BicgSolveResult {
    solution: Vec<f64>,
    residual_history: Vec<f64>,
    iterations: usize,
    converged: bool,
}

fn solve_with_lu(matrix: &SparseColMat<usize, f64>, b: &[f64]) -> Result<Vec<f64>, String> {
    let lu = SparseLu::<usize, f64>::factorize(
        matrix.as_ref(),
        Par::Seq,
        LuSymbolicParams::default(),
        Spec::default(),
    )
    .map_err(|err| err.to_string())?;
    let solution = lu.solve_rhs(b, Par::Seq).map_err(|err| err.to_string())?;
    Ok((0..b.len()).map(|i| solution[i]).collect())
}

fn solve_with_bicg(
    matrix: &SparseColMat<usize, f64>,
    initial_guess: &[f64],
    b: &[f64],
    tolerance: f64,
    max_iterations: usize,
    preconditioner: BicgPreconditioner,
) -> Result<BicgSolveResult, String> {
    match preconditioner {
        BicgPreconditioner::None => {
            let solver =
                BiCGSTAB::new(matrix.as_ref(), initial_guess, b).map_err(|err| err.to_string())?;
            finish_bicg_solve(solver, b.len(), tolerance, max_iterations)
        }
        BicgPreconditioner::Diagonal => {
            let diagonal =
                DiagonalPrecond::try_from(matrix.as_ref()).map_err(|err| format!("{err:?}"))?;
            let solver = BiCGSTAB::new_with_precond(matrix.as_ref(), diagonal, initial_guess, b)
                .map_err(|err| err.to_string())?;
            finish_bicg_solve(solver, b.len(), tolerance, max_iterations)
        }
    }
}

fn finish_bicg_solve<P>(
    mut solver: BiCGSTAB<f64, faer::sparse::SparseColMatRef<'_, usize, f64>, P>,
    n: usize,
    tolerance: f64,
    max_iterations: usize,
) -> Result<BicgSolveResult, String>
where
    P: numerical::sparse::Precond<f64>,
{
    let mut residual_history = vec![solver.err()];
    let mut converged = solver.err() < tolerance;
    while solver.iteration_count() < max_iterations && !converged {
        solver.step();
        residual_history.push(solver.err());
        if solver.err() < tolerance {
            converged = true;
        }
    }
    Ok(BicgSolveResult {
        solution: (0..n).map(|i| solver.x()[i]).collect(),
        residual_history,
        iterations: solver.iteration_count(),
        converged,
    })
}

fn shifted_laplacian_matrix(
    n: usize,
    shift: f64,
) -> Result<SparseColMat<usize, f64>, faer::sparse::CreationError> {
    let mut triplets = Vec::with_capacity(3 * n.saturating_sub(2) + 2);
    for row in 0..n {
        triplets.push(Triplet::new(row, row, 2.0 + shift));
        if row > 0 {
            triplets.push(Triplet::new(row, row - 1, -1.0));
        }
        if row + 1 < n {
            triplets.push(Triplet::new(row, row + 1, -1.0));
        }
    }
    SparseColMat::try_new_from_triplets(n, n, &triplets)
}

fn planted_solution(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let xi = i as f64 / ((n - 1).max(1) as f64);
            (std::f64::consts::PI * xi).sin() + 0.35 * (2.0 * std::f64::consts::PI * xi).cos()
        })
        .collect()
}

fn apply_shifted_laplacian(x: &[f64], shift: f64) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0; n];
    for row in 0..n {
        let mut acc = (2.0 + shift) * x[row];
        if row > 0 {
            acc -= x[row - 1];
        }
        if row + 1 < n {
            acc -= x[row + 1];
        }
        out[row] = acc;
    }
    out
}

fn diagonal_inverse_guess(b: &[f64], shift: f64) -> Vec<f64> {
    let diag = 2.0 + shift;
    b.iter().map(|rhs| rhs / diag).collect()
}

fn residual_norm(x: &[f64], b: &[f64], shift: f64) -> f64 {
    let residual = apply_shifted_laplacian(x, shift)
        .iter()
        .zip(b.iter())
        .map(|(ax, rhs)| ax - rhs)
        .collect::<Vec<_>>();
    euclidean_norm(&residual)
}

fn relative_error(lhs: &[f64], rhs: &[f64]) -> f64 {
    let diff = lhs
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| lhs - rhs)
        .collect::<Vec<_>>();
    euclidean_norm(&diff) / euclidean_norm(rhs).max(1.0e-16)
}

fn pointwise_log10_error(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| safe_log10((lhs - rhs).abs()))
        .collect()
}

fn euclidean_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn safe_log10(value: f64) -> f64 {
    value.max(1.0e-16).log10()
}

fn millis_to_micros(millis: f64) -> f64 {
    millis * 1.0e3
}

#[cfg(test)]
mod tests {
    use super::{BicgPreconditioner, SparseSolverInputs, run_sparse_solver_demo};

    #[test]
    fn sparse_solver_demo_runs() {
        let demo = run_sparse_solver_demo(SparseSolverInputs {
            dimension: 80,
            diagonal_shift: 0.25,
            tolerance: 1.0e-6,
            max_iterations: 180,
            warm_start: false,
            preconditioner: BicgPreconditioner::None,
        })
        .unwrap();
        assert_eq!(demo.state_index.len(), 80);
        assert!(!demo.bicg_log10_residual_history.is_empty());
    }
}

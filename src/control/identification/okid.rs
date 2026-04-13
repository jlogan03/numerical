use crate::control::realization::MarkovSequence;
use crate::decomp::{DecompError, DenseDecompParams, dense_svd};
use crate::sparse::compensated::{CompensatedField, CompensatedSum};
use faer::{Col, Mat, MatRef};
use faer_traits::ComplexField;
use faer_traits::ext::ComplexFieldExt;
use num_traits::{Float, Zero};
use std::fmt;

/// Builder-style parameters for OKID.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OkidParams {
    /// Number of system Markov blocks to recover, including `H_0 = D`.
    pub n_markov: usize,
    /// Number of observer-Markov lags used in the regression.
    pub observer_order: usize,
}

impl OkidParams {
    /// Creates OKID parameters with the required Markov horizon and observer
    /// order.
    #[must_use]
    pub fn new(n_markov: usize, observer_order: usize) -> Self {
        Self {
            n_markov,
            observer_order,
        }
    }
}

/// Result of OKID Markov estimation.
#[derive(Clone, Debug)]
pub struct OkidResult<T> {
    /// Recovered discrete-time Markov sequence.
    pub markov: MarkovSequence<T>,
    /// Estimated direct-feedthrough block `D`.
    pub direct_feedthrough: Mat<T>,
    /// Estimated observer Markov blocks for lags `1..=observer_order`.
    ///
    /// Each block has shape `noutputs x (ninputs + noutputs)` and is partitioned
    /// as `[Y_u(i) | Y_y(i)]`.
    pub observer_markov_blocks: Vec<Mat<T>>,
}

/// Errors produced by OKID.
#[derive(Debug)]
pub enum OkidError {
    /// Input and output data did not have compatible matrix dimensions.
    DimensionMismatch {
        /// Identifies the incompatible matrix.
        which: &'static str,
        /// Required row count.
        expected_nrows: usize,
        /// Required column count.
        expected_ncols: usize,
        /// Actual row count supplied by the caller.
        actual_nrows: usize,
        /// Actual column count supplied by the caller.
        actual_ncols: usize,
    },
    /// The requested Markov horizon was zero.
    InvalidMarkovCount,
    /// The requested observer order was zero.
    InvalidObserverOrder,
    /// There were not enough time samples for the requested observer order.
    NotEnoughSamples {
        /// Number of available time samples.
        samples: usize,
        /// Requested observer-Markov regression depth.
        observer_order: usize,
    },
    /// Dense SVD failed while solving the regression.
    Decomposition(DecompError),
    /// The regression matrix was numerically rank-deficient.
    RankDeficientRegression,
    /// A recovered quantity contained a non-finite entry.
    NonFiniteResult {
        /// Identifies the recovered quantity with the non-finite entry.
        which: &'static str,
    },
}

impl fmt::Display for OkidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for OkidError {}

impl From<DecompError> for OkidError {
    fn from(value: DecompError) -> Self {
        Self::Decomposition(value)
    }
}

/// Estimates a discrete-time Markov sequence from input/output data using the
/// OKID observer-Markov regression.
///
/// The current implementation assumes the supplied data are compatible with the
/// linear time-invariant discrete-time model used by the regression and that
/// the initial-condition effect has either decayed or been trimmed away.
pub fn okid<T>(
    outputs: MatRef<'_, T>,
    inputs: MatRef<'_, T>,
    params: &OkidParams,
) -> Result<OkidResult<T>, OkidError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    validate_okid_inputs(outputs, inputs, params)?;
    let observer = observer_markov_regression(outputs, inputs, params.observer_order)?;
    let direct_feedthrough = first_columns(observer.as_ref(), inputs.nrows());
    let observer_markov_blocks =
        observer_blocks(observer.as_ref(), outputs.nrows(), inputs.nrows());
    let markov = recover_markov_sequence(
        direct_feedthrough.as_ref(),
        &observer_markov_blocks,
        params.n_markov,
    )?;

    Ok(OkidResult {
        markov,
        direct_feedthrough,
        observer_markov_blocks,
    })
}

fn validate_okid_inputs<T>(
    outputs: MatRef<'_, T>,
    inputs: MatRef<'_, T>,
    params: &OkidParams,
) -> Result<(), OkidError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    if params.n_markov == 0 {
        return Err(OkidError::InvalidMarkovCount);
    }
    if params.observer_order == 0 {
        return Err(OkidError::InvalidObserverOrder);
    }
    if outputs.ncols() != inputs.ncols() {
        return Err(OkidError::DimensionMismatch {
            which: "outputs.ncols",
            expected_nrows: outputs.nrows(),
            expected_ncols: inputs.ncols(),
            actual_nrows: outputs.nrows(),
            actual_ncols: outputs.ncols(),
        });
    }
    if outputs.ncols() <= params.observer_order {
        return Err(OkidError::NotEnoughSamples {
            samples: outputs.ncols(),
            observer_order: params.observer_order,
        });
    }
    Ok(())
}

fn observer_markov_regression<T>(
    outputs: MatRef<'_, T>,
    inputs: MatRef<'_, T>,
    observer_order: usize,
) -> Result<Mat<T>, OkidError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    let noutputs = outputs.nrows();
    let ninputs = inputs.nrows();
    let nsamples = inputs.ncols();
    let regressor_rows = ninputs + observer_order * (ninputs + noutputs);
    let ncols = nsamples - observer_order;

    let phi = Mat::from_fn(regressor_rows, ncols, |row, col| {
        let k = observer_order + col;
        if row < ninputs {
            return inputs[(row, k)];
        }
        let offset = row - ninputs;
        let lag = offset / (ninputs + noutputs) + 1;
        let inner = offset % (ninputs + noutputs);
        let sample = k - lag;
        if inner < ninputs {
            inputs[(inner, sample)]
        } else {
            outputs[(inner - ninputs, sample)]
        }
    });
    let y = Mat::from_fn(noutputs, ncols, |row, col| {
        outputs[(row, observer_order + col)]
    });
    let phi_pinv = pseudo_inverse(phi.as_ref())?;
    let theta = dense_mul(y.as_ref(), phi_pinv.as_ref());
    check_finite(theta.as_ref(), "observer_markov_regression")?;
    Ok(theta)
}

fn recover_markov_sequence<T>(
    direct_feedthrough: MatRef<'_, T>,
    observer_markov_blocks: &[Mat<T>],
    n_markov: usize,
) -> Result<MarkovSequence<T>, OkidError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    let noutputs = direct_feedthrough.nrows();
    let ninputs = direct_feedthrough.ncols();
    let observer_order = observer_markov_blocks.len();
    let mut blocks = Vec::with_capacity(n_markov);
    blocks.push(clone_mat(direct_feedthrough));

    for k in 1..n_markov {
        let mut h_k = Mat::<T>::zeros(noutputs, ninputs);
        if k <= observer_order {
            let y_u = first_columns(observer_markov_blocks[k - 1].as_ref(), ninputs);
            h_k = dense_add(h_k.as_ref(), y_u.as_ref());
        }
        for i in 1..=k.min(observer_order) {
            let y_y = trailing_columns(observer_markov_blocks[i - 1].as_ref(), noutputs);
            let term = dense_mul(y_y.as_ref(), blocks[k - i].as_ref());
            h_k = dense_add(h_k.as_ref(), term.as_ref());
        }
        check_finite(h_k.as_ref(), "markov_sequence")?;
        blocks.push(h_k);
    }

    Ok(MarkovSequence::from_blocks(blocks)
        .expect("recovered OKID blocks should be shape-consistent"))
}

fn observer_blocks<T>(theta: MatRef<'_, T>, noutputs: usize, ninputs: usize) -> Vec<Mat<T>>
where
    T: Copy,
{
    let block_width = ninputs + noutputs;
    let observer_order = (theta.ncols() - ninputs) / block_width;
    (0..observer_order)
        .map(|i| {
            let start = ninputs + i * block_width;
            Mat::from_fn(noutputs, block_width, |row, col| theta[(row, start + col)])
        })
        .collect()
}

fn pseudo_inverse<T>(matrix: MatRef<'_, T>) -> Result<Mat<T>, OkidError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    let svd = dense_svd(matrix, &DenseDecompParams::<T>::new())?;
    let singular_values = Col::from_fn(svd.s.nrows(), |i| svd.s[i].abs());
    let mut max_sigma = <T::Real as Zero>::zero();
    for i in 0..singular_values.nrows() {
        if singular_values[i] > max_sigma {
            max_sigma = singular_values[i];
        }
    }
    let tol = T::Real::epsilon().sqrt() * max_sigma;
    let retained = (0..singular_values.nrows())
        .take_while(|&i| singular_values[i] > tol)
        .count();
    if retained == 0 {
        return Err(OkidError::RankDeficientRegression);
    }

    let v_r = Mat::from_fn(svd.v.nrows(), retained, |row, col| svd.v[(row, col)]);
    let u_r_h = Mat::from_fn(retained, svd.u.nrows(), |row, col| svd.u[(col, row)].conj());
    let sigma_inv = Mat::from_fn(retained, retained, |row, col| {
        if row == col {
            T::one().mul_real(singular_values[row].recip())
        } else {
            T::zero()
        }
    });
    Ok(dense_mul(
        dense_mul(v_r.as_ref(), sigma_inv.as_ref()).as_ref(),
        u_r_h.as_ref(),
    ))
}

fn first_columns<T: Copy>(matrix: MatRef<'_, T>, ncols: usize) -> Mat<T> {
    Mat::from_fn(matrix.nrows(), ncols, |row, col| matrix[(row, col)])
}

fn trailing_columns<T: Copy>(matrix: MatRef<'_, T>, ncols: usize) -> Mat<T> {
    let start = matrix.ncols() - ncols;
    Mat::from_fn(matrix.nrows(), ncols, |row, col| matrix[(row, start + col)])
}

fn dense_add<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: ComplexField + Copy,
{
    assert_eq!(lhs.nrows(), rhs.nrows());
    assert_eq!(lhs.ncols(), rhs.ncols());
    Mat::from_fn(lhs.nrows(), lhs.ncols(), |row, col| {
        lhs[(row, col)] + rhs[(row, col)]
    })
}

fn dense_mul<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    assert_eq!(lhs.ncols(), rhs.nrows());
    Mat::from_fn(lhs.nrows(), rhs.ncols(), |row, col| {
        let mut acc = CompensatedSum::<T>::default();
        for k in 0..lhs.ncols() {
            acc.add(lhs[(row, k)] * rhs[(k, col)]);
        }
        acc.finish()
    })
}

fn clone_mat<T: Copy>(matrix: MatRef<'_, T>) -> Mat<T> {
    Mat::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        matrix[(row, col)]
    })
}

fn check_finite<T>(matrix: MatRef<'_, T>, which: &'static str) -> Result<(), OkidError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    for col in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            let value = matrix[(row, col)];
            if !value.real().is_finite() || !value.imag().is_finite() {
                return Err(OkidError::NonFiniteResult { which });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{OkidError, OkidParams, okid};
    use crate::control::identification::{EraParams, era_from_markov};
    use crate::control::lti::state_space::DiscreteStateSpace;
    use faer::{Mat, MatRef};

    fn assert_close(lhs: MatRef<'_, f64>, rhs: MatRef<'_, f64>, tol: f64) {
        assert_eq!(lhs.nrows(), rhs.nrows());
        assert_eq!(lhs.ncols(), rhs.ncols());
        for col in 0..lhs.ncols() {
            for row in 0..lhs.nrows() {
                let err = (lhs[(row, col)] - rhs[(row, col)]).abs();
                assert!(
                    err <= tol,
                    "entry ({row}, {col}) differs: lhs={}, rhs={}, err={err}, tol={tol}",
                    lhs[(row, col)],
                    rhs[(row, col)],
                );
            }
        }
    }

    fn scalar_system() -> DiscreteStateSpace<f64> {
        DiscreteStateSpace::new(
            Mat::from_fn(1, 1, |_, _| 0.5),
            Mat::from_fn(1, 1, |_, _| 2.0),
            Mat::from_fn(1, 1, |_, _| 3.0),
            Mat::from_fn(1, 1, |_, _| 4.0),
            0.1,
        )
        .unwrap()
    }

    #[test]
    fn okid_recovers_scalar_markov_sequence_from_noiseless_data() {
        let sys = scalar_system();
        let inputs = Mat::from_fn(1, 24, |_, col| match col % 6 {
            0 => 1.0,
            1 => -0.75,
            2 => 0.5,
            3 => 2.0,
            4 => -1.25,
            _ => 0.25,
        });
        let sim = sys.simulate(&[0.0], inputs.as_ref()).unwrap();
        let result = okid(
            sim.outputs.as_ref(),
            sim.inputs.as_ref(),
            &OkidParams::new(8, 4),
        )
        .unwrap();

        let expected = sys.markov_parameters(8);
        for k in 0..expected.len() {
            assert_close(result.markov.block(k), expected.block(k), 1.0e-9);
        }
        assert_eq!(result.observer_markov_blocks.len(), 4);
    }

    #[test]
    fn okid_handles_small_mimo_data() {
        let sys = DiscreteStateSpace::new(
            Mat::from_fn(2, 2, |row, col| match (row, col) {
                (0, 0) => 0.6,
                (0, 1) => 0.1,
                (1, 0) => 0.0,
                (1, 1) => 0.4,
                _ => unreachable!(),
            }),
            Mat::from_fn(2, 2, |row, col| match (row, col) {
                (0, 0) => 1.0,
                (0, 1) => -0.5,
                (1, 0) => 0.25,
                (1, 1) => 0.75,
                _ => unreachable!(),
            }),
            Mat::from_fn(2, 2, |row, col| match (row, col) {
                (0, 0) => 1.0,
                (0, 1) => 0.2,
                (1, 0) => -0.3,
                (1, 1) => 0.9,
                _ => unreachable!(),
            }),
            Mat::from_fn(2, 2, |row, col| if row == col { 0.5 } else { 0.0 }),
            0.2,
        )
        .unwrap();

        let inputs = Mat::from_fn(2, 80, |row, col| {
            if row == 0 {
                (((3 * col + 1) % 11) as f64 - 5.0) / 3.0
            } else {
                (((7 * col + 2) % 13) as f64 - 6.0) / 4.0
            }
        });
        let sim = sys.simulate(&[0.0, 0.0], inputs.as_ref()).unwrap();
        let result = okid(
            sim.outputs.as_ref(),
            sim.inputs.as_ref(),
            &OkidParams::new(8, 5),
        )
        .unwrap();
        let expected = sys.markov_parameters(8);
        for k in 0..expected.len() {
            assert_close(result.markov.block(k), expected.block(k), 1.0e-7);
        }
    }

    #[test]
    fn okid_to_era_pipeline_recovers_response() {
        let sys = scalar_system();
        let inputs = Mat::from_fn(1, 48, |_, col| (((5 * col + 2) % 17) as f64 - 8.0) / 3.0);
        let sim = sys.simulate(&[0.0], inputs.as_ref()).unwrap();
        let identified = okid(
            sim.outputs.as_ref(),
            sim.inputs.as_ref(),
            &OkidParams::new(10, 5),
        )
        .unwrap();
        let era =
            era_from_markov(&identified.markov, 3, 3, &EraParams::new(sys.sample_time())).unwrap();
        let recovered = era.realized.markov_parameters(10);
        let expected = sys.markov_parameters(10);
        for k in 0..expected.len() {
            assert_close(recovered.block(k), expected.block(k), 1.0e-8);
        }
    }

    #[test]
    fn okid_rejects_too_few_samples() {
        let outputs = Mat::<f64>::zeros(1, 3);
        let inputs = Mat::<f64>::zeros(1, 3);
        let err = okid(outputs.as_ref(), inputs.as_ref(), &OkidParams::new(4, 3)).unwrap_err();
        assert!(matches!(
            err,
            OkidError::NotEnoughSamples {
                samples: 3,
                observer_order: 3
            }
        ));
    }
}

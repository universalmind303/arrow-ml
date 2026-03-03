use arrow::array::{ArrowPrimitiveType, PrimitiveArray};
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_kernels_common::KernelError;
use arrow_kernels_common::Result;
use num_traits::{Float, One, Zero};

/// Layer Normalization over the last dimension.
///
/// Input: 2D tensor (batch x features).
/// Gamma/beta: 1D arrays of length `features` (scale and shift).
///
/// For each row: y = gamma * (x - mean) / sqrt(var + eps) + beta
pub fn layer_norm<T>(
    input: &Tensor<'_, T>,
    gamma: &PrimitiveArray<T>,
    beta: &PrimitiveArray<T>,
    eps: T::Native,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("layer_norm: tensor has no shape".into()))?;
    if shape.len() != 2 {
        return Err(KernelError::InvalidArgument(format!(
            "layer_norm: expected 2D tensor, got {}D",
            shape.len()
        )));
    }
    let (rows, cols) = (shape[0], shape[1]);

    if gamma.len() != cols || beta.len() != cols {
        return Err(KernelError::ShapeMismatch {
            operation: "layer_norm",
            expected: format!("gamma/beta length {cols}"),
            actual: format!("gamma={}, beta={}", gamma.len(), beta.len()),
        });
    }

    let data: &[T::Native] = input.data().typed_data();
    let gamma_vals = gamma.values();
    let beta_vals = beta.values();
    let n: T::Native = <T::Native as num_traits::NumCast>::from(cols).unwrap();

    let mut out = Vec::with_capacity(rows * cols);

    for i in 0..rows {
        let row = &data[i * cols..(i + 1) * cols];

        let mut mean = T::Native::zero();
        for &v in row {
            mean = mean + v;
        }
        mean = mean / n;

        let mut var = T::Native::zero();
        for &v in row {
            let d = v - mean;
            var = var + d * d;
        }
        var = var / n;

        let inv_std = T::Native::one() / (var + eps).sqrt();

        for j in 0..cols {
            out.push(gamma_vals[j] * (row[j] - mean) * inv_std + beta_vals[j]);
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(vec![rows, cols]), None).map_err(KernelError::from)
}

/// RMS Normalization over the last dimension.
///
/// Input: 2D tensor (batch x features).
/// Gamma: 1D array of length `features` (scale).
///
/// For each row: y = gamma * x / sqrt(mean(x²) + eps)
///
/// Simpler than LayerNorm — no mean centering, no beta.
/// Used by LLaMA, Mistral, and other modern LLMs.
pub fn rms_norm<T>(
    input: &Tensor<'_, T>,
    gamma: &PrimitiveArray<T>,
    eps: T::Native,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("rms_norm: tensor has no shape".into()))?;
    if shape.len() != 2 {
        return Err(KernelError::InvalidArgument(format!(
            "rms_norm: expected 2D tensor, got {}D",
            shape.len()
        )));
    }
    let (rows, cols) = (shape[0], shape[1]);

    if gamma.len() != cols {
        return Err(KernelError::ShapeMismatch {
            operation: "rms_norm",
            expected: format!("gamma length {cols}"),
            actual: format!("gamma={}", gamma.len()),
        });
    }

    let data: &[T::Native] = input.data().typed_data();
    let gamma_vals = gamma.values();
    let n: T::Native = <T::Native as num_traits::NumCast>::from(cols).unwrap();

    let mut out = Vec::with_capacity(rows * cols);

    for i in 0..rows {
        let row = &data[i * cols..(i + 1) * cols];

        let mut ms = T::Native::zero();
        for &v in row {
            ms = ms + v * v;
        }
        ms = ms / n;

        let inv_rms = T::Native::one() / (ms + eps).sqrt();

        for j in 0..cols {
            out.push(gamma_vals[j] * row[j] * inv_rms);
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(vec![rows, cols]), None).map_err(KernelError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float32Array;
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::Float32Type;

    fn make_f32_2d(data: Vec<f32>, rows: usize, cols: usize) -> Tensor<'static, Float32Type> {
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(vec![rows, cols]), None).unwrap()
    }

    #[test]
    fn test_layer_norm_basic() {
        let input = make_f32_2d(vec![1.0, 2.0, 3.0, 4.0], 1, 4);
        let gamma = Float32Array::from(vec![1.0, 1.0, 1.0, 1.0]);
        let beta = Float32Array::from(vec![0.0, 0.0, 0.0, 0.0]);
        let out = layer_norm(&input, &gamma, &beta, 1e-5).unwrap();
        let data = out.data().typed_data::<f32>();

        let std = (1.25f32 + 1e-5).sqrt();
        assert!((data[0] - (-1.5 / std)).abs() < 1e-4);
        assert!((data[1] - (-0.5 / std)).abs() < 1e-4);
        assert!((data[2] - (0.5 / std)).abs() < 1e-4);
        assert!((data[3] - (1.5 / std)).abs() < 1e-4);
    }

    #[test]
    fn test_layer_norm_with_scale_shift() {
        let input = make_f32_2d(vec![1.0, 2.0, 3.0, 4.0], 1, 4);
        let gamma = Float32Array::from(vec![2.0, 2.0, 2.0, 2.0]);
        let beta = Float32Array::from(vec![1.0, 1.0, 1.0, 1.0]);
        let out = layer_norm(&input, &gamma, &beta, 1e-5).unwrap();
        let data = out.data().typed_data::<f32>();

        let std = (1.25f32 + 1e-5).sqrt();
        assert!((data[0] - (2.0 * (-1.5 / std) + 1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm_basic() {
        let input = make_f32_2d(vec![1.0, 2.0, 3.0, 4.0], 1, 4);
        let gamma = Float32Array::from(vec![1.0, 1.0, 1.0, 1.0]);
        let out = rms_norm(&input, &gamma, 1e-5).unwrap();
        let data = out.data().typed_data::<f32>();

        let rms = (7.5f32 + 1e-5).sqrt();
        assert!((data[0] - (1.0 / rms)).abs() < 1e-4);
        assert!((data[1] - (2.0 / rms)).abs() < 1e-4);
        assert!((data[2] - (3.0 / rms)).abs() < 1e-4);
        assert!((data[3] - (4.0 / rms)).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm_batched() {
        let input = make_f32_2d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let gamma = Float32Array::from(vec![1.0, 1.0, 1.0]);
        let out = rms_norm(&input, &gamma, 1e-5).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 3]);
    }
}

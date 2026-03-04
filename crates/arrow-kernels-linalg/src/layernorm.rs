use arrow::array::{ArrowPrimitiveType, PrimitiveArray};
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_kernels_common::KernelError;
use arrow_kernels_common::Result;
use num_traits::{Float, One, Zero};

/// Layer Normalization over the last dimension.
///
/// Input: N-D tensor (rank >= 1). All leading dimensions are treated as
/// batch dimensions; normalization is performed over the last dimension.
/// Gamma/beta: 1D arrays of length `features` (scale and shift).
///
/// For each row along the last axis: y = gamma * (x - mean) / sqrt(var + eps) + beta
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
    if shape.is_empty() {
        return Err(KernelError::InvalidArgument(
            "layer_norm: tensor must be at least 1D".into(),
        ));
    }
    let cols = *shape.last().unwrap();
    let rows: usize = shape[..shape.len() - 1].iter().product();
    let rows = if rows == 0 { 1 } else { rows };

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
    Tensor::new_row_major(buf, Some(shape.to_vec()), None).map_err(KernelError::from)
}

/// RMS Normalization over the last dimension.
///
/// Input: N-D tensor (rank >= 1). All leading dimensions are treated as
/// batch dimensions; normalization is performed over the last dimension.
/// Gamma: 1D array of length `features` (scale).
///
/// For each row along the last axis: y = gamma * x / sqrt(mean(x^2) + eps)
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
    if shape.is_empty() {
        return Err(KernelError::InvalidArgument(
            "rms_norm: tensor must be at least 1D".into(),
        ));
    }
    let cols = *shape.last().unwrap();
    let rows: usize = shape[..shape.len() - 1].iter().product();
    let rows = if rows == 0 { 1 } else { rows };

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
    Tensor::new_row_major(buf, Some(shape.to_vec()), None).map_err(KernelError::from)
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

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
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

    // ---- N-D tests ----

    #[test]
    fn test_layer_norm_3d() {
        // 3D: batch=2, seq=3, hidden=4  (BERT-like shape)
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let input = make_f32(data.clone(), vec![2, 3, 4]);
        let gamma = Float32Array::from(vec![1.0, 1.0, 1.0, 1.0]);
        let beta = Float32Array::from(vec![0.0, 0.0, 0.0, 0.0]);

        let out = layer_norm(&input, &gamma, &beta, 1e-5).unwrap();

        // Shape must be preserved as [2, 3, 4]
        assert_eq!(out.shape().unwrap(), &vec![2, 3, 4]);

        let out_data = out.data().typed_data::<f32>();

        // Verify normalization for the first row (elements 1,2,3,4)
        // mean = 2.5, var = 1.25
        let mean = 2.5f32;
        let var = 1.25f32;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        assert!((out_data[0] - (1.0 - mean) * inv_std).abs() < 1e-4);
        assert!((out_data[1] - (2.0 - mean) * inv_std).abs() < 1e-4);
        assert!((out_data[2] - (3.0 - mean) * inv_std).abs() < 1e-4);
        assert!((out_data[3] - (4.0 - mean) * inv_std).abs() < 1e-4);

        // Verify normalization for the last row (elements 21,22,23,24)
        let mean_last = 22.5f32;
        let var_last = 1.25f32;
        let inv_std_last = 1.0 / (var_last + 1e-5).sqrt();
        assert!((out_data[20] - (21.0 - mean_last) * inv_std_last).abs() < 1e-4);
        assert!((out_data[23] - (24.0 - mean_last) * inv_std_last).abs() < 1e-4);
    }

    #[test]
    fn test_layer_norm_1d() {
        // 1D tensor: 4 elements, single "row"
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let gamma = Float32Array::from(vec![1.0, 1.0, 1.0, 1.0]);
        let beta = Float32Array::from(vec![0.0, 0.0, 0.0, 0.0]);

        let out = layer_norm(&input, &gamma, &beta, 1e-5).unwrap();

        // Shape must be preserved as [4]
        assert_eq!(out.shape().unwrap(), &vec![4]);

        let out_data = out.data().typed_data::<f32>();

        // mean = 2.5, var = 1.25 — same as the basic 2D (1,4) test
        let std = (1.25f32 + 1e-5).sqrt();
        assert!((out_data[0] - (-1.5 / std)).abs() < 1e-4);
        assert!((out_data[1] - (-0.5 / std)).abs() < 1e-4);
        assert!((out_data[2] - (0.5 / std)).abs() < 1e-4);
        assert!((out_data[3] - (1.5 / std)).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm_3d() {
        // 3D: batch=2, seq=2, features=3
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let input = make_f32(data.clone(), vec![2, 2, 3]);
        let gamma = Float32Array::from(vec![1.0, 1.0, 1.0]);

        let out = rms_norm(&input, &gamma, 1e-5).unwrap();

        // Shape must be preserved as [2, 2, 3]
        assert_eq!(out.shape().unwrap(), &vec![2, 2, 3]);

        let out_data = out.data().typed_data::<f32>();

        // Verify first row (1,2,3): mean_sq = (1+4+9)/3 = 14/3
        let ms = 14.0f32 / 3.0;
        let inv_rms = 1.0 / (ms + 1e-5).sqrt();
        assert!((out_data[0] - 1.0 * inv_rms).abs() < 1e-4);
        assert!((out_data[1] - 2.0 * inv_rms).abs() < 1e-4);
        assert!((out_data[2] - 3.0 * inv_rms).abs() < 1e-4);
    }

    #[test]
    fn test_layer_norm_4d() {
        // 4D: (1, 2, 3, 4) — e.g. batch x channel x height x width
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let input = make_f32(data, vec![1, 2, 3, 4]);
        let gamma = Float32Array::from(vec![1.0, 1.0, 1.0, 1.0]);
        let beta = Float32Array::from(vec![0.0, 0.0, 0.0, 0.0]);

        let out = layer_norm(&input, &gamma, &beta, 1e-5).unwrap();

        // Shape must be preserved as [1, 2, 3, 4]
        assert_eq!(out.shape().unwrap(), &vec![1, 2, 3, 4]);

        let out_data = out.data().typed_data::<f32>();

        // Total elements = 24, number of "rows" = 1*2*3 = 6, each of length 4
        // First row (1,2,3,4): mean=2.5, var=1.25
        let mean = 2.5f32;
        let inv_std = 1.0 / (1.25f32 + 1e-5).sqrt();
        assert!((out_data[0] - (1.0 - mean) * inv_std).abs() < 1e-4);
        assert!((out_data[3] - (4.0 - mean) * inv_std).abs() < 1e-4);

        // Each normalized row should have zero mean (within tolerance)
        for row_idx in 0..6 {
            let row_start = row_idx * 4;
            let row_sum: f32 = out_data[row_start..row_start + 4].iter().sum();
            assert!(row_sum.abs() < 1e-4, "Row {row_idx} mean should be ~0, got {row_sum}");
        }
    }
}

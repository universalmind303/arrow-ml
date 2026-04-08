use arrow::array::{ArrowPrimitiveType, PrimitiveArray};
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_ml_common::{KernelError, Result};
use num_traits::Float;

/// Inference-mode Batch Normalization over 4D NCHW tensors.
///
/// For each channel c:
///   y = scale[c] * (x - mean[c]) / sqrt(var[c] + eps) + bias[c]
///
/// Pre-computes affine coefficients so inner loop is just y = a*x + b.
pub fn batch_norm<T>(
    input: &Tensor<'_, T>,
    scale: &PrimitiveArray<T>,
    bias: &PrimitiveArray<T>,
    mean: &PrimitiveArray<T>,
    var: &PrimitiveArray<T>,
    eps: T::Native,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("batch_norm: tensor has no shape".into()))?;
    if shape.len() != 4 {
        return Err(KernelError::InvalidArgument(format!(
            "batch_norm: expected 4D tensor (NCHW), got {}D",
            shape.len()
        )));
    }
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let hw = h * w;

    if scale.len() != c || bias.len() != c || mean.len() != c || var.len() != c {
        return Err(KernelError::ShapeMismatch {
            operation: "batch_norm",
            expected: format!("all parameter arrays length {c}"),
            actual: format!(
                "scale={}, bias={}, mean={}, var={}",
                scale.len(),
                bias.len(),
                mean.len(),
                var.len()
            ),
        });
    }

    let data: &[T::Native] = input.data().typed_data();
    let scale_vals = scale.values();
    let bias_vals = bias.values();
    let mean_vals = mean.values();
    let var_vals = var.values();

    // Pre-compute per-channel affine: a[c] = scale[c] / sqrt(var[c] + eps), b[c] = bias[c] - a[c] * mean[c]
    let mut a_coeff = Vec::with_capacity(c);
    let mut b_coeff = Vec::with_capacity(c);
    for ci in 0..c {
        let a = scale_vals[ci] / (var_vals[ci] + eps).sqrt();
        let b = bias_vals[ci] - a * mean_vals[ci];
        a_coeff.push(a);
        b_coeff.push(b);
    }

    let mut out = Vec::with_capacity(n * c * hw);

    for _ni in 0..n {
        for ci in 0..c {
            let a = a_coeff[ci];
            let b = b_coeff[ci];
            let offset = _ni * c * hw + ci * hw;
            for i in 0..hw {
                out.push(a * data[offset + i] + b);
            }
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(vec![n, c, h, w]), None).map_err(KernelError::from)
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use arrow::array::Float32Array;
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::Float32Type;

    fn make_f32_4d(
        data: Vec<f32>,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> Tensor<'static, Float32Type> {
        let buffer = ScalarBuffer::<f32>::from(data).into_inner();
        Tensor::new_row_major(buffer, Some(vec![n, c, h, w]), None).unwrap()
    }

    #[test]
    fn test_identity_bn() {
        // scale=1, bias=0, mean=0, var=1 => output ≈ input
        let input = make_f32_4d(vec![1.0, 2.0, 3.0, 4.0], 1, 1, 2, 2);
        let scale = Float32Array::from(vec![1.0]);
        let bias = Float32Array::from(vec![0.0]);
        let mean = Float32Array::from(vec![0.0]);
        let var = Float32Array::from(vec![1.0]);
        let out = batch_norm(&input, &scale, &bias, &mean, &var, 1e-5).unwrap();
        let data = out.data().typed_data::<f32>();
        for i in 0..4 {
            assert!((data[i] - (i as f32 + 1.0)).abs() < 1e-3);
        }
    }

    #[test]
    fn test_nontrivial_scale_bias() {
        // 2 channels, 1x1 spatial
        let input = make_f32_4d(vec![2.0, 4.0], 1, 2, 1, 1);
        let scale = Float32Array::from(vec![2.0, 3.0]);
        let bias = Float32Array::from(vec![1.0, -1.0]);
        let mean = Float32Array::from(vec![1.0, 2.0]);
        let var = Float32Array::from(vec![1.0, 4.0]);
        let eps = 0.0f32;
        let out = batch_norm(&input, &scale, &bias, &mean, &var, eps).unwrap();
        let data = out.data().typed_data::<f32>();
        // ch0: 2*(2-1)/sqrt(1) + 1 = 3
        assert!((data[0] - 3.0).abs() < 1e-5);
        // ch1: 3*(4-2)/sqrt(4) + (-1) = 3*1 - 1 = 2
        assert!((data[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_multi_batch() {
        let input = make_f32_4d(vec![1.0, 2.0], 2, 1, 1, 1);
        let scale = Float32Array::from(vec![1.0]);
        let bias = Float32Array::from(vec![0.0]);
        let mean = Float32Array::from(vec![0.0]);
        let var = Float32Array::from(vec![1.0]);
        let out = batch_norm(&input, &scale, &bias, &mean, &var, 1e-5).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 1, 1, 1]);
    }

    #[test]
    fn test_shape_preserved() {
        let data = vec![0.0f32; 2 * 3 * 4 * 5];
        let input = make_f32_4d(data, 2, 3, 4, 5);
        let scale = Float32Array::from(vec![1.0, 1.0, 1.0]);
        let bias = Float32Array::from(vec![0.0, 0.0, 0.0]);
        let mean = Float32Array::from(vec![0.0, 0.0, 0.0]);
        let var = Float32Array::from(vec![1.0, 1.0, 1.0]);
        let out = batch_norm(&input, &scale, &bias, &mean, &var, 1e-5).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_wrong_dimensionality() {
        let buffer = ScalarBuffer::<f32>::from(vec![1.0, 2.0]).into_inner();
        let input: Tensor<'_, Float32Type> =
            Tensor::new_row_major(buffer, Some(vec![1, 2]), None).unwrap();
        let scale = Float32Array::from(vec![1.0]);
        let bias = Float32Array::from(vec![0.0]);
        let mean = Float32Array::from(vec![0.0]);
        let var = Float32Array::from(vec![1.0]);
        assert!(batch_norm(&input, &scale, &bias, &mean, &var, 1e-5).is_err());
    }

    #[test]
    fn test_param_length_mismatch() {
        let input = make_f32_4d(vec![1.0, 2.0, 3.0, 4.0], 1, 2, 1, 2);
        let scale = Float32Array::from(vec![1.0, 1.0]);
        let bias = Float32Array::from(vec![0.0, 0.0]);
        let mean = Float32Array::from(vec![0.0, 0.0]);
        let var = Float32Array::from(vec![1.0]); // wrong length
        assert!(batch_norm(&input, &scale, &bias, &mean, &var, 1e-5).is_err());
    }
}

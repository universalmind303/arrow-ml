use arrow::array::{ArrowPrimitiveType, PrimitiveArray};
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_ml_common::{KernelError, Result};
use num_traits::{Float, One, Zero};

/// Instance Normalization over 4D NCHW tensors.
///
/// Normalizes each (N, C) instance independently over (H, W).
///
/// For each instance: y = scale[c] * (x - mean) / sqrt(var + eps) + bias[c]
pub fn instance_norm<T>(
    input: &Tensor<'_, T>,
    scale: &PrimitiveArray<T>,
    bias: &PrimitiveArray<T>,
    eps: T::Native,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("instance_norm: tensor has no shape".into()))?;
    if shape.len() != 4 {
        return Err(KernelError::InvalidArgument(format!(
            "instance_norm: expected 4D tensor (NCHW), got {}D",
            shape.len()
        )));
    }
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let hw = h * w;

    if scale.len() != c || bias.len() != c {
        return Err(KernelError::ShapeMismatch {
            operation: "instance_norm",
            expected: format!("scale/bias length {c}"),
            actual: format!("scale={}, bias={}", scale.len(), bias.len()),
        });
    }

    let data: &[T::Native] = input.data().typed_data();
    let scale_vals = scale.values();
    let bias_vals = bias.values();
    let hw_f: T::Native = <T::Native as num_traits::NumCast>::from(hw).unwrap();

    let mut out = Vec::with_capacity(n * c * hw);

    for batch in 0..n {
        for ch in 0..c {
            let base = batch * c * hw + ch * hw;
            let slice = &data[base..base + hw];

            // Compute mean
            let mut mean = T::Native::zero();
            for &v in slice {
                mean = mean + v;
            }
            mean = mean / hw_f;

            // Compute variance
            let mut var = T::Native::zero();
            for &v in slice {
                let d = v - mean;
                var = var + d * d;
            }
            var = var / hw_f;

            let inv_std = T::Native::one() / (var + eps).sqrt();

            for &v in slice {
                out.push(scale_vals[ch] * (v - mean) * inv_std + bias_vals[ch]);
            }
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(vec![n, c, h, w]), None).map_err(KernelError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float32Array;
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::Float32Type;

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
        let buffer = ScalarBuffer::<f32>::from(data).into_inner();
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_instance_norm_basic() {
        // 1 batch, 1 channel, 2x2 spatial
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let scale = Float32Array::from(vec![1.0]);
        let bias = Float32Array::from(vec![0.0]);
        let out = instance_norm::<Float32Type>(&input, &scale, &bias, 1e-5).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1, 2, 2]);
        let data = out.data().typed_data::<f32>();
        // Mean = 2.5, var = 1.25
        let sum: f32 = data.iter().sum();
        assert!(sum.abs() < 1e-4); // zero mean
    }

    #[test]
    fn test_instance_norm_multi_channel() {
        // 1 batch, 2 channels, 1x2 spatial
        let input = make_f32(vec![1.0, 3.0, 10.0, 20.0], vec![1, 2, 1, 2]);
        let scale = Float32Array::from(vec![1.0, 1.0]);
        let bias = Float32Array::from(vec![0.0, 0.0]);
        let out = instance_norm::<Float32Type>(&input, &scale, &bias, 1e-5).unwrap();
        let data = out.data().typed_data::<f32>();
        // Ch 0: mean=2, var=1 -> [-1, 1] / sqrt(1+eps)
        // Ch 1: mean=15, var=25 -> [-1, 1] / sqrt(25+eps)
        assert!((data[0] + data[1]).abs() < 1e-4);
        assert!((data[2] + data[3]).abs() < 1e-4);
    }

    #[test]
    fn test_instance_norm_with_scale_bias() {
        let input = make_f32(vec![1.0, 3.0], vec![1, 1, 1, 2]);
        let scale = Float32Array::from(vec![2.0]);
        let bias = Float32Array::from(vec![5.0]);
        let out = instance_norm::<Float32Type>(&input, &scale, &bias, 1e-5).unwrap();
        let data = out.data().typed_data::<f32>();
        // mean=2, var=1, normalized: [-1, 1]/sqrt(1+eps)
        // scaled: [-2, 2]/sqrt(1+eps), shifted: [3, 7]/sqrt(1+eps) (approximately)
        assert!((data[0] + data[1] - 10.0).abs() < 1e-3); // sum of outputs = 2*bias
    }

    #[test]
    fn test_instance_norm_preserves_shape() {
        let input = make_f32(vec![0.0; 2 * 3 * 4 * 4], vec![2, 3, 4, 4]);
        let scale = Float32Array::from(vec![1.0, 1.0, 1.0]);
        let bias = Float32Array::from(vec![0.0, 0.0, 0.0]);
        let out = instance_norm::<Float32Type>(&input, &scale, &bias, 1e-5).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 3, 4, 4]);
    }
}

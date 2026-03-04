use arrow::array::{ArrowPrimitiveType, PrimitiveArray};
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_kernels_common::{KernelError, Result};
use num_traits::{Float, One, Zero};

/// Group Normalization over 4D NCHW tensors.
///
/// Divides C channels into `num_groups` groups, each of size C/num_groups.
/// Normalizes over (C/num_groups, H, W) for each (N, group).
///
/// For each group: y = gamma[c] * (x - mean) / sqrt(var + eps) + beta[c]
pub fn group_norm<T>(
    input: &Tensor<'_, T>,
    num_groups: usize,
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
        .ok_or_else(|| KernelError::InvalidArgument("group_norm: tensor has no shape".into()))?;
    if shape.len() != 4 {
        return Err(KernelError::InvalidArgument(format!(
            "group_norm: expected 4D tensor (NCHW), got {}D",
            shape.len()
        )));
    }
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let hw = h * w;

    if num_groups == 0 || c % num_groups != 0 {
        return Err(KernelError::InvalidArgument(format!(
            "group_norm: channels ({c}) must be divisible by num_groups ({num_groups})"
        )));
    }
    let channels_per_group = c / num_groups;

    if gamma.len() != c || beta.len() != c {
        return Err(KernelError::ShapeMismatch {
            operation: "group_norm",
            expected: format!("gamma/beta length {c}"),
            actual: format!("gamma={}, beta={}", gamma.len(), beta.len()),
        });
    }

    let data: &[T::Native] = input.data().typed_data();
    let gamma_vals = gamma.values();
    let beta_vals = beta.values();

    let group_size = channels_per_group * hw;
    let n_f: T::Native = <T::Native as num_traits::NumCast>::from(group_size).unwrap();

    let mut out = vec![T::Native::zero(); n * c * hw];

    for batch in 0..n {
        for g in 0..num_groups {
            // Compute mean and var over this group
            let mut mean = T::Native::zero();
            for cl in 0..channels_per_group {
                let ch = g * channels_per_group + cl;
                let base = batch * c * hw + ch * hw;
                for i in 0..hw {
                    mean = mean + data[base + i];
                }
            }
            mean = mean / n_f;

            let mut var = T::Native::zero();
            for cl in 0..channels_per_group {
                let ch = g * channels_per_group + cl;
                let base = batch * c * hw + ch * hw;
                for i in 0..hw {
                    let d = data[base + i] - mean;
                    var = var + d * d;
                }
            }
            var = var / n_f;

            let inv_std = T::Native::one() / (var + eps).sqrt();

            // Normalize and apply affine
            for cl in 0..channels_per_group {
                let ch = g * channels_per_group + cl;
                let base = batch * c * hw + ch * hw;
                for i in 0..hw {
                    out[base + i] =
                        gamma_vals[ch] * (data[base + i] - mean) * inv_std + beta_vals[ch];
                }
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
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_group_norm_basic() {
        // 1 batch, 4 channels, 1x1 spatial, 2 groups
        // Group 0: channels 0,1; Group 1: channels 2,3
        let input = make_f32(vec![1.0, 3.0, 5.0, 7.0], vec![1, 4, 1, 1]);
        let gamma = Float32Array::from(vec![1.0, 1.0, 1.0, 1.0]);
        let beta = Float32Array::from(vec![0.0, 0.0, 0.0, 0.0]);
        let out = group_norm::<Float32Type>(&input, 2, &gamma, &beta, 1e-5).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 4, 1, 1]);
        let data = out.data().typed_data::<f32>();
        // Group 0: mean=2, var=1; normalized: [-1, 1]
        assert!((data[0] - (-1.0 / (1.0 + 1e-5_f32).sqrt())).abs() < 1e-3);
        assert!((data[1] - (1.0 / (1.0 + 1e-5_f32).sqrt())).abs() < 1e-3);
    }

    #[test]
    fn test_group_norm_with_spatial() {
        // 1 batch, 2 channels, 2x2 spatial, 1 group (== layer norm)
        let input = make_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1, 2, 2, 2],
        );
        let gamma = Float32Array::from(vec![1.0, 1.0]);
        let beta = Float32Array::from(vec![0.0, 0.0]);
        let out = group_norm::<Float32Type>(&input, 1, &gamma, &beta, 1e-5).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 2, 2, 2]);
        // Mean over all 8 values = 4.5
        let data = out.data().typed_data::<f32>();
        let sum: f32 = data.iter().sum();
        assert!(sum.abs() < 1e-3); // should be ~0 (zero mean)
    }

    #[test]
    fn test_group_norm_channels_not_divisible() {
        let input = make_f32(vec![0.0; 12], vec![1, 3, 2, 2]);
        let gamma = Float32Array::from(vec![1.0, 1.0, 1.0]);
        let beta = Float32Array::from(vec![0.0, 0.0, 0.0]);
        assert!(group_norm::<Float32Type>(&input, 2, &gamma, &beta, 1e-5).is_err());
    }

    #[test]
    fn test_group_norm_preserves_shape() {
        let input = make_f32(vec![0.0; 2 * 4 * 3 * 3], vec![2, 4, 3, 3]);
        let gamma = Float32Array::from(vec![1.0, 1.0, 1.0, 1.0]);
        let beta = Float32Array::from(vec![0.0, 0.0, 0.0, 0.0]);
        let out = group_norm::<Float32Type>(&input, 2, &gamma, &beta, 1e-5).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 4, 3, 3]);
    }
}

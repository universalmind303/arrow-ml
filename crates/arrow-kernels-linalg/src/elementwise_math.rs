use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_kernels_common::{KernelError, Result};
use num_traits::Float;

/// Raise each element to a scalar exponent.
pub fn pow<T>(input: &Tensor<'_, T>, exponent: T::Native) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("pow: tensor has no shape".into()))?
        .to_vec();
    let data: &[T::Native] = input.data().typed_data();
    let out: Vec<T::Native> = data.iter().map(|&v| v.powf(exponent)).collect();
    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(shape), None).map_err(KernelError::from)
}

/// Element-wise power: a[i] ^ b[i]. Tensors must have the same shape.
pub fn pow_tensor<T>(a: &Tensor<'_, T>, b: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let shape_a = a
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("pow_tensor: tensor a has no shape".into()))?
        .to_vec();
    let shape_b = b
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("pow_tensor: tensor b has no shape".into()))?;
    if shape_a != *shape_b {
        return Err(KernelError::ShapeMismatch {
            operation: "pow_tensor",
            expected: format!("{:?}", shape_a),
            actual: format!("{:?}", shape_b),
        });
    }
    let data_a: &[T::Native] = a.data().typed_data();
    let data_b: &[T::Native] = b.data().typed_data();
    let out: Vec<T::Native> = data_a
        .iter()
        .zip(data_b.iter())
        .map(|(&x, &e)| x.powf(e))
        .collect();
    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(shape_a), None).map_err(KernelError::from)
}

/// Element-wise error function using Abramowitz & Stegun approximation (7.1.26).
pub fn erf<T>(input: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("erf: tensor has no shape".into()))?
        .to_vec();
    let data: &[T::Native] = input.data().typed_data();
    let out: Vec<T::Native> = data.iter().map(|&v| erf_approx(v)).collect();
    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(shape), None).map_err(KernelError::from)
}

/// Abramowitz & Stegun approximation of erf (max error ~1.5e-7).
fn erf_approx<F: Float>(x: F) -> F {
    let one = F::one();
    let zero = F::zero();
    let sign = if x >= zero { one } else { -one };
    let x = x.abs();

    let a1 = F::from(0.254829592).unwrap();
    let a2 = F::from(-0.284496736).unwrap();
    let a3 = F::from(1.421413741).unwrap();
    let a4 = F::from(-1.453152027).unwrap();
    let a5 = F::from(1.061405429).unwrap();
    let p = F::from(0.3275911).unwrap();

    let t = one / (one + p * x);
    let y = one - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Element-wise reciprocal: 1/x.
pub fn reciprocal<T>(input: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("reciprocal: tensor has no shape".into()))?
        .to_vec();
    let data: &[T::Native] = input.data().typed_data();
    let out: Vec<T::Native> = data.iter().map(|&v| v.recip()).collect();
    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(shape), None).map_err(KernelError::from)
}

/// Element-wise cosine.
pub fn cos_op<T>(input: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("cos_op: tensor has no shape".into()))?
        .to_vec();
    let data: &[T::Native] = input.data().typed_data();
    let out: Vec<T::Native> = data.iter().map(|&v| v.cos()).collect();
    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(shape), None).map_err(KernelError::from)
}

/// Element-wise sine.
pub fn sin_op<T>(input: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("sin_op: tensor has no shape".into()))?
        .to_vec();
    let data: &[T::Native] = input.data().typed_data();
    let out: Vec<T::Native> = data.iter().map(|&v| v.sin()).collect();
    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(shape), None).map_err(KernelError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::Float32Type;

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_pow_scalar() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = pow::<Float32Type>(&input, 2.0).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_pow_tensor_elementwise() {
        let a = make_f32(vec![2.0, 3.0, 4.0], vec![3]);
        let b = make_f32(vec![3.0, 2.0, 0.5], vec![3]);
        let out = pow_tensor::<Float32Type>(&a, &b).unwrap();
        let data = out.data().typed_data::<f32>();
        assert!((data[0] - 8.0).abs() < 1e-5);
        assert!((data[1] - 9.0).abs() < 1e-5);
        assert!((data[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_pow_tensor_shape_mismatch() {
        let a = make_f32(vec![1.0, 2.0], vec![2]);
        let b = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(pow_tensor::<Float32Type>(&a, &b).is_err());
    }

    #[test]
    fn test_erf_values() {
        let input = make_f32(vec![0.0, 1.0, -1.0, 0.5], vec![4]);
        let out = erf::<Float32Type>(&input).unwrap();
        let data = out.data().typed_data::<f32>();
        assert!((data[0] - 0.0).abs() < 1e-5); // erf(0) = 0
        assert!((data[1] - 0.8427).abs() < 1e-3); // erf(1) ~ 0.8427
        assert!((data[2] - (-0.8427)).abs() < 1e-3); // erf(-1) ~ -0.8427
        assert!((data[3] - 0.5205).abs() < 1e-3); // erf(0.5) ~ 0.5205
    }

    #[test]
    fn test_reciprocal() {
        let input = make_f32(vec![1.0, 2.0, 4.0, 0.5], vec![4]);
        let out = reciprocal::<Float32Type>(&input).unwrap();
        let data = out.data().typed_data::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 0.5).abs() < 1e-6);
        assert!((data[2] - 0.25).abs() < 1e-6);
        assert!((data[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_cos_op() {
        let input = make_f32(
            vec![0.0, std::f32::consts::PI, std::f32::consts::FRAC_PI_2],
            vec![3],
        );
        let out = cos_op::<Float32Type>(&input).unwrap();
        let data = out.data().typed_data::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - (-1.0)).abs() < 1e-5);
        assert!((data[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sin_op() {
        let input = make_f32(
            vec![0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI],
            vec![3],
        );
        let out = sin_op::<Float32Type>(&input).unwrap();
        let data = out.data().typed_data::<f32>();
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[1] - 1.0).abs() < 1e-6);
        assert!((data[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_preserves_shape() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reciprocal::<Float32Type>(&input).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 3]);
    }
}

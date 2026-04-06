use arrow::array::ArrowPrimitiveType;
use arrow::tensor::Tensor;
use arrow_kernels_common::Result;
use num_traits::Float;

use crate::broadcast::broadcast_binary_op;

/// Element-wise modulo with broadcasting. Uses IEEE 754 fmod semantics.
pub fn modulo<T>(a: &Tensor<'_, T>, b: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    broadcast_binary_op(a, b, |x, y| x % y, "mod")
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::buffer::{Buffer, ScalarBuffer};
    use arrow::datatypes::Float32Type;

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_mod_same_shape() {
        let a = make_f32(vec![7.0, 10.0, 5.0], vec![3]);
        let b = make_f32(vec![3.0, 4.0, 5.0], vec![3]);
        let out = modulo::<Float32Type>(&a, &b).unwrap();
        let data = out.data().typed_data::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mod_broadcast() {
        let a = make_f32(vec![5.0, 7.0, 10.0, 13.0], vec![2, 2]);
        let b = make_f32(vec![3.0], vec![1]);
        let out = modulo::<Float32Type>(&a, &b).unwrap();
        let data = out.data().typed_data::<f32>();
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 1.0).abs() < 1e-6);
        assert!((data[2] - 1.0).abs() < 1e-6);
        assert!((data[3] - 1.0).abs() < 1e-6);
    }
}

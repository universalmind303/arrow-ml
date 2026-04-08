use arrow::array::ArrowPrimitiveType;
use arrow::tensor::Tensor;
use arrow_ml_common::Result;
use num_traits::Float;

use crate::broadcast::broadcast_binary_op;

pub fn add<T>(a: &Tensor<'_, T>, b: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    broadcast_binary_op(a, b, |x, y| x + y, "add")
}

pub fn sub<T>(a: &Tensor<'_, T>, b: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    broadcast_binary_op(a, b, |x, y| x - y, "sub")
}

pub fn mul<T>(a: &Tensor<'_, T>, b: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    broadcast_binary_op(a, b, |x, y| x * y, "mul")
}

pub fn div<T>(a: &Tensor<'_, T>, b: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    broadcast_binary_op(a, b, |x, y| x / y, "div")
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::Float32Type;

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
        let buffer = ScalarBuffer::<f32>::from(data).into_inner();
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_add_same_shape() {
        let a = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let b = make_f32(vec![4.0, 5.0, 6.0], vec![3]);
        let out = add::<Float32Type>(&a, &b).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub_broadcast() {
        // [2,3] - [3] -> [2,3]
        let a = make_f32(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]);
        let b = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let out = sub::<Float32Type>(&a, &b).unwrap();
        assert_eq!(out.shape().unwrap(), &[2, 3]);
        assert_eq!(
            out.data().typed_data::<f32>(),
            &[9.0, 18.0, 27.0, 39.0, 48.0, 57.0]
        );
    }

    #[test]
    fn test_mul_scalar() {
        let a = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let b = make_f32(vec![2.0], vec![1]);
        let out = mul::<Float32Type>(&a, &b).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_div_broadcast() {
        let a = make_f32(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        let b = make_f32(vec![2.0, 5.0], vec![2]);
        let out = div::<Float32Type>(&a, &b).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[5.0, 4.0, 15.0, 8.0]);
    }

    #[test]
    fn test_div_by_zero() {
        let a = make_f32(vec![1.0, -1.0, 0.0], vec![3]);
        let b = make_f32(vec![0.0, 0.0, 0.0], vec![3]);
        let out = div::<Float32Type>(&a, &b).unwrap();
        let data = out.data().typed_data::<f32>();
        assert!(data[0].is_infinite() && data[0] > 0.0);
        assert!(data[1].is_infinite() && data[1] < 0.0);
        assert!(data[2].is_nan());
    }

    #[test]
    fn test_add_incompatible() {
        let a = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let b = make_f32(vec![1.0, 2.0], vec![2]);
        assert!(add::<Float32Type>(&a, &b).is_err());
    }
}

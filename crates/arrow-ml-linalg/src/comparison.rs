use arrow::array::ArrowPrimitiveType;
use arrow::datatypes::UInt8Type;
use arrow::tensor::Tensor;
use arrow_ml_common::Result;

use crate::broadcast::broadcast_binary_op;

pub fn equal<T>(a: &Tensor<'_, T>, b: &Tensor<'_, T>) -> Result<Tensor<'static, UInt8Type>>
where
    T: ArrowPrimitiveType,
    T::Native: PartialEq + Copy,
{
    broadcast_binary_op(a, b, |x, y| if x == y { 1u8 } else { 0u8 }, "equal")
}

pub fn less<T>(a: &Tensor<'_, T>, b: &Tensor<'_, T>) -> Result<Tensor<'static, UInt8Type>>
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd + Copy,
{
    broadcast_binary_op(a, b, |x, y| if x < y { 1u8 } else { 0u8 }, "less")
}

pub fn greater<T>(a: &Tensor<'_, T>, b: &Tensor<'_, T>) -> Result<Tensor<'static, UInt8Type>>
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd + Copy,
{
    broadcast_binary_op(a, b, |x, y| if x > y { 1u8 } else { 0u8 }, "greater")
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::Float32Type;

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
        let buffer = ScalarBuffer::<f32>::from(data).into_inner();
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_equal_same_shape() {
        let a = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let b = make_f32(vec![1.0, 5.0, 3.0], vec![3]);
        let out = equal::<Float32Type>(&a, &b).unwrap();
        assert_eq!(out.data().typed_data::<u8>(), &[1, 0, 1]);
    }

    #[test]
    fn test_equal_broadcast() {
        let a = make_f32(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3]);
        let b = make_f32(vec![2.0], vec![1]);
        let out = equal::<Float32Type>(&a, &b).unwrap();
        assert_eq!(out.shape().unwrap(), &[2, 3]);
        assert_eq!(out.data().typed_data::<u8>(), &[0, 1, 0, 0, 1, 0]);
    }

    #[test]
    fn test_less() {
        let a = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let b = make_f32(vec![2.0, 2.0, 2.0], vec![3]);
        let out = less::<Float32Type>(&a, &b).unwrap();
        assert_eq!(out.data().typed_data::<u8>(), &[1, 0, 0]);
    }

    #[test]
    fn test_greater() {
        let a = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let b = make_f32(vec![2.0, 2.0, 2.0], vec![3]);
        let out = greater::<Float32Type>(&a, &b).unwrap();
        assert_eq!(out.data().typed_data::<u8>(), &[0, 0, 1]);
    }

    #[test]
    fn test_comparison_broadcast_row_col() {
        // [3,1] vs [1,3]
        let a = make_f32(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let b = make_f32(vec![2.0, 1.0, 3.0], vec![1, 3]);
        let out = less::<Float32Type>(&a, &b).unwrap();
        assert_eq!(out.shape().unwrap(), &[3, 3]);
        assert_eq!(
            out.data().typed_data::<u8>(),
            &[
                1, 0, 1, // 1 < [2,1,3]
                0, 0, 1, // 2 < [2,1,3]
                0, 0, 0, // 3 < [2,1,3]
            ]
        );
    }
}

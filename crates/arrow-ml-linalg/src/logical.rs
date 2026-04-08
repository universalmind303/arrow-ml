use arrow::buffer::Buffer;
use arrow::datatypes::UInt8Type;
use arrow::tensor::Tensor;
use arrow_ml_common::{KernelError, Result};

use crate::broadcast::broadcast_binary_op;

pub fn logical_not(input: &Tensor<'_, UInt8Type>) -> Result<Tensor<'static, UInt8Type>> {
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("logical_not: tensor has no shape".into()))?
        .to_vec();
    let data: &[u8] = input.data().typed_data();
    let out: Vec<u8> = data
        .iter()
        .map(|&v| if v == 0 { 1u8 } else { 0u8 })
        .collect();
    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(shape), None).map_err(KernelError::from)
}

pub fn logical_and(
    a: &Tensor<'_, UInt8Type>,
    b: &Tensor<'_, UInt8Type>,
) -> Result<Tensor<'static, UInt8Type>> {
    broadcast_binary_op(
        a,
        b,
        |x, y| {
            if x != 0 && y != 0 {
                1u8
            } else {
                0u8
            }
        },
        "logical_and",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::buffer::ScalarBuffer;

    fn make_u8(data: Vec<u8>, shape: Vec<usize>) -> Tensor<'static, UInt8Type> {
        let buffer = Buffer::from(ScalarBuffer::<u8>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_logical_not() {
        let input = make_u8(vec![1, 0, 1, 0], vec![4]);
        let out = logical_not(&input).unwrap();
        assert_eq!(out.data().typed_data::<u8>(), &[0, 1, 0, 1]);
    }

    #[test]
    fn test_logical_not_nonbinary() {
        // Any nonzero value is truthy
        let input = make_u8(vec![0, 1, 2, 255], vec![4]);
        let out = logical_not(&input).unwrap();
        assert_eq!(out.data().typed_data::<u8>(), &[1, 0, 0, 0]);
    }

    #[test]
    fn test_logical_and_same_shape() {
        let a = make_u8(vec![1, 1, 0, 0], vec![4]);
        let b = make_u8(vec![1, 0, 1, 0], vec![4]);
        let out = logical_and(&a, &b).unwrap();
        assert_eq!(out.data().typed_data::<u8>(), &[1, 0, 0, 0]);
    }

    #[test]
    fn test_logical_and_broadcast() {
        let a = make_u8(vec![1, 0, 1], vec![3]);
        let b = make_u8(vec![1], vec![1]);
        let out = logical_and(&a, &b).unwrap();
        assert_eq!(out.data().typed_data::<u8>(), &[1, 0, 1]);
    }

    #[test]
    fn test_logical_preserves_shape() {
        let input = make_u8(vec![1, 0, 1, 0, 1, 0], vec![2, 3]);
        let out = logical_not(&input).unwrap();
        assert_eq!(out.shape().unwrap(), &[2, 3]);
    }
}

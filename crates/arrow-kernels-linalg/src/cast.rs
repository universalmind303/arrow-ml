use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_kernels_common::{KernelError, Result};
use num_traits::NumCast;

pub fn cast_tensor<A, B>(input: &Tensor<'_, A>) -> Result<Tensor<'static, B>>
where
    A: ArrowPrimitiveType,
    A::Native: NumCast + Copy,
    B: ArrowPrimitiveType,
    B::Native: NumCast + Copy,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("cast: tensor has no shape".into()))?
        .to_vec();
    let data: &[A::Native] = input.data().typed_data();
    let mut out = Vec::with_capacity(data.len());
    for &v in data {
        let converted = <B::Native as NumCast>::from(v).ok_or_else(|| {
            KernelError::InvalidArgument("cast: value cannot be represented in target type".into())
        })?;
        out.push(converted);
    }
    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(shape), None).map_err(KernelError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::{Float32Type, Float64Type, Int32Type};

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    fn make_i32(data: Vec<i32>, shape: Vec<usize>) -> Tensor<'static, Int32Type> {
        let buffer = Buffer::from(ScalarBuffer::<i32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_cast_f32_to_f64() {
        let input = make_f32(vec![1.0, 2.5, -3.0], vec![3]);
        let out: Tensor<Float64Type> = cast_tensor(&input).unwrap();
        assert_eq!(out.shape().unwrap(), &[3]);
        let data = out.data().typed_data::<f64>();
        assert!((data[0] - 1.0).abs() < 1e-10);
        assert!((data[1] - 2.5).abs() < 1e-10);
        assert!((data[2] - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cast_i32_to_f32() {
        let input = make_i32(vec![1, 2, -3, 0], vec![4]);
        let out: Tensor<Float32Type> = cast_tensor(&input).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[1.0, 2.0, -3.0, 0.0]);
    }

    #[test]
    fn test_cast_f32_to_i32() {
        let input = make_f32(vec![1.0, 2.9, -3.1], vec![3]);
        let out: Tensor<Int32Type> = cast_tensor(&input).unwrap();
        let data = out.data().typed_data::<i32>();
        assert_eq!(data[0], 1);
        assert_eq!(data[1], 2); // truncates toward zero
        assert_eq!(data[2], -3);
    }

    #[test]
    fn test_cast_preserves_shape() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out: Tensor<Float64Type> = cast_tensor(&input).unwrap();
        assert_eq!(out.shape().unwrap(), &[2, 3]);
    }
}

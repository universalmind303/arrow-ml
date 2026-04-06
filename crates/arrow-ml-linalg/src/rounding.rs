use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_ml_common::{KernelError, Result};
use num_traits::Float;

pub fn floor<T>(input: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("floor: tensor has no shape".into()))?
        .to_vec();
    let data: &[T::Native] = input.data().typed_data();
    let out: Vec<T::Native> = data.iter().map(|&v| v.floor()).collect();
    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(shape), None).map_err(KernelError::from)
}

pub fn ceil<T>(input: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("ceil: tensor has no shape".into()))?
        .to_vec();
    let data: &[T::Native] = input.data().typed_data();
    let out: Vec<T::Native> = data.iter().map(|&v| v.ceil()).collect();
    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(shape), None).map_err(KernelError::from)
}

pub fn round<T>(input: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("round: tensor has no shape".into()))?
        .to_vec();
    let data: &[T::Native] = input.data().typed_data();
    let out: Vec<T::Native> = data.iter().map(|&v| v.round()).collect();
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
    fn test_floor() {
        let input = make_f32(vec![1.5, 2.3, -1.7, 0.0], vec![4]);
        let out = floor::<Float32Type>(&input).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[1.0, 2.0, -2.0, 0.0]);
    }

    #[test]
    fn test_ceil() {
        let input = make_f32(vec![1.5, 2.3, -1.7, 0.0], vec![4]);
        let out = ceil::<Float32Type>(&input).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[2.0, 3.0, -1.0, 0.0]);
    }

    #[test]
    fn test_round() {
        let input = make_f32(vec![1.5, 2.3, -1.7, 0.4], vec![4]);
        let out = round::<Float32Type>(&input).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[2.0, 2.0, -2.0, 0.0]);
    }

    #[test]
    fn test_floor_preserves_shape() {
        let input = make_f32(vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5], vec![2, 3]);
        let out = floor::<Float32Type>(&input).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 3]);
    }
}

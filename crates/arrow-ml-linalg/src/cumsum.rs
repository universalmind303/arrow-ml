use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_ml_common::{KernelError, Result};
use num_traits::Float;

pub fn cumsum<T>(input: &Tensor<'_, T>, axis: i64) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("cumsum: tensor has no shape".into()))?;
    let ndim = shape.len();
    if ndim == 0 {
        return Err(KernelError::InvalidArgument(
            "cumsum: tensor must be at least 1D".into(),
        ));
    }

    let axis = if axis < 0 { ndim as i64 + axis } else { axis };
    if axis < 0 || axis >= ndim as i64 {
        return Err(KernelError::InvalidArgument(format!(
            "cumsum: axis {} out of range for {}D tensor",
            axis, ndim
        )));
    }
    let axis = axis as usize;

    let outer_size: usize = shape[..axis].iter().product();
    let dim_size = shape[axis];
    let inner_size: usize = shape[axis + 1..].iter().product();
    let outer_size = if outer_size == 0 { 1 } else { outer_size };
    let inner_size = if inner_size == 0 { 1 } else { inner_size };

    let data: &[T::Native] = input.data().typed_data();
    let mut out = data.to_vec();

    for o in 0..outer_size {
        for i in 0..inner_size {
            for d in 1..dim_size {
                let curr = o * dim_size * inner_size + d * inner_size + i;
                let prev = o * dim_size * inner_size + (d - 1) * inner_size + i;
                out[curr] = out[prev] + out[curr];
            }
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(shape.to_vec()), None).map_err(KernelError::from)
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
    fn test_cumsum_1d() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = cumsum::<Float32Type>(&input, 0).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_cumsum_2d_axis0() {
        // [[1,2,3],[4,5,6]] -> cumsum axis 0 -> [[1,2,3],[5,7,9]]
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = cumsum::<Float32Type>(&input, 0).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_cumsum_2d_axis1() {
        // [[1,2,3],[4,5,6]] -> cumsum axis 1 -> [[1,3,6],[4,9,15]]
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = cumsum::<Float32Type>(&input, 1).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    }

    #[test]
    fn test_cumsum_negative_axis() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = cumsum::<Float32Type>(&input, -1).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0]); // same as axis=1
    }

    #[test]
    fn test_cumsum_preserves_shape() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = cumsum::<Float32Type>(&input, 0).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 3]);
    }
}

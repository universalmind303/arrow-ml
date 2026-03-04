use arrow::array::{ArrowPrimitiveType, Int64Array};
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_kernels_common::{KernelError, Result};
use num_traits::{Float, Zero};

/// Return the shape of a tensor as an Int64Array.
pub fn tensor_shape<T>(input: &Tensor<'_, T>) -> Result<Int64Array>
where
    T: ArrowPrimitiveType,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("tensor_shape: tensor has no shape".into()))?;
    let values: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    Ok(Int64Array::from(values))
}

/// Create a tensor filled with a constant value.
pub fn constant_of_shape<T>(shape: &[usize], value: T::Native) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    if shape.is_empty() {
        return Err(KernelError::InvalidArgument(
            "constant_of_shape: shape must not be empty".into(),
        ));
    }
    let total: usize = shape.iter().product();
    let out = vec![value; total];
    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(shape.to_vec()), None).map_err(KernelError::from)
}

/// Create a 1D tensor with values from start to limit (exclusive) with step delta.
pub fn range_op<T>(start: T::Native, limit: T::Native, delta: T::Native) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    if delta == T::Native::zero() {
        return Err(KernelError::InvalidArgument("range_op: delta cannot be zero".into()));
    }
    let mut out = Vec::new();
    let mut val = start;
    if delta > T::Native::zero() {
        while val < limit {
            out.push(val);
            val = val + delta;
        }
    } else {
        while val > limit {
            out.push(val);
            val = val + delta;
        }
    }
    let len = out.len();
    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(vec![len]), None).map_err(KernelError::from)
}

/// Repeat a tensor along each axis by the given number of times.
pub fn tile<T>(
    input: &Tensor<'_, T>,
    repeats: &[usize],
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let in_shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("tile: tensor has no shape".into()))?;
    let ndim = in_shape.len();

    if repeats.len() != ndim {
        return Err(KernelError::ShapeMismatch {
            operation: "tile",
            expected: format!("repeats length {ndim}"),
            actual: format!("repeats length {}", repeats.len()),
        });
    }

    let out_shape: Vec<usize> = in_shape.iter().zip(repeats.iter()).map(|(&d, &r)| d * r).collect();
    let total: usize = out_shape.iter().product();
    let data: &[T::Native] = input.data().typed_data();

    let mut out = Vec::with_capacity(total);

    // Iterate all output coordinates, map to input using modular indexing
    let mut coords = vec![0usize; ndim];
    for _ in 0..total {
        let mut flat = 0;
        let mut stride = 1;
        for d in (0..ndim).rev() {
            flat += (coords[d] % in_shape[d]) * stride;
            stride *= in_shape[d];
        }
        out.push(data[flat]);

        // Increment coords
        for d in (0..ndim).rev() {
            coords[d] += 1;
            if coords[d] < out_shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(out_shape), None).map_err(KernelError::from)
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
    fn test_tensor_shape() {
        let input = make_f32(vec![1.0; 24], vec![2, 3, 4]);
        let shape = tensor_shape::<Float32Type>(&input).unwrap();
        assert_eq!(shape.len(), 3);
        assert_eq!(shape.value(0), 2);
        assert_eq!(shape.value(1), 3);
        assert_eq!(shape.value(2), 4);
    }

    #[test]
    fn test_constant_of_shape() {
        let out = constant_of_shape::<Float32Type>(&[2, 3], 7.0).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 3]);
        let data = out.data().typed_data::<f32>();
        assert!(data.iter().all(|&v| v == 7.0));
    }

    #[test]
    fn test_range_op_positive() {
        let out = range_op::<Float32Type>(0.0, 5.0, 1.0).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_range_op_step() {
        let out = range_op::<Float32Type>(1.0, 10.0, 3.0).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[1.0, 4.0, 7.0]);
    }

    #[test]
    fn test_range_op_negative_delta() {
        let out = range_op::<Float32Type>(5.0, 0.0, -1.0).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_range_op_zero_delta() {
        assert!(range_op::<Float32Type>(0.0, 5.0, 0.0).is_err());
    }

    #[test]
    fn test_tile_1d() {
        let input = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let out = tile::<Float32Type>(&input, &[3]).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![9]);
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_tile_2d() {
        // [[1,2],[3,4]] tiled 2x3 -> shape [4, 6]
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let out = tile::<Float32Type>(&input, &[2, 3]).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![4, 6]);
        let data = out.data().typed_data::<f32>();
        // Row 0: [1,2,1,2,1,2]
        assert_eq!(&data[0..6], &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        // Row 1: [3,4,3,4,3,4]
        assert_eq!(&data[6..12], &[3.0, 4.0, 3.0, 4.0, 3.0, 4.0]);
        // Row 2 = Row 0 (tiled along axis 0)
        assert_eq!(&data[12..18], &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_tile_repeats_mismatch() {
        let input = make_f32(vec![1.0, 2.0], vec![2]);
        assert!(tile::<Float32Type>(&input, &[2, 3]).is_err());
    }
}

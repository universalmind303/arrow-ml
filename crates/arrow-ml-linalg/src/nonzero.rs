use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::datatypes::Int64Type;
use arrow::tensor::Tensor;
use arrow_ml_common::{KernelError, Result};
use num_traits::Zero;

/// Return the coordinates of nonzero elements as an Int64 tensor of shape `(ndim, num_nonzero)`.
/// Matches ONNX NonZero spec.
pub fn nonzero<T>(input: &Tensor<'_, T>) -> Result<Tensor<'static, Int64Type>>
where
    T: ArrowPrimitiveType,
    T::Native: PartialEq + Zero + Copy,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("nonzero: tensor has no shape".into()))?;
    let ndim = shape.len();
    let data: &[T::Native] = input.data().typed_data();
    let total: usize = shape.iter().product();

    // Collect coordinates of nonzero elements
    let mut coord_lists: Vec<Vec<i64>> = vec![vec![]; ndim];
    let mut coords = vec![0usize; ndim];

    #[allow(clippy::needless_range_loop)]
    for i in 0..total {
        if !data[i].is_zero() {
            for d in 0..ndim {
                coord_lists[d].push(coords[d] as i64);
            }
        }

        // Increment coords
        for d in (0..ndim).rev() {
            coords[d] += 1;
            if coords[d] < shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    let num_nonzero = coord_lists.first().map_or(0, |v| v.len());

    if num_nonzero == 0 {
        // Arrow Tensor doesn't support zero-size dimensions.
        // Return a shape [ndim, 1] tensor filled with -1 as sentinel for "no nonzero elements".
        let sentinel = vec![-1i64; ndim];
        let buf = Buffer::from_vec(sentinel);
        return Tensor::new_row_major(buf, Some(vec![ndim, 1]), None).map_err(KernelError::from);
    }

    // Output shape: (ndim, num_nonzero) — row-major means we lay out dim0 coords, then dim1 coords, etc.
    let mut flat: Vec<i64> = Vec::with_capacity(ndim * num_nonzero);
    for dim_coords in &coord_lists {
        flat.extend_from_slice(dim_coords);
    }

    let buf = Buffer::from_vec(flat);
    Tensor::new_row_major(buf, Some(vec![ndim, num_nonzero]), None).map_err(KernelError::from)
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
    fn test_nonzero_1d() {
        let input = make_f32(vec![0.0, 1.0, 0.0, 3.0, 0.0], vec![5]);
        let out = nonzero::<Float32Type>(&input).unwrap();
        assert_eq!(out.shape().unwrap(), &[1, 2]); // 1 dim, 2 nonzero
        let data = out.data().typed_data::<i64>();
        assert_eq!(data, &[1, 3]); // indices 1 and 3
    }

    #[test]
    fn test_nonzero_2d() {
        // [[0, 1], [2, 0]]
        let input = make_f32(vec![0.0, 1.0, 2.0, 0.0], vec![2, 2]);
        let out = nonzero::<Float32Type>(&input).unwrap();
        assert_eq!(out.shape().unwrap(), &[2, 2]); // 2 dims, 2 nonzero
        let data = out.data().typed_data::<i64>();
        // Row indices: [0, 1], Col indices: [1, 0]
        assert_eq!(data, &[0, 1, 1, 0]);
    }

    #[test]
    fn test_nonzero_all_zero() {
        let input = make_f32(vec![0.0, 0.0, 0.0], vec![3]);
        let out = nonzero::<Float32Type>(&input).unwrap();
        // Arrow Tensor can't represent zero-size dims.
        // Returns [ndim, 1] with sentinel -1 values.
        assert_eq!(out.shape().unwrap(), &[1, 1]);
        assert_eq!(out.data().typed_data::<i64>(), &[-1]);
    }

    #[test]
    fn test_nonzero_all_nonzero() {
        let input = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let out = nonzero::<Float32Type>(&input).unwrap();
        assert_eq!(out.shape().unwrap(), &[1, 3]);
        assert_eq!(out.data().typed_data::<i64>(), &[0, 1, 2]);
    }
}

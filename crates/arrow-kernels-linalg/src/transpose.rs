use arrow::array::ArrowPrimitiveType;
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::tensor::Tensor;
use arrow_kernels_common::KernelError;
use arrow_kernels_common::Result;
use num_traits::{Float, Zero};

/// Casts a Tensor's buffer to a typed slice.
fn tensor_as_slice<'a, T: ArrowPrimitiveType>(tensor: &'a Tensor<'_, T>) -> &'a [T::Native] {
    let buf = tensor.data();
    
    let ptr = buf.as_ptr() as *const T::Native;
    let len = buf.len() / std::mem::size_of::<T::Native>();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// Transpose a 2D tensor: (m x n) -> (n x m).
///
/// Returns a new row-major tensor with the transposed data.
pub fn transpose<T>(tensor: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let shape = tensor.shape().ok_or_else(|| {
        KernelError::InvalidArgument("transpose: tensor has no shape".to_string())
    })?;
    if shape.len() != 2 {
        return Err(KernelError::InvalidArgument(format!(
            "transpose: expected 2D tensor, got {}D",
            shape.len()
        )));
    }
    let (m, n) = (shape[0], shape[1]);
    let data = tensor_as_slice(tensor);

    let mut result = vec![T::Native::zero(); m * n];
    for i in 0..m {
        for j in 0..n {
            result[j * m + i] = data[i * n + j];
        }
    }

    let buffer = Buffer::from(ScalarBuffer::<T::Native>::from(result).into_inner());
    Tensor::new_row_major(buffer, Some(vec![n, m]), None).map_err(KernelError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::Float32Type;

    fn make_tensor_2d<T: ArrowPrimitiveType>(
        data: Vec<T::Native>,
        rows: usize,
        cols: usize,
    ) -> Tensor<'static, T> {
        let buffer = Buffer::from(ScalarBuffer::<T::Native>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(vec![rows, cols]), None).unwrap()
    }

    #[test]
    fn test_transpose_2x3() {
        // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
        let a = make_tensor_2d::<Float32Type>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let t = transpose(&a).unwrap();

        let shape = t.shape().unwrap();
        assert_eq!(shape, &vec![3, 2]);

        let data = tensor_as_slice(&t);
        assert_eq!(data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_square() {
        // [[1, 2], [3, 4]] -> [[1, 3], [2, 4]]
        let a = make_tensor_2d::<Float32Type>(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let t = transpose(&a).unwrap();

        let shape = t.shape().unwrap();
        assert_eq!(shape, &vec![2, 2]);

        let data = tensor_as_slice(&t);
        assert_eq!(data, &[1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_transpose_roundtrip() {
        let a = make_tensor_2d::<Float32Type>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let t = transpose(&a).unwrap();
        let tt = transpose(&t).unwrap();

        let original = tensor_as_slice(&a);
        let roundtrip = tensor_as_slice(&tt);
        assert_eq!(original, roundtrip);
    }
}

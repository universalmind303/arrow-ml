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

/// Transpose a tensor by permuting its axes.
///
/// `perm` must be a permutation of `[0..ndim)`. For a 2D tensor with `perm = [1, 0]`,
/// this is equivalent to `transpose`.
pub fn transpose_axes<T>(input: &Tensor<'_, T>, perm: &[usize]) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let shape = input.shape().ok_or_else(|| {
        KernelError::InvalidArgument("transpose_axes: tensor has no shape".to_string())
    })?;
    let ndim = shape.len();

    if perm.len() != ndim {
        return Err(KernelError::InvalidArgument(format!(
            "transpose_axes: perm length {} does not match tensor rank {}",
            perm.len(),
            ndim,
        )));
    }

    // Validate perm is a valid permutation
    let mut seen = vec![false; ndim];
    for &p in perm {
        if p >= ndim {
            return Err(KernelError::InvalidArgument(format!(
                "transpose_axes: perm value {p} out of range for {ndim}D tensor",
            )));
        }
        if seen[p] {
            return Err(KernelError::InvalidArgument(format!(
                "transpose_axes: duplicate axis {p} in perm",
            )));
        }
        seen[p] = true;
    }

    // Compute output shape
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();

    // Compute input strides (row-major)
    let mut in_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * shape[i + 1];
    }

    let total: usize = out_shape.iter().product();
    let data = tensor_as_slice(input);
    let mut result = Vec::with_capacity(total);

    let mut coords = vec![0usize; ndim];
    for _ in 0..total {
        // Map output coords to input: input_idx = sum(coords[i] * in_strides[perm[i]])
        let mut flat = 0;
        for i in 0..ndim {
            flat += coords[i] * in_strides[perm[i]];
        }
        result.push(data[flat]);

        for d in (0..ndim).rev() {
            coords[d] += 1;
            if coords[d] < out_shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    let buffer = Buffer::from(ScalarBuffer::<T::Native>::from(result).into_inner());
    Tensor::new_row_major(buffer, Some(out_shape), None).map_err(KernelError::from)
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

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_transpose_axes_2d() {
        // Should produce same result as transpose for 2D
        let a = make_tensor_2d::<Float32Type>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let t = transpose_axes(&a, &[1, 0]).unwrap();
        assert_eq!(t.shape().unwrap(), &[3, 2]);
        let data = tensor_as_slice(&t);
        assert_eq!(data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_axes_3d() {
        // Shape [2,3,4] with perm [0,2,1] -> [2,4,3]
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let a = make_f32(data, vec![2, 3, 4]);
        let t = transpose_axes(&a, &[0, 2, 1]).unwrap();
        assert_eq!(t.shape().unwrap(), &[2, 4, 3]);
        let out = tensor_as_slice(&t);
        // Element at output [0,0,0] = input [0,0,0] = 0
        assert_eq!(out[0], 0.0);
        // Element at output [0,1,0] = input [0,0,1] = 1
        assert_eq!(out[3], 1.0);
        // Element at output [0,0,1] = input [0,1,0] = 4
        assert_eq!(out[1], 4.0);
    }

    #[test]
    fn test_transpose_axes_4d_attention() {
        // Typical attention head reshape: (1,seq,heads,dim) -> (1,heads,seq,dim)
        // Shape [1,2,3,4] with perm [0,2,1,3] -> [1,3,2,4]
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let a = make_f32(data, vec![1, 2, 3, 4]);
        let t = transpose_axes(&a, &[0, 2, 1, 3]).unwrap();
        assert_eq!(t.shape().unwrap(), &[1, 3, 2, 4]);
    }

    #[test]
    fn test_transpose_axes_roundtrip() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let a = make_f32(data.clone(), vec![2, 3, 4]);
        let t = transpose_axes(&a, &[2, 0, 1]).unwrap();
        // Inverse of [2,0,1] is [1,2,0]
        let tt = transpose_axes(&t, &[1, 2, 0]).unwrap();
        assert_eq!(tt.shape().unwrap(), &[2, 3, 4]);
        assert_eq!(tensor_as_slice(&tt), &data[..]);
    }

    #[test]
    fn test_transpose_axes_identity() {
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let a = make_f32(data.clone(), vec![2, 3]);
        let t = transpose_axes(&a, &[0, 1]).unwrap();
        assert_eq!(t.shape().unwrap(), &[2, 3]);
        assert_eq!(tensor_as_slice(&t), &data[..]);
    }

    #[test]
    fn test_transpose_axes_invalid_perm() {
        let a = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert!(transpose_axes(&a, &[0]).is_err()); // wrong length
        assert!(transpose_axes(&a, &[0, 2]).is_err()); // out of range
        assert!(transpose_axes(&a, &[0, 0]).is_err()); // duplicate
    }
}

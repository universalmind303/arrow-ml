use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_ml_common::{KernelError, Result};

/// Broadcast (expand) a tensor to a target shape following numpy broadcasting rules.
///
/// Rules:
/// - Dimensions are compared from the right
/// - A dimension of size 1 can be broadcast to any size
/// - Missing dimensions on the left are treated as 1
pub fn expand<T>(input: &Tensor<'_, T>, target_shape: &[usize]) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let in_shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("expand: tensor has no shape".into()))?;

    let ndim = target_shape.len().max(in_shape.len());

    // Pad input shape with leading 1s to match ndim
    let mut padded_in = vec![1usize; ndim];
    let offset = ndim - in_shape.len();
    for (i, &d) in in_shape.iter().enumerate() {
        padded_in[offset + i] = d;
    }

    // Pad target shape with leading 1s
    let mut padded_target = vec![1usize; ndim];
    let offset_t = ndim - target_shape.len();
    for (i, &d) in target_shape.iter().enumerate() {
        padded_target[offset_t + i] = d;
    }

    // Validate broadcast compatibility and compute output shape
    let mut out_shape = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let s = padded_in[i];
        let t = padded_target[i];
        if s == t {
            out_shape.push(s);
        } else if s == 1 {
            out_shape.push(t);
        } else if t == 1 {
            out_shape.push(s);
        } else {
            return Err(KernelError::ShapeMismatch {
                operation: "expand",
                expected: format!("{:?}", target_shape),
                actual: format!("{:?}", in_shape),
            });
        }
    }

    let total: usize = out_shape.iter().product();
    let data: &[T::Native] = input.data().typed_data();

    // Compute input strides (0 for broadcast dims)
    let mut in_strides = vec![0usize; ndim];
    let mut stride = 1usize;
    for i in (0..ndim).rev() {
        if padded_in[i] == 1 {
            in_strides[i] = 0; // broadcast
        } else {
            in_strides[i] = stride;
        }
        stride *= padded_in[i];
    }

    let mut out = Vec::with_capacity(total);

    // Iterate over all output indices
    let mut coords = vec![0usize; ndim];
    for _ in 0..total {
        // Compute flat input index
        let mut flat = 0;
        for d in 0..ndim {
            flat += coords[d] * in_strides[d];
        }
        out.push(data[flat]);

        // Increment coords (rightmost first)
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
    fn test_expand_scalar_to_vector() {
        let input = make_f32(vec![5.0], vec![1]);
        let out = expand(&input, &[4]).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![4]);
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_expand_row_to_matrix() {
        // [1, 3] -> [4, 3]
        let input = make_f32(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let out = expand(&input, &[4, 3]).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![4, 3]);
        let data = out.data().typed_data::<f32>();
        for row in 0..4 {
            assert_eq!(data[row * 3], 1.0);
            assert_eq!(data[row * 3 + 1], 2.0);
            assert_eq!(data[row * 3 + 2], 3.0);
        }
    }

    #[test]
    fn test_expand_col_to_matrix() {
        // [3, 1] -> [3, 4]
        let input = make_f32(vec![10.0, 20.0, 30.0], vec![3, 1]);
        let out = expand(&input, &[3, 4]).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![3, 4]);
        let data = out.data().typed_data::<f32>();
        assert_eq!(&data[0..4], &[10.0, 10.0, 10.0, 10.0]);
        assert_eq!(&data[4..8], &[20.0, 20.0, 20.0, 20.0]);
        assert_eq!(&data[8..12], &[30.0, 30.0, 30.0, 30.0]);
    }

    #[test]
    fn test_expand_add_dim() {
        // [3] -> [2, 3] (input gets leading 1 prepended)
        let input = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let out = expand(&input, &[2, 3]).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 3]);
        let data = out.data().typed_data::<f32>();
        assert_eq!(&data[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&data[3..6], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_expand_incompatible() {
        let input = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(expand(&input, &[4]).is_err());
    }
}

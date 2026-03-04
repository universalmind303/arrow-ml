use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_kernels_common::{KernelError, Result};

/// Concatenate tensors along the given axis.
///
/// All tensors must have the same shape except on the concatenation axis.
pub fn concat<T>(tensors: &[&Tensor<'_, T>], axis: usize) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    if tensors.is_empty() {
        return Err(KernelError::InvalidArgument(
            "concat: need at least one tensor".into(),
        ));
    }

    let first_shape = tensors[0]
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("concat: tensor has no shape".into()))?;
    let ndim = first_shape.len();

    if axis >= ndim {
        return Err(KernelError::InvalidArgument(format!(
            "concat: axis {axis} out of range for {ndim}D tensor"
        )));
    }

    for (i, t) in tensors.iter().enumerate().skip(1) {
        let s = t
            .shape()
            .ok_or_else(|| KernelError::InvalidArgument(format!("concat: tensor {i} has no shape")))?;
        if s.len() != ndim {
            return Err(KernelError::ShapeMismatch {
                operation: "concat",
                expected: format!("{ndim}D"),
                actual: format!("{}D (tensor {i})", s.len()),
            });
        }
        for d in 0..ndim {
            if d != axis && s[d] != first_shape[d] {
                return Err(KernelError::ShapeMismatch {
                    operation: "concat",
                    expected: format!("dim[{d}] = {}", first_shape[d]),
                    actual: format!("dim[{d}] = {} (tensor {i})", s[d]),
                });
            }
        }
    }

    let outer_size: usize = first_shape[..axis].iter().product();
    let inner_size: usize = first_shape[axis + 1..].iter().product();
    let axis_dims: Vec<usize> = tensors.iter().map(|t| t.shape().unwrap()[axis]).collect();
    let total_axis: usize = axis_dims.iter().sum();

    let mut out_shape = first_shape.to_vec();
    out_shape[axis] = total_axis;

    let total_elements = outer_size * total_axis * inner_size;
    let mut out = Vec::with_capacity(total_elements);

    let slices: Vec<&[T::Native]> = tensors
        .iter()
        .map(|t| t.data().typed_data::<T::Native>())
        .collect();

    for o in 0..outer_size {
        for (t_idx, data) in slices.iter().enumerate() {
            let chunk = axis_dims[t_idx] * inner_size;
            let start = o * chunk;
            out.extend_from_slice(&data[start..start + chunk]);
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(out_shape), None).map_err(KernelError::from)
}

/// Split a tensor into parts along the given axis.
///
/// `split_sizes` specifies the size of each output tensor along `axis`.
pub fn split<T>(
    input: &Tensor<'_, T>,
    split_sizes: &[usize],
    axis: usize,
) -> Result<Vec<Tensor<'static, T>>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("split: tensor has no shape".into()))?;
    let ndim = shape.len();

    if axis >= ndim {
        return Err(KernelError::InvalidArgument(format!(
            "split: axis {axis} out of range for {ndim}D tensor"
        )));
    }

    let total: usize = split_sizes.iter().sum();
    if total != shape[axis] {
        return Err(KernelError::ShapeMismatch {
            operation: "split",
            expected: format!("split sizes sum to {}", shape[axis]),
            actual: format!("split sizes sum to {total}"),
        });
    }

    let outer_size: usize = shape[..axis].iter().product();
    let inner_size: usize = shape[axis + 1..].iter().product();
    let data: &[T::Native] = input.data().typed_data();
    let full_axis = shape[axis];

    let mut results: Vec<(Vec<usize>, Vec<T::Native>, usize)> = split_sizes
        .iter()
        .map(|&sz| {
            let mut part_shape = shape.to_vec();
            part_shape[axis] = sz;
            let cap = outer_size * sz * inner_size;
            (part_shape, Vec::with_capacity(cap), sz)
        })
        .collect();

    for o in 0..outer_size {
        let row_start = o * full_axis * inner_size;
        let mut offset = 0;
        for part in results.iter_mut() {
            let chunk = part.2 * inner_size;
            part.1.extend_from_slice(&data[row_start + offset..row_start + offset + chunk]);
            offset += chunk;
        }
    }

    results
        .into_iter()
        .map(|(part_shape, part_data, _)| {
            let buf = Buffer::from_vec(part_data);
            Tensor::new_row_major(buf, Some(part_shape), None).map_err(KernelError::from)
        })
        .collect()
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
    fn test_concat_axis0() {
        let a = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = make_f32(vec![7.0, 8.0, 9.0], vec![1, 3]);
        let c = concat(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape().unwrap(), &vec![3, 3]);
        assert_eq!(c.data().typed_data::<f32>(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_concat_axis1() {
        let a = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = make_f32(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3]);
        let c = concat(&[&a, &b], 1).unwrap();
        assert_eq!(c.shape().unwrap(), &vec![2, 5]);
        assert_eq!(c.data().typed_data::<f32>(), &[1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn test_split_and_reconcat() {
        let orig = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]);
        let parts = split(&orig, &[2, 1], 0).unwrap();
        assert_eq!(parts[0].shape().unwrap(), &vec![2, 3]);
        assert_eq!(parts[1].shape().unwrap(), &vec![1, 3]);
        let refs: Vec<&Tensor<'_, Float32Type>> = parts.iter().collect();
        let recon = concat(&refs, 0).unwrap();
        assert_eq!(orig.data().typed_data::<f32>(), recon.data().typed_data::<f32>());
    }

    #[test]
    fn test_concat_shape_mismatch() {
        let a = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(concat(&[&a, &b], 0).is_err());
    }

    #[test]
    fn test_split_sizes_mismatch() {
        let a = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert!(split(&a, &[3, 1], 0).is_err());
    }

    #[test]
    fn test_3d_concat_axis1() {
        #[rustfmt::skip]
        let a = make_f32(vec![
            1.0, 2.0, 3.0,  4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,  10.0, 11.0, 12.0,
        ], vec![2, 2, 3]);
        #[rustfmt::skip]
        let b = make_f32(vec![
            13.0, 14.0, 15.0,
            16.0, 17.0, 18.0,
        ], vec![2, 1, 3]);
        let c = concat(&[&a, &b], 1).unwrap();
        assert_eq!(c.shape().unwrap(), &vec![2, 3, 3]);
        #[rustfmt::skip]
        let expected = &[
            1.0, 2.0, 3.0,  4.0, 5.0, 6.0,  13.0, 14.0, 15.0,
            7.0, 8.0, 9.0,  10.0, 11.0, 12.0,  16.0, 17.0, 18.0,
        ];
        assert_eq!(c.data().typed_data::<f32>(), expected);
    }
}

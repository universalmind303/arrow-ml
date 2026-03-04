use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::datatypes::Int64Type;
use arrow::tensor::Tensor;
use arrow_kernels_common::{KernelError, Result};

/// ONNX GatherElements: gather elements along an axis.
/// indices has same rank as input but may differ in the gather axis dimension.
/// For each position in indices, gather from input at that index along the axis.
pub fn gather_elements<T>(
    input: &Tensor<'_, T>,
    indices: &Tensor<'_, Int64Type>,
    axis: i64,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let in_shape = input
        .shape()
        .ok_or_else(|| {
            KernelError::InvalidArgument("gather_elements: input has no shape".into())
        })?;
    let idx_shape = indices
        .shape()
        .ok_or_else(|| {
            KernelError::InvalidArgument("gather_elements: indices has no shape".into())
        })?;
    let ndim = in_shape.len();

    if idx_shape.len() != ndim {
        return Err(KernelError::ShapeMismatch {
            operation: "gather_elements",
            expected: format!("{ndim}D indices"),
            actual: format!("{}D indices", idx_shape.len()),
        });
    }

    let axis = if axis < 0 { ndim as i64 + axis } else { axis };
    if axis < 0 || axis >= ndim as i64 {
        return Err(KernelError::InvalidArgument(format!(
            "gather_elements: axis {} out of range for {}D tensor",
            axis, ndim
        )));
    }
    let axis = axis as usize;

    let in_data: &[T::Native] = input.data().typed_data();
    let idx_data: &[i64] = indices.data().typed_data();

    // Compute strides for input
    let mut in_strides = vec![1usize; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
    }

    // Compute strides for indices
    let mut idx_strides = vec![1usize; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        idx_strides[d] = idx_strides[d + 1] * idx_shape[d + 1];
    }

    let total: usize = idx_shape.iter().product();
    let mut out = Vec::with_capacity(total);

    let mut coords = vec![0usize; ndim];
    for _ in 0..total {
        // Get index value at this position
        let mut flat_idx = 0;
        for d in 0..ndim {
            flat_idx += coords[d] * idx_strides[d];
        }
        let mut gather_idx = idx_data[flat_idx];
        if gather_idx < 0 {
            gather_idx += in_shape[axis] as i64;
        }
        if gather_idx < 0 || gather_idx >= in_shape[axis] as i64 {
            return Err(KernelError::InvalidArgument(format!(
                "gather_elements: index {} out of range for axis dim {}",
                idx_data[flat_idx], in_shape[axis]
            )));
        }

        // Build input flat index: same coords except axis dim uses gather_idx
        let mut in_flat = 0;
        for d in 0..ndim {
            if d == axis {
                in_flat += gather_idx as usize * in_strides[d];
            } else {
                in_flat += coords[d] * in_strides[d];
            }
        }
        out.push(in_data[in_flat]);

        // Increment coords
        for d in (0..ndim).rev() {
            coords[d] += 1;
            if coords[d] < idx_shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(idx_shape.to_vec()), None).map_err(KernelError::from)
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

    fn make_i64(data: Vec<i64>, shape: Vec<usize>) -> Tensor<'static, Int64Type> {
        let buffer = Buffer::from(ScalarBuffer::<i64>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_gather_elements_axis0() {
        // input: [[1,2],[3,4]], indices: [[0,0],[1,0]], axis=0
        // output: [[1,2],[3,2]]
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let indices = make_i64(vec![0, 0, 1, 0], vec![2, 2]);
        let out = gather_elements::<Float32Type>(&input, &indices, 0).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[1.0, 2.0, 3.0, 2.0]);
    }

    #[test]
    fn test_gather_elements_axis1() {
        // input: [[1,2,3],[4,5,6]], indices: [[2,0],[1,1]], axis=1
        // output: [[3,1],[5,5]]
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let indices = make_i64(vec![2, 0, 1, 1], vec![2, 2]);
        let out = gather_elements::<Float32Type>(&input, &indices, 1).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 2]);
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[3.0, 1.0, 5.0, 5.0]);
    }

    #[test]
    fn test_gather_elements_negative_index() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let indices = make_i64(vec![-1, -1], vec![1, 2]);
        let out = gather_elements::<Float32Type>(&input, &indices, 0).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[3.0, 4.0]); // last row
    }
}

use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::datatypes::Int64Type;
use arrow::tensor::Tensor;
use arrow_ml_common::{KernelError, Result};

/// ONNX ScatterElements: scatter updates into a copy of data at positions given by indices along an axis.
///
/// For each position in indices, set `output[..., indices[i], ...] = updates[i]`
/// where the indices[i] applies along the specified axis.
pub fn scatter_elements<T>(
    data: &Tensor<'_, T>,
    indices: &Tensor<'_, Int64Type>,
    updates: &Tensor<'_, T>,
    axis: i64,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let in_shape = data.shape().ok_or_else(|| {
        KernelError::InvalidArgument("scatter_elements: data has no shape".into())
    })?;
    let idx_shape = indices.shape().ok_or_else(|| {
        KernelError::InvalidArgument("scatter_elements: indices has no shape".into())
    })?;
    let upd_shape = updates.shape().ok_or_else(|| {
        KernelError::InvalidArgument("scatter_elements: updates has no shape".into())
    })?;
    let ndim = in_shape.len();

    if idx_shape.len() != ndim || upd_shape.len() != ndim {
        return Err(KernelError::ShapeMismatch {
            operation: "scatter_elements",
            expected: format!("{ndim}D indices and updates"),
            actual: format!("{}D indices, {}D updates", idx_shape.len(), upd_shape.len()),
        });
    }
    if idx_shape != upd_shape {
        return Err(KernelError::ShapeMismatch {
            operation: "scatter_elements",
            expected: format!("indices shape {:?}", idx_shape),
            actual: format!("updates shape {:?}", upd_shape),
        });
    }

    let axis = if axis < 0 { ndim as i64 + axis } else { axis };
    if axis < 0 || axis >= ndim as i64 {
        return Err(KernelError::InvalidArgument(format!(
            "scatter_elements: axis {} out of range for {}D tensor",
            axis, ndim
        )));
    }
    let axis = axis as usize;

    let in_data: &[T::Native] = data.data().typed_data();
    let idx_data: &[i64] = indices.data().typed_data();
    let upd_data: &[T::Native] = updates.data().typed_data();

    // Start with a copy of the input data
    let mut out: Vec<T::Native> = in_data.to_vec();

    // Compute strides for data (output)
    let mut in_strides = vec![1usize; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
    }

    // Compute strides for indices/updates
    let mut idx_strides = vec![1usize; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        idx_strides[d] = idx_strides[d + 1] * idx_shape[d + 1];
    }

    let total: usize = idx_shape.iter().product();
    let mut coords = vec![0usize; ndim];

    for _ in 0..total {
        let mut flat_idx = 0;
        for d in 0..ndim {
            flat_idx += coords[d] * idx_strides[d];
        }

        let mut scatter_idx = idx_data[flat_idx];
        if scatter_idx < 0 {
            scatter_idx += in_shape[axis] as i64;
        }
        if scatter_idx < 0 || scatter_idx >= in_shape[axis] as i64 {
            return Err(KernelError::InvalidArgument(format!(
                "scatter_elements: index {} out of range for axis dim {}",
                idx_data[flat_idx], in_shape[axis]
            )));
        }

        // Compute output flat index: same coords except axis uses scatter_idx
        let mut out_flat = 0;
        for d in 0..ndim {
            if d == axis {
                out_flat += scatter_idx as usize * in_strides[d];
            } else {
                out_flat += coords[d] * in_strides[d];
            }
        }
        out[out_flat] = upd_data[flat_idx];

        for d in (0..ndim).rev() {
            coords[d] += 1;
            if coords[d] < idx_shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(in_shape.to_vec()), None).map_err(KernelError::from)
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
    fn test_scatter_elements_axis0() {
        // data: [[1,2],[3,4]], indices: [[1,0]], updates: [[9,8]], axis=0
        // pos (0,0): idx=1 -> output[1,0] = 9
        // pos (0,1): idx=0 -> output[0,1] = 8
        // output: [[1,8],[9,4]]
        let data = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let indices = make_i64(vec![1, 0], vec![1, 2]);
        let updates = make_f32(vec![9.0, 8.0], vec![1, 2]);
        let out = scatter_elements::<Float32Type>(&data, &indices, &updates, 0).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[1.0, 8.0, 9.0, 4.0]);
    }

    #[test]
    fn test_scatter_elements_axis1() {
        // data: [[1,2,3],[4,5,6]], indices: [[2],[0]], updates: [[9],[8]], axis=1
        let data = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let indices = make_i64(vec![2, 0], vec![2, 1]);
        let updates = make_f32(vec![9.0, 8.0], vec![2, 1]);
        let out = scatter_elements::<Float32Type>(&data, &indices, &updates, 1).unwrap();
        assert_eq!(
            out.data().typed_data::<f32>(),
            &[1.0, 2.0, 9.0, 8.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_scatter_elements_negative_index() {
        let data = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let indices = make_i64(vec![-1], vec![1, 1]);
        let updates = make_f32(vec![99.0], vec![1, 1]);
        let out = scatter_elements::<Float32Type>(&data, &indices, &updates, 0).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[1.0, 2.0, 99.0, 4.0]);
    }

    #[test]
    fn test_scatter_elements_preserves_unscattered() {
        let data = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let indices = make_i64(vec![1], vec![1]);
        let updates = make_f32(vec![99.0], vec![1]);
        let out = scatter_elements::<Float32Type>(&data, &indices, &updates, 0).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[1.0, 99.0, 3.0]);
    }
}

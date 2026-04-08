use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::datatypes::Int64Type;
use arrow::tensor::Tensor;
use arrow_ml_common::{KernelError, Result};

/// ONNX ScatterND: scatter updates into data tensor at positions given by indices.
///
/// data shape: any
/// indices shape: (..., K) where K <= rank(data). The last dim indexes into data.
/// updates shape: indices.shape[:-1] + data.shape[K:]
pub fn scatter_nd<T>(
    data: &Tensor<'_, T>,
    indices: &Tensor<'_, Int64Type>,
    updates: &Tensor<'_, T>,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let data_shape = data
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("scatter_nd: data has no shape".into()))?;
    let idx_shape = indices
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("scatter_nd: indices has no shape".into()))?;

    if idx_shape.is_empty() {
        return Err(KernelError::InvalidArgument(
            "scatter_nd: indices must be at least 1D".into(),
        ));
    }

    let k = *idx_shape.last().unwrap(); // last dim of indices
    if k > data_shape.len() {
        return Err(KernelError::InvalidArgument(format!(
            "scatter_nd: indices last dim {} > data rank {}",
            k,
            data_shape.len()
        )));
    }

    let data_data: &[T::Native] = data.data().typed_data();
    let idx_data: &[i64] = indices.data().typed_data();
    let upd_data: &[T::Native] = updates.data().typed_data();

    // Number of scatter operations (product of indices shape except last dim)
    let num_updates: usize = idx_shape[..idx_shape.len() - 1].iter().product();
    let num_updates = if num_updates == 0 { 1 } else { num_updates };

    // Size of each scattered slice (product of data shape from index K onward)
    let slice_size: usize = data_shape[k..].iter().product();
    let slice_size = if slice_size == 0 { 1 } else { slice_size };

    // Compute data strides for the first K dimensions
    let mut data_strides = vec![1usize; data_shape.len()];
    for d in (0..data_shape.len().saturating_sub(1)).rev() {
        data_strides[d] = data_strides[d + 1] * data_shape[d + 1];
    }

    let mut out = data_data.to_vec();

    for u in 0..num_updates {
        // Compute flat offset into data for this update
        let idx_base = u * k;
        let mut flat_offset = 0usize;
        for j in 0..k {
            let mut idx_val = idx_data[idx_base + j];
            if idx_val < 0 {
                idx_val += data_shape[j] as i64;
            }
            if idx_val < 0 || idx_val >= data_shape[j] as i64 {
                return Err(KernelError::InvalidArgument(format!(
                    "scatter_nd: index {} out of range for dim {} of size {}",
                    idx_data[idx_base + j],
                    j,
                    data_shape[j]
                )));
            }
            flat_offset += idx_val as usize * data_strides[j];
        }

        // Copy slice from updates to output
        let upd_base = u * slice_size;
        out[flat_offset..flat_offset + slice_size]
            .copy_from_slice(&upd_data[upd_base..upd_base + slice_size]);
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(data_shape.to_vec()), None).map_err(KernelError::from)
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

    fn make_i64(data: Vec<i64>, shape: Vec<usize>) -> Tensor<'static, Int64Type> {
        let buffer = ScalarBuffer::<i64>::from(data).into_inner();
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_scatter_nd_1d() {
        // data: [1,2,3,4,5], indices: [[1],[3]], updates: [10, 30]
        let data = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let indices = make_i64(vec![1, 3], vec![2, 1]);
        let updates = make_f32(vec![10.0, 30.0], vec![2]);
        let out = scatter_nd::<Float32Type>(&data, &indices, &updates).unwrap();
        let result = out.data().typed_data::<f32>();
        assert_eq!(result, &[1.0, 10.0, 3.0, 30.0, 5.0]);
    }

    #[test]
    fn test_scatter_nd_2d_rows() {
        // data: [[1,2],[3,4],[5,6]], scatter at row 0 and row 2
        let data = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let indices = make_i64(vec![0, 2], vec![2, 1]);
        let updates = make_f32(vec![10.0, 20.0, 50.0, 60.0], vec![2, 2]);
        let out = scatter_nd::<Float32Type>(&data, &indices, &updates).unwrap();
        let result = out.data().typed_data::<f32>();
        assert_eq!(result, &[10.0, 20.0, 3.0, 4.0, 50.0, 60.0]);
    }

    #[test]
    fn test_scatter_nd_2d_elements() {
        // data: [[1,2],[3,4]], indices: [[0,1],[1,0]], updates: [20, 30]
        let data = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let indices = make_i64(vec![0, 1, 1, 0], vec![2, 2]);
        let updates = make_f32(vec![20.0, 30.0], vec![2]);
        let out = scatter_nd::<Float32Type>(&data, &indices, &updates).unwrap();
        let result = out.data().typed_data::<f32>();
        assert_eq!(result, &[1.0, 20.0, 30.0, 4.0]);
    }

    #[test]
    fn test_scatter_nd_out_of_range() {
        let data = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let indices = make_i64(vec![5], vec![1, 1]);
        let updates = make_f32(vec![99.0], vec![1]);
        assert!(scatter_nd::<Float32Type>(&data, &indices, &updates).is_err());
    }
}

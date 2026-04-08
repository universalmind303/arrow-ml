use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::datatypes::Int64Type;
use arrow::tensor::Tensor;
use arrow_ml_common::{KernelError, Result};

/// One-hot encoding: produce a tensor with an extra dimension of size `depth`
/// where position `indices[i]` is set to `on_value` and all others to `off_value`.
///
/// The one-hot dimension is inserted at `axis` (supports negative indexing).
/// Default axis is -1 (append at the end).
pub fn onehot<T>(
    indices: &Tensor<'_, Int64Type>,
    depth: usize,
    on_value: T::Native,
    off_value: T::Native,
    axis: i64,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let idx_shape = indices
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("onehot: indices has no shape".into()))?;
    let idx_ndim = idx_shape.len();
    let out_ndim = idx_ndim + 1;

    let axis = if axis < 0 {
        out_ndim as i64 + axis
    } else {
        axis
    };
    if axis < 0 || axis >= out_ndim as i64 {
        return Err(KernelError::InvalidArgument(format!(
            "onehot: axis {} out of range for output rank {}",
            axis, out_ndim
        )));
    }
    let axis = axis as usize;

    // Build output shape: insert `depth` at `axis`
    let mut out_shape = Vec::with_capacity(out_ndim);
    for (i, &d) in idx_shape.iter().enumerate() {
        if i == axis {
            out_shape.push(depth);
        }
        out_shape.push(d);
    }
    if axis == idx_ndim {
        out_shape.push(depth);
    }

    let total: usize = out_shape.iter().product();
    let idx_data: &[i64] = indices.data().typed_data();
    let idx_total: usize = idx_shape.iter().product();

    // Fill with off_value
    let mut out = vec![off_value; total];

    // Compute output strides
    let mut out_strides = vec![1usize; out_ndim];
    for d in (0..out_ndim.saturating_sub(1)).rev() {
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
    }

    // For each index element, compute its coordinates, insert the one-hot dim
    let mut idx_coords = vec![0usize; idx_ndim];
    for &raw_idx in idx_data.iter().take(idx_total) {
        let mut hot_idx = raw_idx;
        if hot_idx < 0 {
            hot_idx += depth as i64;
        }
        if hot_idx >= 0 && hot_idx < depth as i64 {
            // Build output coordinate: insert hot_idx at axis position
            let mut out_flat = 0;
            let mut src_d = 0;
            for (d, &stride) in out_strides.iter().enumerate().take(out_ndim) {
                if d == axis {
                    out_flat += hot_idx as usize * stride;
                } else {
                    out_flat += idx_coords[src_d] * stride;
                    src_d += 1;
                }
            }
            out[out_flat] = on_value;
        }

        // Increment idx_coords
        for d in (0..idx_ndim).rev() {
            idx_coords[d] += 1;
            if idx_coords[d] < idx_shape[d] {
                break;
            }
            idx_coords[d] = 0;
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

    fn make_i64(data: Vec<i64>, shape: Vec<usize>) -> Tensor<'static, Int64Type> {
        let buffer = ScalarBuffer::<i64>::from(data).into_inner();
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_onehot_1d_default_axis() {
        // indices: [0, 1, 2], depth=3, axis=-1
        // output shape: [3, 3] (axis=-1 appends)
        let indices = make_i64(vec![0, 1, 2], vec![3]);
        let out = onehot::<Float32Type>(&indices, 3, 1.0, 0.0, -1).unwrap();
        assert_eq!(out.shape().unwrap(), &[3, 3]);
        let data = out.data().typed_data::<f32>();
        assert_eq!(
            data,
            &[
                1.0, 0.0, 0.0, // index 0
                0.0, 1.0, 0.0, // index 1
                0.0, 0.0, 1.0, // index 2
            ]
        );
    }

    #[test]
    fn test_onehot_1d_axis0() {
        // indices: [0, 1, 2], depth=3, axis=0
        // output shape: [3, 3] — but transposed vs axis=-1
        let indices = make_i64(vec![0, 1, 2], vec![3]);
        let out = onehot::<Float32Type>(&indices, 3, 1.0, 0.0, 0).unwrap();
        assert_eq!(out.shape().unwrap(), &[3, 3]);
        let data = out.data().typed_data::<f32>();
        assert_eq!(
            data,
            &[
                1.0, 0.0, 0.0, // depth=0: only index 0 is hot
                0.0, 1.0, 0.0, // depth=1: only index 1 is hot
                0.0, 0.0, 1.0, // depth=2: only index 2 is hot
            ]
        );
    }

    #[test]
    fn test_onehot_custom_values() {
        let indices = make_i64(vec![1, 0], vec![2]);
        let out = onehot::<Float32Type>(&indices, 3, 5.0, -1.0, -1).unwrap();
        assert_eq!(out.shape().unwrap(), &[2, 3]);
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[-1.0, 5.0, -1.0, 5.0, -1.0, -1.0]);
    }

    #[test]
    fn test_onehot_2d_indices() {
        // indices shape [2,2], depth=3, axis=-1 -> output [2,2,3]
        let indices = make_i64(vec![0, 2, 1, 0], vec![2, 2]);
        let out = onehot::<Float32Type>(&indices, 3, 1.0, 0.0, -1).unwrap();
        assert_eq!(out.shape().unwrap(), &[2, 2, 3]);
        let data = out.data().typed_data::<f32>();
        assert_eq!(&data[0..3], &[1.0, 0.0, 0.0]); // [0,0] = index 0
        assert_eq!(&data[3..6], &[0.0, 0.0, 1.0]); // [0,1] = index 2
        assert_eq!(&data[6..9], &[0.0, 1.0, 0.0]); // [1,0] = index 1
        assert_eq!(&data[9..12], &[1.0, 0.0, 0.0]); // [1,1] = index 0
    }

    #[test]
    fn test_onehot_out_of_range_ignored() {
        // Out-of-range indices should leave all values as off_value
        let indices = make_i64(vec![0, 5], vec![2]);
        let out = onehot::<Float32Type>(&indices, 3, 1.0, 0.0, -1).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(&data[0..3], &[1.0, 0.0, 0.0]);
        assert_eq!(&data[3..6], &[0.0, 0.0, 0.0]); // index 5 out of range
    }
}

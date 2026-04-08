use arrow::array::{ArrowPrimitiveType, Int64Array};
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_ml_common::{KernelError, Result};

/// Gather slices along an axis using indices.
///
/// Output shape: input shape with `dim[axis]` replaced by `indices.len()`.
pub fn gather<T>(
    input: &Tensor<'_, T>,
    indices: &Int64Array,
    axis: usize,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("gather: tensor has no shape".into()))?;
    let ndim = shape.len();

    if axis >= ndim {
        return Err(KernelError::InvalidArgument(format!(
            "gather: axis {axis} out of range for {ndim}D tensor"
        )));
    }

    let axis_dim = shape[axis] as i64;
    let outer_size: usize = shape[..axis].iter().product();
    let inner_size: usize = shape[axis + 1..].iter().product();
    let data: &[T::Native] = input.data().typed_data();

    let num_indices = indices.len();
    let mut out_shape = shape.to_vec();
    out_shape[axis] = num_indices;

    let total = outer_size * num_indices * inner_size;
    let mut out = Vec::with_capacity(total);

    for o in 0..outer_size {
        for i in 0..num_indices {
            let mut idx = indices.value(i);
            if idx < 0 {
                idx += axis_dim;
            }
            if idx < 0 || idx >= axis_dim {
                return Err(KernelError::InvalidArgument(format!(
                    "gather: index {} out of range for axis dim {}",
                    indices.value(i),
                    axis_dim
                )));
            }
            let src = o * (shape[axis] * inner_size) + (idx as usize) * inner_size;
            out.extend_from_slice(&data[src..src + inner_size]);
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(out_shape), None).map_err(KernelError::from)
}

/// Slice a tensor along specified axes with start/end/step.
///
/// - Negative indices wrap around.
/// - `i64::MAX` for end means "to the end".
/// - If `axes` is None, defaults to 0..N for the first N starts.
/// - If `steps` is None, defaults to all 1s.
pub fn slice<T>(
    input: &Tensor<'_, T>,
    starts: &[i64],
    ends: &[i64],
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("slice: tensor has no shape".into()))?;
    let ndim = shape.len();

    if starts.len() != ends.len() {
        return Err(KernelError::InvalidArgument(
            "slice: starts and ends must have same length".into(),
        ));
    }

    let default_axes: Vec<i64> = (0..starts.len() as i64).collect();
    let ax = axes.unwrap_or(&default_axes);
    let default_steps: Vec<i64> = vec![1; starts.len()];
    let st = steps.unwrap_or(&default_steps);

    if ax.len() != starts.len() || st.len() != starts.len() {
        return Err(KernelError::InvalidArgument(
            "slice: axes and steps must match starts/ends length".into(),
        ));
    }

    // Resolve per-axis (start, end, step) for all dims
    let mut dim_ranges: Vec<(i64, i64, i64)> = (0..ndim).map(|d| (0, shape[d] as i64, 1)).collect();

    for i in 0..ax.len() {
        let a = {
            let v = ax[i];
            if v < 0 {
                (ndim as i64 + v) as usize
            } else {
                v as usize
            }
        };
        if a >= ndim {
            return Err(KernelError::InvalidArgument(format!(
                "slice: axis {} out of range for {ndim}D tensor",
                ax[i]
            )));
        }

        let step = st[i];
        if step == 0 {
            return Err(KernelError::InvalidArgument(
                "slice: step cannot be 0".into(),
            ));
        }

        let dim = shape[a] as i64;

        let mut s = starts[i];
        let mut e = ends[i];

        // Clamp to valid range
        if step > 0 {
            if s < 0 {
                s += dim;
            }
            if e < 0 {
                e += dim;
            }
            s = s.clamp(0, dim);
            e = e.clamp(0, dim);
        } else {
            if s < 0 {
                s += dim;
            }
            if e < 0 {
                e += dim;
            }
            s = s.clamp(-1, dim - 1);
            e = e.clamp(-1, dim - 1);
        }

        dim_ranges[a] = (s, e, step);
    }

    // Compute output shape
    let mut out_shape = Vec::with_capacity(ndim);
    #[allow(clippy::needless_range_loop)]
    for d in 0..ndim {
        let (s, e, step) = dim_ranges[d];
        let count = if step > 0 {
            ((e - s + step - 1) / step).max(0) as usize
        } else {
            ((s - e - step - 1) / (-step)).max(0) as usize
        };
        out_shape.push(count);
    }

    let total: usize = out_shape.iter().product();
    if total == 0 {
        let buf = Buffer::from_vec(Vec::<T::Native>::new());
        return Tensor::new_row_major(buf, Some(out_shape), None).map_err(KernelError::from);
    }

    let data: &[T::Native] = input.data().typed_data();

    // Compute input strides
    let mut in_strides = vec![1usize; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        in_strides[d] = in_strides[d + 1] * shape[d + 1];
    }

    let mut out = Vec::with_capacity(total);
    let mut coords = vec![0usize; ndim];

    for _ in 0..total {
        // Map output coords to input indices
        let mut flat = 0;
        for d in 0..ndim {
            let (s, _, step) = dim_ranges[d];
            let in_idx = (s + step * coords[d] as i64) as usize;
            flat += in_idx * in_strides[d];
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
        let buffer = ScalarBuffer::<f32>::from(data).into_inner();
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_gather_axis0() {
        // 3x4 matrix, gather rows [2, 0]
        let input = make_f32((0..12).map(|i| i as f32).collect(), vec![3, 4]);
        let indices = Int64Array::from(vec![2, 0]);
        let out = gather(&input, &indices, 0).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 4]);
        let data = out.data().typed_data::<f32>();
        assert_eq!(&data[0..4], &[8.0, 9.0, 10.0, 11.0]); // row 2
        assert_eq!(&data[4..8], &[0.0, 1.0, 2.0, 3.0]); // row 0
    }

    #[test]
    fn test_gather_axis1() {
        // 3x4 matrix, gather cols [1, 3]
        let input = make_f32((0..12).map(|i| i as f32).collect(), vec![3, 4]);
        let indices = Int64Array::from(vec![1, 3]);
        let out = gather(&input, &indices, 1).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![3, 2]);
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[1.0, 3.0, 5.0, 7.0, 9.0, 11.0]);
    }

    #[test]
    fn test_gather_negative_index() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let indices = Int64Array::from(vec![-1]); // last row
        let out = gather(&input, &indices, 0).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[5.0, 6.0]);
    }

    #[test]
    fn test_gather_out_of_range() {
        let input = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let indices = Int64Array::from(vec![5]);
        assert!(gather(&input, &indices, 0).is_err());
    }

    #[test]
    fn test_slice_basic() {
        // 3x4 matrix, slice rows [1:3], cols [0:2]
        let input = make_f32((0..12).map(|i| i as f32).collect(), vec![3, 4]);
        let out = slice(&input, &[1, 0], &[3, 2], Some(&[0, 1]), None).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 2]);
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[4.0, 5.0, 8.0, 9.0]);
    }

    #[test]
    fn test_slice_negative_indices() {
        // [0,1,2,3,4] -> slice [-3:MAX] = [2,3,4]
        let input = make_f32(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![5]);
        let out = slice(&input, &[-3], &[i64::MAX], None, None).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![3]);
        assert_eq!(out.data().typed_data::<f32>(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_slice_with_step() {
        // [0,1,2,3,4,5] with step 2 -> [0,2,4]
        let input = make_f32(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![6]);
        let out = slice(&input, &[0], &[6], None, Some(&[2])).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![3]);
        assert_eq!(out.data().typed_data::<f32>(), &[0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_slice_step_zero_error() {
        let input = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(slice(&input, &[0], &[3], None, Some(&[0])).is_err());
    }

    #[test]
    fn test_slice_default_axes() {
        let input = make_f32((0..12).map(|i| i as f32).collect(), vec![3, 4]);
        // axes=None, starts=[1,1], ends=[3,3] => axes=[0,1]
        let out = slice(&input, &[1, 1], &[3, 3], None, None).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 2]);
        assert_eq!(out.data().typed_data::<f32>(), &[5.0, 6.0, 9.0, 10.0]);
    }
}

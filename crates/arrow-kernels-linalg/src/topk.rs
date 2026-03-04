use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::datatypes::Int64Type;
use arrow::tensor::Tensor;
use arrow_kernels_common::{KernelError, Result};

/// Returns the top-K largest or smallest values and their indices along an axis.
///
/// - `largest`: if true, returns the K largest; if false, returns the K smallest.
/// - `sorted`: if true, results are sorted (descending for largest, ascending for smallest).
pub fn topk<T>(
    input: &Tensor<'_, T>,
    k: usize,
    axis: i64,
    largest: bool,
    sorted: bool,
) -> Result<(Tensor<'static, T>, Tensor<'static, Int64Type>)>
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd + Copy,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("topk: tensor has no shape".into()))?;
    let ndim = shape.len();
    if ndim == 0 {
        return Err(KernelError::InvalidArgument(
            "topk: tensor must be at least 1D".into(),
        ));
    }

    let axis = if axis < 0 { ndim as i64 + axis } else { axis };
    if axis < 0 || axis >= ndim as i64 {
        return Err(KernelError::InvalidArgument(format!(
            "topk: axis {} out of range for {}D tensor",
            axis, ndim
        )));
    }
    let axis = axis as usize;
    let dim_size = shape[axis];

    if k == 0 || k > dim_size {
        return Err(KernelError::InvalidArgument(format!(
            "topk: k ({k}) must be in [1, {dim_size}]"
        )));
    }

    let outer_size: usize = shape[..axis].iter().product();
    let inner_size: usize = shape[axis + 1..].iter().product();
    let outer_size = if outer_size == 0 { 1 } else { outer_size };
    let inner_size = if inner_size == 0 { 1 } else { inner_size };

    let data: &[T::Native] = input.data().typed_data();

    let mut out_values: Vec<T::Native> = Vec::with_capacity(outer_size * k * inner_size);
    let mut out_indices: Vec<i64> = Vec::with_capacity(outer_size * k * inner_size);

    for o in 0..outer_size {
        for i in 0..inner_size {
            // Extract elements along axis
            let mut pairs: Vec<(usize, T::Native)> = (0..dim_size)
                .map(|d| {
                    let idx = o * dim_size * inner_size + d * inner_size + i;
                    (d, data[idx])
                })
                .collect();

            // Sort: for largest, sort descending by value; for smallest, sort ascending
            if largest {
                pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            } else {
                pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }

            // Take top k
            let top: Vec<(usize, T::Native)> = pairs[..k].to_vec();

            // If not sorted, restore original order
            let top = if !sorted {
                let mut t = top;
                t.sort_by_key(|&(idx, _)| idx);
                t
            } else {
                top
            };

            for &(idx, val) in &top {
                out_values.push(val);
                out_indices.push(idx as i64);
            }
        }
    }

    let mut out_shape = shape.to_vec();
    out_shape[axis] = k;

    let val_buf = Buffer::from_vec(out_values);
    let idx_buf = Buffer::from_vec(out_indices);

    let val_tensor =
        Tensor::new_row_major(val_buf, Some(out_shape.clone()), None).map_err(KernelError::from)?;
    let idx_tensor =
        Tensor::new_row_major(idx_buf, Some(out_shape), None).map_err(KernelError::from)?;

    Ok((val_tensor, idx_tensor))
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
    fn test_topk_1d_largest() {
        let input = make_f32(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], vec![6]);
        let (vals, indices) = topk::<Float32Type>(&input, 3, 0, true, true).unwrap();
        assert_eq!(vals.shape().unwrap(), &vec![3]);
        let v = vals.data().typed_data::<f32>();
        assert_eq!(v, &[9.0, 5.0, 4.0]);
        let i = indices.data().typed_data::<i64>();
        assert_eq!(i, &[5, 4, 2]);
    }

    #[test]
    fn test_topk_1d_smallest() {
        let input = make_f32(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], vec![6]);
        let (vals, _) = topk::<Float32Type>(&input, 2, 0, false, true).unwrap();
        let v = vals.data().typed_data::<f32>();
        assert_eq!(v, &[1.0, 1.0]);
    }

    #[test]
    fn test_topk_2d_axis1() {
        // [[3,1,4],[5,9,2]] -> top 2 along axis 1
        let input = make_f32(vec![3.0, 1.0, 4.0, 5.0, 9.0, 2.0], vec![2, 3]);
        let (vals, indices) = topk::<Float32Type>(&input, 2, 1, true, true).unwrap();
        assert_eq!(vals.shape().unwrap(), &vec![2, 2]);
        let v = vals.data().typed_data::<f32>();
        assert_eq!(v, &[4.0, 3.0, 9.0, 5.0]); // row 0: [4,3], row 1: [9,5]
        let i = indices.data().typed_data::<i64>();
        assert_eq!(i, &[2, 0, 1, 0]); // row 0: [2,0], row 1: [1,0]
    }

    #[test]
    fn test_topk_unsorted() {
        let input = make_f32(vec![3.0, 1.0, 4.0, 5.0], vec![4]);
        let (_, indices) = topk::<Float32Type>(&input, 2, 0, true, false).unwrap();
        let i = indices.data().typed_data::<i64>();
        // Top 2 largest are at indices 2 and 3 (4.0 and 5.0), unsorted = original order
        assert_eq!(i, &[2, 3]);
    }

    #[test]
    fn test_topk_k_out_of_range() {
        let input = make_f32(vec![1.0, 2.0], vec![2]);
        assert!(topk::<Float32Type>(&input, 5, 0, true, true).is_err());
    }
}

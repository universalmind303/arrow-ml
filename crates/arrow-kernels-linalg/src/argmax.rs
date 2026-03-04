use arrow::array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow::buffer::Buffer;
use arrow::datatypes::Int64Type;
use arrow::tensor::Tensor;
use arrow_kernels_common::KernelError;
use arrow_kernels_common::Result;

/// Returns the index of the maximum element in the array.
///
/// Useful for greedy token decoding (argmax over logits).
/// The array must be non-empty and contain no null values.
pub fn argmax<T>(array: &PrimitiveArray<T>) -> Result<usize>
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd,
{
    if array.is_empty() {
        return Err(KernelError::EmptyArray {
            operation: "argmax",
        });
    }
    if array.null_count() > 0 {
        return Err(KernelError::NullsNotSupported {
            operation: "argmax",
        });
    }

    let vals = array.values();
    let mut max_idx = 0;
    for i in 1..vals.len() {
        if vals[i] > vals[max_idx] {
            max_idx = i;
        }
    }
    Ok(max_idx)
}

/// Returns the index of the minimum element in the array.
///
/// The array must be non-empty and contain no null values.
pub fn argmin<T>(array: &PrimitiveArray<T>) -> Result<usize>
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd,
{
    if array.is_empty() {
        return Err(KernelError::EmptyArray {
            operation: "argmin",
        });
    }
    if array.null_count() > 0 {
        return Err(KernelError::NullsNotSupported {
            operation: "argmin",
        });
    }

    let vals = array.values();
    let mut min_idx = 0;
    for i in 1..vals.len() {
        if vals[i] < vals[min_idx] {
            min_idx = i;
        }
    }
    Ok(min_idx)
}

// ---------------------------------------------------------------------------
// N-D Tensor variants with axis support
// ---------------------------------------------------------------------------

/// Returns the indices of the maximum values along the given axis of an N-D tensor.
///
/// The output is an `Int64Type` tensor whose shape is the input shape with the
/// reduction axis removed (or set to 1 when `keepdims` is true).
/// Negative axis values are resolved relative to the number of dimensions.
pub fn argmax_tensor<T>(
    input: &Tensor<'_, T>,
    axis: i64,
    keepdims: bool,
) -> Result<Tensor<'static, Int64Type>>
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd + Copy,
{
    argreduce_tensor(input, axis, keepdims, "argmax_tensor", |a, b| a > b)
}

/// Returns the indices of the minimum values along the given axis of an N-D tensor.
///
/// The output is an `Int64Type` tensor whose shape is the input shape with the
/// reduction axis removed (or set to 1 when `keepdims` is true).
/// Negative axis values are resolved relative to the number of dimensions.
pub fn argmin_tensor<T>(
    input: &Tensor<'_, T>,
    axis: i64,
    keepdims: bool,
) -> Result<Tensor<'static, Int64Type>>
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd + Copy,
{
    argreduce_tensor(input, axis, keepdims, "argmin_tensor", |a, b| a < b)
}

fn argreduce_tensor<T, F>(
    input: &Tensor<'_, T>,
    axis: i64,
    keepdims: bool,
    op: &str,
    is_better: F,
) -> Result<Tensor<'static, Int64Type>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
    F: Fn(T::Native, T::Native) -> bool,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument(format!("{op}: tensor has no shape")))?;
    let ndim = shape.len();

    // Resolve negative axis
    let axis_resolved = if axis < 0 {
        ndim as i64 + axis
    } else {
        axis
    };
    if axis_resolved < 0 || axis_resolved >= ndim as i64 {
        return Err(KernelError::InvalidArgument(format!(
            "{op}: axis {axis} out of range for {ndim}D tensor"
        )));
    }
    let axis_usize = axis_resolved as usize;

    let outer_size: usize = shape[..axis_usize].iter().product();
    let dim_size = shape[axis_usize];
    let inner_size: usize = shape[axis_usize + 1..].iter().product();

    if dim_size == 0 {
        return Err(KernelError::EmptyArray { operation: "argreduce_tensor" });
    }

    let data: &[T::Native] = input.data().typed_data();
    let mut result = Vec::with_capacity(outer_size * inner_size);

    for o in 0..outer_size {
        for i in 0..inner_size {
            let mut best_idx = 0i64;
            let mut best_val = data[o * dim_size * inner_size + i];
            for d in 1..dim_size {
                let idx = o * dim_size * inner_size + d * inner_size + i;
                if is_better(data[idx], best_val) {
                    best_val = data[idx];
                    best_idx = d as i64;
                }
            }
            result.push(best_idx);
        }
    }

    let out_shape = if keepdims {
        let mut s = shape.to_vec();
        s[axis_usize] = 1;
        s
    } else {
        shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis_usize)
            .map(|(_, &d)| d)
            .collect()
    };
    let out_shape = if out_shape.is_empty() {
        vec![1]
    } else {
        out_shape
    };

    let buf = Buffer::from_vec(result);
    Tensor::new_row_major(buf, Some(out_shape), None).map_err(KernelError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float32Array, Int32Array};
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::Float32Type;

    #[test]
    fn test_argmax_f32() {
        let arr = Float32Array::from(vec![1.0f32, 5.0, 3.0, 2.0]);
        assert_eq!(argmax(&arr).unwrap(), 1);
    }

    #[test]
    fn test_argmax_first_occurrence() {
        let arr = Float32Array::from(vec![5.0f32, 5.0, 5.0]);
        assert_eq!(argmax(&arr).unwrap(), 0);
    }

    #[test]
    fn test_argmax_i32() {
        let arr = Int32Array::from(vec![10, 30, 20]);
        assert_eq!(argmax(&arr).unwrap(), 1);
    }

    #[test]
    fn test_argmin_f32() {
        let arr = Float32Array::from(vec![3.0f32, 1.0, 2.0]);
        assert_eq!(argmin(&arr).unwrap(), 1);
    }

    #[test]
    fn test_argmax_empty() {
        let arr = Float32Array::from(Vec::<f32>::new());
        assert!(argmax(&arr).is_err());
    }

    #[test]
    fn test_argmax_single() {
        let arr = Float32Array::from(vec![42.0f32]);
        assert_eq!(argmax(&arr).unwrap(), 0);
    }

    // --- Tensor tests ---

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_argmax_tensor_2d_axis1() {
        // [[1,5,3],[4,2,6]] -> argmax over axis 1 -> [1, 2]
        let input = make_f32(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]);
        let out = argmax_tensor::<Float32Type>(&input, 1, false).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2]);
        let data = out.data().typed_data::<i64>();
        assert_eq!(data, &[1, 2]);
    }

    #[test]
    fn test_argmax_tensor_2d_axis0() {
        // [[1,5,3],[4,2,6]] -> argmax over axis 0 -> [1, 0, 1]
        let input = make_f32(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]);
        let out = argmax_tensor::<Float32Type>(&input, 0, false).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![3]);
        let data = out.data().typed_data::<i64>();
        assert_eq!(data, &[1, 0, 1]);
    }

    #[test]
    fn test_argmax_tensor_keepdims() {
        let input = make_f32(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]);
        let out = argmax_tensor::<Float32Type>(&input, 1, true).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 1]);
        let data = out.data().typed_data::<i64>();
        assert_eq!(data, &[1, 2]);
    }

    #[test]
    fn test_argmin_tensor_2d() {
        let input = make_f32(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]);
        let out = argmin_tensor::<Float32Type>(&input, 1, false).unwrap();
        let data = out.data().typed_data::<i64>();
        assert_eq!(data, &[0, 1]);
    }

    #[test]
    fn test_argmax_tensor_negative_axis() {
        let input = make_f32(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]);
        let out = argmax_tensor::<Float32Type>(&input, -1, false).unwrap();
        let data = out.data().typed_data::<i64>();
        assert_eq!(data, &[1, 2]); // same as axis=1
    }
}

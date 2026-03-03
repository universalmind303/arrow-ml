use arrow::array::{Array, ArrowPrimitiveType, PrimitiveArray};
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float32Array, Int32Array};

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
}

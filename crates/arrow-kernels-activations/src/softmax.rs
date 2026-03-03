use arrow::array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow_kernels_common::KernelError;
use arrow_kernels_common::Result;
use num_traits::{Float, Zero};
use std::ops::AddAssign;

/// Softmax over all elements: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max)).
///
/// Uses the max-subtraction trick for numerical stability.
/// Returns an error if the array contains null values or is empty.
pub fn softmax<T>(array: &PrimitiveArray<T>) -> Result<PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float + AddAssign,
{
    if array.null_count() > 0 {
        return Err(KernelError::NullsNotSupported {
            operation: "softmax",
        });
    }
    if array.is_empty() {
        return Err(KernelError::EmptyArray {
            operation: "softmax",
        });
    }

    let values = array.values();

    // Find max for numerical stability
    let max_val = values
        .iter()
        .copied()
        .fold(T::Native::neg_infinity(), |a, b| a.max(b));

    // Compute exp(x - max) and running sum
    let mut sum = T::Native::zero();
    let exp_vals: Vec<T::Native> = values
        .iter()
        .map(|&x| {
            let e = (x - max_val).exp();
            sum += e;
            e
        })
        .collect();

    // Normalize
    let result: Vec<T::Native> = exp_vals.into_iter().map(|e| e / sum).collect();
    Ok(PrimitiveArray::from_iter_values(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float32Array;

    #[test]
    fn test_softmax_uniform() {
        // All equal values -> uniform distribution
        let input = Float32Array::from(vec![1.0_f32, 1.0, 1.0, 1.0]);
        let output = softmax(&input).unwrap();
        for i in 0..4 {
            assert!((output.value(i) - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let input = Float32Array::from(vec![1.0_f32, 2.0, 3.0, 4.0]);
        let output = softmax(&input).unwrap();
        let sum: f32 = output.values().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_ordering() {
        // Larger inputs should have larger softmax values
        let input = Float32Array::from(vec![1.0_f32, 3.0, 2.0]);
        let output = softmax(&input).unwrap();
        assert!(output.value(1) > output.value(2));
        assert!(output.value(2) > output.value(0));
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow without max subtraction
        let input = Float32Array::from(vec![1000.0_f32, 1001.0, 1002.0]);
        let output = softmax(&input).unwrap();
        let sum: f32 = output.values().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Should still preserve ordering
        assert!(output.value(2) > output.value(1));
        assert!(output.value(1) > output.value(0));
    }

    #[test]
    fn test_softmax_rejects_nulls() {
        let input = Float32Array::from(vec![Some(1.0_f32), None, Some(3.0)]);
        assert!(softmax(&input).is_err());
    }

    #[test]
    fn test_softmax_rejects_empty() {
        let input = Float32Array::from(Vec::<f32>::new());
        assert!(softmax(&input).is_err());
    }
}

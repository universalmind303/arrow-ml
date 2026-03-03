use arrow::array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow_kernels_common::KernelError;
use arrow_kernels_common::Result;
use num_traits::{Float, Zero};
use std::ops::AddAssign;

/// Computes the dot product of two 1D arrays.
///
/// Both arrays must have the same length and contain no null values.
pub fn dot<T>(a: &PrimitiveArray<T>, b: &PrimitiveArray<T>) -> Result<T::Native>
where
    T: ArrowPrimitiveType,
    T::Native: Float + AddAssign,
{
    if a.len() != b.len() {
        return Err(KernelError::ShapeMismatch {
            operation: "dot",
            expected: format!("length {}", a.len()),
            actual: format!("length {}", b.len()),
        });
    }
    if a.null_count() > 0 || b.null_count() > 0 {
        return Err(KernelError::NullsNotSupported { operation: "dot" });
    }
    if a.is_empty() {
        return Err(KernelError::EmptyArray { operation: "dot" });
    }

    let a_vals = a.values();
    let b_vals = b.values();
    let mut sum = T::Native::zero();
    for i in 0..a_vals.len() {
        sum += a_vals[i] * b_vals[i];
    }
    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float32Array, Float64Array};

    #[test]
    fn test_dot_f32() {
        let a = Float32Array::from(vec![1.0_f32, 2.0, 3.0]);
        let b = Float32Array::from(vec![4.0_f32, 5.0, 6.0]);
        let result = dot(&a, &b).unwrap();
        assert!((result - 32.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_dot_f64() {
        let a = Float64Array::from(vec![1.0_f64, 2.0, 3.0]);
        let b = Float64Array::from(vec![4.0_f64, 5.0, 6.0]);
        let result = dot(&a, &b).unwrap();
        assert!((result - 32.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_dot_length_mismatch() {
        let a = Float32Array::from(vec![1.0_f32, 2.0]);
        let b = Float32Array::from(vec![1.0_f32, 2.0, 3.0]);
        assert!(dot(&a, &b).is_err());
    }

    #[test]
    fn test_dot_empty() {
        let a = Float32Array::from(Vec::<f32>::new());
        let b = Float32Array::from(Vec::<f32>::new());
        assert!(dot(&a, &b).is_err());
    }
}

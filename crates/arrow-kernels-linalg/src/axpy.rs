use arrow::array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow_kernels_common::KernelError;
use arrow_kernels_common::Result;
use num_traits::Zero;
use std::ops::{Add, Mul};

/// BLAS Level 1: y = α * x + y.
///
/// Computes the scaled vector addition. Both arrays must have the same length
/// and contain no null values.
///
/// Returns a new array containing the result.
pub fn axpy<T>(alpha: T::Native, x: &PrimitiveArray<T>, y: &PrimitiveArray<T>) -> Result<PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    T::Native: Zero + Copy + Mul<Output = T::Native> + Add<Output = T::Native>,
{
    if x.len() != y.len() {
        return Err(KernelError::ShapeMismatch {
            operation: "axpy",
            expected: format!("length {}", x.len()),
            actual: format!("length {}", y.len()),
        });
    }
    if x.null_count() > 0 || y.null_count() > 0 {
        return Err(KernelError::NullsNotSupported { operation: "axpy" });
    }

    let x_vals = x.values();
    let y_vals = y.values();
    let result: Vec<T::Native> = x_vals
        .iter()
        .zip(y_vals.iter())
        .map(|(&xi, &yi)| alpha * xi + yi)
        .collect();

    Ok(PrimitiveArray::from_iter_values(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float32Array, Float64Array, Int32Array};

    #[test]
    fn test_axpy_f32() {
        let x = Float32Array::from(vec![1.0f32, 2.0, 3.0]);
        let y = Float32Array::from(vec![10.0f32, 20.0, 30.0]);
        let result = axpy(2.0f32, &x, &y).unwrap();
        let vals = result.values();
        // 2*[1,2,3] + [10,20,30] = [12, 24, 36]
        assert!((vals[0] - 12.0).abs() < 1e-6);
        assert!((vals[1] - 24.0).abs() < 1e-6);
        assert!((vals[2] - 36.0).abs() < 1e-6);
    }

    #[test]
    fn test_axpy_f64() {
        let x = Float64Array::from(vec![1.0, 2.0, 3.0]);
        let y = Float64Array::from(vec![10.0, 20.0, 30.0]);
        let result = axpy(0.5, &x, &y).unwrap();
        let vals = result.values();
        // 0.5*[1,2,3] + [10,20,30] = [10.5, 21.0, 31.5]
        assert!((vals[0] - 10.5).abs() < 1e-10);
        assert!((vals[1] - 21.0).abs() < 1e-10);
        assert!((vals[2] - 31.5).abs() < 1e-10);
    }

    #[test]
    fn test_axpy_i32() {
        let x = Int32Array::from(vec![1, 2, 3]);
        let y = Int32Array::from(vec![10, 20, 30]);
        let result = axpy(3, &x, &y).unwrap();
        let vals = result.values();
        assert_eq!(vals.as_ref(), &[13, 26, 39]);
    }

    #[test]
    fn test_axpy_length_mismatch() {
        let x = Float32Array::from(vec![1.0f32, 2.0]);
        let y = Float32Array::from(vec![1.0f32, 2.0, 3.0]);
        assert!(axpy(1.0, &x, &y).is_err());
    }
}

use arrow::array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow_ml_common::KernelError;
use arrow_ml_common::Result;
use std::ops::Mul;

/// BLAS Level 1: x = α * x.
///
/// Scales every element of the array by the scalar `alpha`.
/// The array must contain no null values.
///
/// Returns a new array containing the result.
pub fn scal<T>(alpha: T::Native, x: &PrimitiveArray<T>) -> Result<PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy + Mul<Output = T::Native>,
{
    if x.null_count() > 0 {
        return Err(KernelError::NullsNotSupported { operation: "scal" });
    }

    let result: Vec<T::Native> = x.values().iter().map(|&xi| alpha * xi).collect();
    Ok(PrimitiveArray::from_iter_values(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float32Array, Float64Array, Int32Array};

    #[test]
    fn test_scal_f32() {
        let x = Float32Array::from(vec![1.0f32, 2.0, 3.0]);
        let result = scal(2.5f32, &x).unwrap();
        let vals = result.values();
        assert!((vals[0] - 2.5).abs() < 1e-6);
        assert!((vals[1] - 5.0).abs() < 1e-6);
        assert!((vals[2] - 7.5).abs() < 1e-6);
    }

    #[test]
    fn test_scal_f64() {
        let x = Float64Array::from(vec![10.0, 20.0, 30.0]);
        let result = scal(0.1, &x).unwrap();
        let vals = result.values();
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 2.0).abs() < 1e-10);
        assert!((vals[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_scal_i32() {
        let x = Int32Array::from(vec![3, 6, 9]);
        let result = scal(4, &x).unwrap();
        let vals = result.values();
        assert_eq!(vals.as_ref(), &[12, 24, 36]);
    }

    #[test]
    fn test_scal_zero() {
        let x = Float32Array::from(vec![1.0f32, 2.0, 3.0]);
        let result = scal(0.0f32, &x).unwrap();
        let vals = result.values();
        assert_eq!(vals.as_ref(), &[0.0, 0.0, 0.0]);
    }
}

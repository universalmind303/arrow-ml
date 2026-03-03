use arrow::array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow_kernels_common::KernelError;
use arrow_kernels_common::Result;
use num_traits::{Float, Zero};
use std::ops::AddAssign;

/// L2 (Euclidean) norm: sqrt(sum(x_i^2)).
pub fn l2_norm<T>(array: &PrimitiveArray<T>) -> Result<T::Native>
where
    T: ArrowPrimitiveType,
    T::Native: Float + AddAssign,
{
    if array.null_count() > 0 {
        return Err(KernelError::NullsNotSupported {
            operation: "l2_norm",
        });
    }
    if array.is_empty() {
        return Err(KernelError::EmptyArray {
            operation: "l2_norm",
        });
    }
    let values = array.values();
    let mut sum = T::Native::zero();
    for &v in values.iter() {
        sum += v * v;
    }
    Ok(sum.sqrt())
}

/// L1 norm: sum(|x_i|).
pub fn l1_norm<T>(array: &PrimitiveArray<T>) -> Result<T::Native>
where
    T: ArrowPrimitiveType,
    T::Native: Float + AddAssign,
{
    if array.null_count() > 0 {
        return Err(KernelError::NullsNotSupported {
            operation: "l1_norm",
        });
    }
    if array.is_empty() {
        return Err(KernelError::EmptyArray {
            operation: "l1_norm",
        });
    }
    let values = array.values();
    let mut sum = T::Native::zero();
    for &v in values.iter() {
        sum += v.abs();
    }
    Ok(sum)
}

/// L-infinity norm: max(|x_i|).
pub fn linf_norm<T>(array: &PrimitiveArray<T>) -> Result<T::Native>
where
    T: ArrowPrimitiveType,
    T::Native: Float + AddAssign,
{
    if array.null_count() > 0 {
        return Err(KernelError::NullsNotSupported {
            operation: "linf_norm",
        });
    }
    if array.is_empty() {
        return Err(KernelError::EmptyArray {
            operation: "linf_norm",
        });
    }
    let values = array.values();
    let mut max = T::Native::zero();
    for &v in values.iter() {
        let abs = v.abs();
        if abs > max {
            max = abs;
        }
    }
    Ok(max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float32Array;

    #[test]
    fn test_l2_norm() {
        let a = Float32Array::from(vec![3.0_f32, 4.0]);
        let result = l2_norm(&a).unwrap();
        assert!((result - 5.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_l1_norm() {
        let a = Float32Array::from(vec![1.0_f32, -2.0, 3.0]);
        let result = l1_norm(&a).unwrap();
        assert!((result - 6.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_linf_norm() {
        let a = Float32Array::from(vec![1.0_f32, -5.0, 3.0]);
        let result = linf_norm(&a).unwrap();
        assert!((result - 5.0_f32).abs() < 1e-6);
    }
}

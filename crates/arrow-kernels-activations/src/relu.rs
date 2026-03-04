use arrow::array::{ArrowPrimitiveType, PrimitiveArray};
use num_traits::{Float, Zero};

/// ReLU activation: max(0, x).
///
/// Null values are propagated (output is null where input is null).
pub fn relu<T>(array: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    array.unary(|x| {
        if x > T::Native::zero() {
            x
        } else {
            T::Native::zero()
        }
    })
}

/// Leaky ReLU activation: x if x > 0, else alpha * x.
///
/// Null values are propagated.
pub fn leaky_relu<T>(array: &PrimitiveArray<T>, alpha: T::Native) -> PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    array.unary(|x| if x > T::Native::zero() { x } else { alpha * x })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float32Array, Float64Array};

    #[test]
    fn test_relu_f32() {
        let input = Float32Array::from(vec![1.0_f32, -2.0, 3.0, -4.0, 0.0]);
        let output = relu(&input);
        assert_eq!(output.values().as_ref(), &[1.0, 0.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn test_relu_f64() {
        let input = Float64Array::from(vec![-1.0_f64, 2.0, -3.0]);
        let output = relu(&input);
        assert_eq!(output.values().as_ref(), &[0.0, 2.0, 0.0]);
    }

    #[test]
    fn test_relu_with_nulls() {
        let input = Float32Array::from(vec![Some(1.0_f32), None, Some(-1.0)]);
        let output = relu(&input);
        assert_eq!(output.value(0), 1.0);
        assert!(output.is_null(1));
        assert_eq!(output.value(2), 0.0);
    }

    #[test]
    fn test_leaky_relu() {
        let input = Float32Array::from(vec![1.0_f32, -2.0, 3.0, -4.0]);
        let output = leaky_relu(&input, 0.1);
        let expected: &[f32] = &[1.0, -0.2, 3.0, -0.4];
        for (a, b) in output.values().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}

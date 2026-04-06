use arrow::array::{ArrowPrimitiveType, PrimitiveArray};
use num_traits::{Float, One};

/// Sigmoid activation: 1 / (1 + exp(-x)).
///
/// Null values are propagated.
pub fn sigmoid<T>(array: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let one = T::Native::one();
    array.unary(|x| one / (one + (-x).exp()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float32Array;

    #[test]
    fn test_sigmoid_zero() {
        let input = Float32Array::from(vec![0.0_f32]);
        let output = sigmoid(&input);
        assert!((output.value(0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_large_positive() {
        let input = Float32Array::from(vec![100.0_f32]);
        let output = sigmoid(&input);
        assert!((output.value(0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sigmoid_large_negative() {
        let input = Float32Array::from(vec![-100.0_f32]);
        let output = sigmoid(&input);
        assert!(output.value(0).abs() < 1e-5);
    }

    #[test]
    fn test_sigmoid_values() {
        let input = Float32Array::from(vec![0.0_f32, 1.0, -1.0, 2.0]);
        let output = sigmoid(&input);
        let expected = [0.5, 0.7310586, 0.26894143, 0.880797];
        for (a, b) in output.values().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
    }
}

use arrow::array::{ArrowPrimitiveType, PrimitiveArray};
use num_traits::Float;

/// Tanh activation: tanh(x).
///
/// Null values are propagated.
pub fn tanh<T>(array: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    array.unary(|x| x.tanh())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float32Array;

    #[test]
    fn test_tanh_zero() {
        let input = Float32Array::from(vec![0.0_f32]);
        let output = tanh(&input);
        assert!(output.value(0).abs() < 1e-6);
    }

    #[test]
    fn test_tanh_values() {
        let input = Float32Array::from(vec![0.0_f32, 1.0, -1.0, 5.0]);
        let output = tanh(&input);
        let expected = [0.0, 0.7615942, -0.7615942, 0.9999092];
        for (a, b) in output.values().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
    }

    #[test]
    fn test_tanh_bounds() {
        // tanh output is always in (-1, 1)
        let input = Float32Array::from(vec![100.0_f32, -100.0]);
        let output = tanh(&input);
        assert!((output.value(0) - 1.0).abs() < 1e-5);
        assert!((output.value(1) + 1.0).abs() < 1e-5);
    }
}

use arrow::array::{ArrowPrimitiveType, PrimitiveArray};
use num_traits::{Float, One};

/// SiLU (Swish) activation: x * sigmoid(x) = x / (1 + exp(-x)).
///
/// Used in LLaMA, Mistral, and other modern LLMs.
pub fn silu<T>(array: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let one = T::Native::one();
    array.unary(|x| x / (one + (-x).exp()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float32Array;

    #[test]
    fn test_silu_f32() {
        let input = Float32Array::from(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0]);
        let output = silu(&input);
        let vals = output.values();
        // SiLU(-2) ≈ -0.2384, SiLU(-1) ≈ -0.2689, SiLU(0) = 0,
        // SiLU(1) ≈ 0.7311, SiLU(2) ≈ 1.7616
        assert!((vals[0] - (-0.2384)).abs() < 0.01);
        assert!((vals[1] - (-0.2689)).abs() < 0.01);
        assert!((vals[2] - 0.0).abs() < 1e-6);
        assert!((vals[3] - 0.7311).abs() < 0.01);
        assert!((vals[4] - 1.7616).abs() < 0.01);
    }

    #[test]
    fn test_silu_large_positive() {
        // For large x, silu(x) ≈ x
        let input = Float32Array::from(vec![10.0f32]);
        let output = silu(&input);
        assert!((output.value(0) - 10.0).abs() < 0.001);
    }
}

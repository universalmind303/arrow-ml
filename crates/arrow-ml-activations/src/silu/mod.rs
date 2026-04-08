mod kernel_f32;
mod kernel_f64;

use arrow::array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::DataType;
use num_traits::{Float, One};

/// Minimum number of elements to use SIMD path.
const SIMD_THRESHOLD: usize = 1024;

/// SiLU (Swish) activation: x * sigmoid(x) = x / (1 + exp(-x)).
///
/// Used in LLaMA, Mistral, and other modern LLMs.
pub fn silu<T>(array: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    if array.len() >= SIMD_THRESHOLD && array.null_count() == 0 {
        if let Some(result) = try_simd_silu(array) {
            return result;
        }
    }
    let one = T::Native::one();
    array.unary(|x| x / (one + (-x).exp()))
}

fn try_simd_silu<T>(array: &PrimitiveArray<T>) -> Option<PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let values = array.values();
    let len = values.len();

    match T::DATA_TYPE {
        DataType::Float32 => {
            let input =
                unsafe { std::slice::from_raw_parts(values.as_ptr() as *const f32, len) };
            let result = kernel_f32::silu(input);
            let buffer = Buffer::from_vec(result);
            let scalar_buffer = ScalarBuffer::<f32>::new(buffer, 0, len);
            Some(PrimitiveArray::new(
                unsafe { std::mem::transmute(scalar_buffer) },
                None,
            ))
        }
        DataType::Float64 => {
            let input =
                unsafe { std::slice::from_raw_parts(values.as_ptr() as *const f64, len) };
            let result = kernel_f64::silu(input);
            let buffer = Buffer::from_vec(result);
            let scalar_buffer = ScalarBuffer::<f64>::new(buffer, 0, len);
            Some(PrimitiveArray::new(
                unsafe { std::mem::transmute(scalar_buffer) },
                None,
            ))
        }
        _ => None,
    }
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

    #[test]
    fn test_silu_simd_path_f32() {
        let data: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) * 0.01).collect();
        let input = Float32Array::from(data.clone());
        let output = silu(&input);
        for (i, &x) in data.iter().enumerate() {
            let expected = x / (1.0 + (-x).exp());
            let actual = output.value(i);
            assert!(
                (actual - expected).abs() < 1e-2,
                "mismatch at index {i}: got {actual}, expected {expected}"
            );
        }
    }
}

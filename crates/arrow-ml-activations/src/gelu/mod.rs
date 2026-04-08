mod kernel_f32;
mod kernel_f64;

use arrow::array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::DataType;
use num_traits::{Float, One};

/// Minimum number of elements to use SIMD path.
const SIMD_THRESHOLD: usize = 1024;

/// GeLU activation (tanh approximation).
///
/// gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
///
/// Used in GPT, BERT, and most modern transformers.
pub fn gelu<T>(array: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    if array.len() >= SIMD_THRESHOLD && array.null_count() == 0 {
        if let Some(result) = try_simd_gelu(array) {
            return result;
        }
    }

    let half = <T::Native as num_traits::NumCast>::from(0.5).unwrap();
    let one = T::Native::one();
    let sqrt_2_over_pi = <T::Native as num_traits::NumCast>::from(0.7978845608).unwrap();
    let coeff = <T::Native as num_traits::NumCast>::from(0.044715).unwrap();

    array.unary(|x| {
        let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        half * x * (one + inner.tanh())
    })
}

/// GeLU activation (exact): x * 0.5 * (1 + erf(x / sqrt(2))).
///
/// Uses an erf approximation. Slightly more accurate than the tanh version.
pub fn gelu_exact<T>(array: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let sqrt2 = <T::Native as num_traits::NumCast>::from(std::f64::consts::SQRT_2).unwrap();
    let half = <T::Native as num_traits::NumCast>::from(0.5).unwrap();
    let one = T::Native::one();

    array.unary(|x| half * x * (one + erf(x / sqrt2)))
}

/// Approximate erf using Abramowitz & Stegun formula 7.1.26.
fn erf<F: Float>(x: F) -> F {
    if x >= F::zero() {
        erf_positive(x)
    } else {
        -erf_positive(-x)
    }
}

fn erf_positive<F: Float>(x: F) -> F {
    let one = F::one();
    let p = <F as num_traits::NumCast>::from(0.3275911).unwrap();
    let a1 = <F as num_traits::NumCast>::from(0.254829592).unwrap();
    let a2 = <F as num_traits::NumCast>::from(-0.284496736).unwrap();
    let a3 = <F as num_traits::NumCast>::from(1.421413741).unwrap();
    let a4 = <F as num_traits::NumCast>::from(-1.453152027).unwrap();
    let a5 = <F as num_traits::NumCast>::from(1.061405429).unwrap();

    let t = one / (one + p * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    one - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * (-x * x).exp()
}

fn try_simd_gelu<T>(array: &PrimitiveArray<T>) -> Option<PrimitiveArray<T>>
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
            let result = kernel_f32::gelu(input);
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
            let result = kernel_f64::gelu(input);
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
    fn test_gelu_f32() {
        let input = Float32Array::from(vec![-1.0f32, 0.0, 1.0, 2.0]);
        let output = gelu(&input);
        let vals = output.values();
        assert!((vals[0] - (-0.1588)).abs() < 0.01);
        assert!((vals[1] - 0.0).abs() < 1e-6);
        assert!((vals[2] - 0.8412).abs() < 0.01);
        assert!((vals[3] - 1.9545).abs() < 0.01);
    }

    #[test]
    fn test_gelu_exact_f32() {
        let input = Float32Array::from(vec![-1.0f32, 0.0, 1.0, 2.0]);
        let output = gelu_exact(&input);
        let vals = output.values();
        assert!((vals[0] - (-0.1587)).abs() < 0.01);
        assert!((vals[1] - 0.0).abs() < 1e-6);
        assert!((vals[2] - 0.8413).abs() < 0.01);
        assert!((vals[3] - 1.9545).abs() < 0.01);
    }

    #[test]
    fn test_gelu_simd_path_f32() {
        let data: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) * 0.005).collect();
        let input = Float32Array::from(data.clone());
        let output = gelu(&input);
        for (i, &x) in data.iter().enumerate() {
            let inner = 0.7978845608f32 * (x + 0.044715f32 * x * x * x);
            let expected = 0.5 * x * (1.0 + inner.tanh());
            let actual = output.value(i);
            assert!(
                (actual - expected).abs() < 1e-2,
                "mismatch at index {i}: got {actual}, expected {expected}"
            );
        }
    }
}

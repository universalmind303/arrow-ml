mod kernel_f32;
mod kernel_f64;

use arrow::array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::DataType;
use num_traits::{Float, One};

/// Minimum number of elements to use SIMD path.
const SIMD_THRESHOLD: usize = 1024;

/// Sigmoid activation: 1 / (1 + exp(-x)).
///
/// Null values are propagated.
pub fn sigmoid<T>(array: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    if array.len() >= SIMD_THRESHOLD && array.null_count() == 0 {
        if let Some(result) = try_simd_sigmoid(array) {
            return result;
        }
    }
    let one = T::Native::one();
    array.unary(|x| one / (one + (-x).exp()))
}

fn try_simd_sigmoid<T>(array: &PrimitiveArray<T>) -> Option<PrimitiveArray<T>>
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
            let result = kernel_f32::sigmoid(input);
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
            let result = kernel_f64::sigmoid(input);
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

    #[test]
    fn test_sigmoid_simd_path_f32() {
        let data: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) * 0.01).collect();
        let input = Float32Array::from(data.clone());
        let output = sigmoid(&input);
        for (i, &x) in data.iter().enumerate() {
            let expected = 1.0 / (1.0 + (-x).exp());
            let actual = output.value(i);
            assert!(
                (actual - expected).abs() < 1e-3,
                "mismatch at index {i}: got {actual}, expected {expected}"
            );
        }
    }
}

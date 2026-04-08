mod kernel_f32;
mod kernel_f64;

use arrow::array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::DataType;
use num_traits::Float;

/// Minimum number of elements to use SIMD path.
const SIMD_THRESHOLD: usize = 1024;

/// Tanh activation: tanh(x).
///
/// Null values are propagated.
pub fn tanh<T>(array: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    if array.len() >= SIMD_THRESHOLD && array.null_count() == 0 {
        if let Some(result) = try_simd_tanh(array) {
            return result;
        }
    }
    array.unary(|x| x.tanh())
}

fn try_simd_tanh<T>(array: &PrimitiveArray<T>) -> Option<PrimitiveArray<T>>
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
            let result = kernel_f32::tanh(input);
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
            let result = kernel_f64::tanh(input);
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

    #[test]
    fn test_tanh_simd_path_f32() {
        let data: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) * 0.005).collect();
        let input = Float32Array::from(data.clone());
        let output = tanh(&input);
        for (i, &x) in data.iter().enumerate() {
            let expected = x.tanh();
            let actual = output.value(i);
            assert!(
                (actual - expected).abs() < 1e-3,
                "mismatch at index {i}: got {actual}, expected {expected}"
            );
        }
    }
}

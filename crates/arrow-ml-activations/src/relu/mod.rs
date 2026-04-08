mod kernel_f32;
mod kernel_f64;

use arrow::array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::DataType;
use num_traits::{Float, Zero};

/// Minimum number of elements to use SIMD path.
const SIMD_THRESHOLD: usize = 1024;

/// ReLU activation: max(0, x).
///
/// Null values are propagated (output is null where input is null).
pub fn relu<T>(array: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    if array.len() >= SIMD_THRESHOLD && array.null_count() == 0 {
        if let Some(result) = try_simd_relu(array) {
            return result;
        }
    }
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
    if array.len() >= SIMD_THRESHOLD && array.null_count() == 0 {
        if let Some(result) = try_simd_leaky_relu(array, alpha) {
            return result;
        }
    }
    array.unary(|x| if x > T::Native::zero() { x } else { alpha * x })
}

fn try_simd_relu<T>(array: &PrimitiveArray<T>) -> Option<PrimitiveArray<T>>
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
            let result = kernel_f32::relu(input);
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
            let result = kernel_f64::relu(input);
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

fn try_simd_leaky_relu<T>(array: &PrimitiveArray<T>, alpha: T::Native) -> Option<PrimitiveArray<T>>
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
            let alpha_f32: f32 = unsafe { *(&alpha as *const T::Native as *const f32) };
            let result = kernel_f32::leaky_relu(input, alpha_f32);
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
            let alpha_f64: f64 = unsafe { *(&alpha as *const T::Native as *const f64) };
            let result = kernel_f64::leaky_relu(input, alpha_f64);
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

    #[test]
    fn test_relu_simd_path_f32() {
        let data: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) * 0.01).collect();
        let input = Float32Array::from(data.clone());
        let output = relu(&input);
        for (i, &x) in data.iter().enumerate() {
            let expected = if x > 0.0 { x } else { 0.0 };
            assert_eq!(output.value(i), expected, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_relu_simd_path_f64() {
        let data: Vec<f64> = (0..2048).map(|i| (i as f64 - 1024.0) * 0.01).collect();
        let input = Float64Array::from(data.clone());
        let output = relu(&input);
        for (i, &x) in data.iter().enumerate() {
            let expected = if x > 0.0 { x } else { 0.0 };
            assert_eq!(output.value(i), expected, "mismatch at index {i}");
        }
    }
}

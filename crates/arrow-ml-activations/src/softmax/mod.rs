mod kernel_f32;
mod kernel_f64;

use arrow::array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::DataType;
use arrow::tensor::Tensor;
use arrow_ml_common::KernelError;
use arrow_ml_common::Result;
use num_traits::{Float, Zero};
use std::ops::AddAssign;

/// Minimum number of elements to use SIMD path.
const SIMD_THRESHOLD: usize = 1024;

/// Softmax over all elements: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max)).
///
/// Uses the max-subtraction trick for numerical stability.
/// Returns an error if the array contains null values or is empty.
pub fn softmax<T>(array: &PrimitiveArray<T>) -> Result<PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float + AddAssign,
{
    if array.null_count() > 0 {
        return Err(KernelError::NullsNotSupported {
            operation: "softmax",
        });
    }
    if array.is_empty() {
        return Err(KernelError::EmptyArray {
            operation: "softmax",
        });
    }

    if array.len() >= SIMD_THRESHOLD {
        if let Some(result) = try_simd_softmax(array) {
            return Ok(result);
        }
    }

    let values = array.values();

    // Find max for numerical stability
    let max_val = values
        .iter()
        .copied()
        .fold(T::Native::neg_infinity(), |a, b| a.max(b));

    // Compute exp(x - max) and running sum
    let mut sum = T::Native::zero();
    let exp_vals: Vec<T::Native> = values
        .iter()
        .map(|&x| {
            let e = (x - max_val).exp();
            sum += e;
            e
        })
        .collect();

    // Normalize
    let result: Vec<T::Native> = exp_vals.into_iter().map(|e| e / sum).collect();
    Ok(PrimitiveArray::from_iter_values(result))
}

fn try_simd_softmax<T>(array: &PrimitiveArray<T>) -> Option<PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float + AddAssign,
{
    let values = array.values();
    let len = values.len();

    match T::DATA_TYPE {
        DataType::Float32 => {
            let input =
                unsafe { std::slice::from_raw_parts(values.as_ptr() as *const f32, len) };
            let result = kernel_f32::softmax(input);
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
            let result = kernel_f64::softmax(input);
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

/// Softmax over a specific axis of an N-D tensor.
///
/// Uses the outer/reduce/inner decomposition: for each (outer, inner) pair,
/// extracts the `dim_size` elements along the given axis, computes softmax
/// over them using the max-subtraction trick, and writes them back.
///
/// Supports negative axis values (e.g., -1 means last axis).
pub fn softmax_tensor<T>(input: &Tensor<'_, T>, axis: i64) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float + AddAssign,
{
    let shape = input.shape().ok_or_else(|| {
        KernelError::InvalidArgument("softmax_tensor: tensor has no shape".into())
    })?;
    let ndim = shape.len();
    if ndim == 0 {
        return Err(KernelError::InvalidArgument(
            "softmax_tensor: tensor must be at least 1D".into(),
        ));
    }

    let axis = if axis < 0 { ndim as i64 + axis } else { axis };
    if axis < 0 || axis >= ndim as i64 {
        return Err(KernelError::InvalidArgument(format!(
            "softmax_tensor: axis {} out of range for {}D tensor",
            axis, ndim
        )));
    }
    let axis = axis as usize;

    let outer_size: usize = shape[..axis].iter().product();
    let dim_size = shape[axis];
    let inner_size: usize = shape[axis + 1..].iter().product();
    let outer_size = if outer_size == 0 { 1 } else { outer_size };
    let inner_size = if inner_size == 0 { 1 } else { inner_size };

    let data: &[T::Native] = input.data().typed_data();
    let mut out = data.to_vec();

    for o in 0..outer_size {
        for i in 0..inner_size {
            // Find max
            let mut max_val = T::Native::neg_infinity();
            for d in 0..dim_size {
                let idx = o * dim_size * inner_size + d * inner_size + i;
                if data[idx] > max_val {
                    max_val = data[idx];
                }
            }
            // Exp and sum
            let mut sum = T::Native::zero();
            for d in 0..dim_size {
                let idx = o * dim_size * inner_size + d * inner_size + i;
                let e = (data[idx] - max_val).exp();
                out[idx] = e;
                sum += e;
            }
            // Normalize
            for d in 0..dim_size {
                let idx = o * dim_size * inner_size + d * inner_size + i;
                out[idx] = out[idx] / sum;
            }
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(shape.to_vec()), None).map_err(KernelError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float32Array;
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::Float32Type;

    #[test]
    fn test_softmax_uniform() {
        // All equal values -> uniform distribution
        let input = Float32Array::from(vec![1.0_f32, 1.0, 1.0, 1.0]);
        let output = softmax(&input).unwrap();
        for i in 0..4 {
            assert!((output.value(i) - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let input = Float32Array::from(vec![1.0_f32, 2.0, 3.0, 4.0]);
        let output = softmax(&input).unwrap();
        let sum: f32 = output.values().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_ordering() {
        // Larger inputs should have larger softmax values
        let input = Float32Array::from(vec![1.0_f32, 3.0, 2.0]);
        let output = softmax(&input).unwrap();
        assert!(output.value(1) > output.value(2));
        assert!(output.value(2) > output.value(0));
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow without max subtraction
        let input = Float32Array::from(vec![1000.0_f32, 1001.0, 1002.0]);
        let output = softmax(&input).unwrap();
        let sum: f32 = output.values().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Should still preserve ordering
        assert!(output.value(2) > output.value(1));
        assert!(output.value(1) > output.value(0));
    }

    #[test]
    fn test_softmax_rejects_nulls() {
        let input = Float32Array::from(vec![Some(1.0_f32), None, Some(3.0)]);
        assert!(softmax(&input).is_err());
    }

    #[test]
    fn test_softmax_rejects_empty() {
        let input = Float32Array::from(Vec::<f32>::new());
        assert!(softmax(&input).is_err());
    }

    #[test]
    fn test_softmax_simd_path_f32() {
        let data: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) * 0.01).collect();
        let input = Float32Array::from(data);
        let output = softmax(&input).unwrap();
        let sum: f32 = output.values().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "softmax sum should be ~1.0, got {sum}"
        );
        // Check ordering: larger inputs have larger softmax
        assert!(output.value(2047) > output.value(0));
    }

    // --- Tensor softmax tests ---

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_softmax_tensor_2d_axis1() {
        // 2x3 tensor, softmax over axis 1 (rows)
        let input = make_f32(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3]);
        let out = softmax_tensor::<Float32Type>(&input, 1).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 3]);
        let data = out.data().typed_data::<f32>();
        // Each row should sum to 1
        let row0_sum: f32 = data[0..3].iter().sum();
        let row1_sum: f32 = data[3..6].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-6);
        assert!((row1_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_tensor_2d_axis0() {
        // 2x3 tensor, softmax over axis 0 (columns)
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = softmax_tensor::<Float32Type>(&input, 0).unwrap();
        let data = out.data().typed_data::<f32>();
        // Each column should sum to 1
        for j in 0..3 {
            let col_sum = data[j] + data[3 + j];
            assert!((col_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_tensor_3d_attention() {
        // Simulate attention: (batch=1, heads=2, seq_len=3), softmax over last axis
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
        let out = softmax_tensor::<Float32Type>(&input, -1).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 2, 3]);
        let data = out.data().typed_data::<f32>();
        let sum0: f32 = data[0..3].iter().sum();
        let sum1: f32 = data[3..6].iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-6);
        assert!((sum1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_tensor_negative_axis() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = softmax_tensor::<Float32Type>(&input, -1).unwrap(); // same as axis=1
        let data = out.data().typed_data::<f32>();
        let row0_sum: f32 = data[0..3].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-6);
    }
}

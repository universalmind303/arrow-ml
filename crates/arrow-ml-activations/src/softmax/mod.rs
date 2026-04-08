mod kernel_f16;
mod kernel_naive;
mod kernel_simd;

use arrow::array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::DataType;
use arrow::tensor::Tensor;
use arrow_ml_common::KernelError;
use arrow_ml_common::Result;
use half::f16;
use num_traits::{Float, Zero};
use std::ops::AddAssign;

/// Minimum array size to use SIMD path (below this, SIMD overhead dominates).
const SIMD_THRESHOLD: usize = 1024;

/// Softmax over all elements: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max)).
///
/// Uses the max-subtraction trick for numerical stability.
/// Empty arrays are returned as-is (no-op).
/// Null values are propagated (output is null where input is null).
///
/// Supports Float16, Float32, and Float64 with optimized kernels:
/// - Float16: native kernel (computed via f32)
/// - Float32/Float64: SIMD kernels for arrays >= SIMD_THRESHOLD elements
/// - Other float types: generic fallback implementation
pub fn softmax<T>(array: &PrimitiveArray<T>) -> Result<PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float + AddAssign,
{
    // Empty array is a no-op
    if array.is_empty() {
        return Ok(array.clone());
    }

    let values = array.values();
    let n = values.len();

    // If there are nulls, we need to compute softmax only on non-null values
    let result_buf = if let Some(nulls) = array.nulls() {
        // Collect non-null values
        let non_null_values: Vec<T::Native> = array.iter()
            .flatten()
            .collect();

        if non_null_values.is_empty() {
            // All nulls - return zeros
            return Ok(PrimitiveArray::new(
                Buffer::from_vec(vec![T::Native::zero(); n]).into(),
                array.nulls().cloned()
            ));
        }

        // Compute softmax on non-null values
        let non_null_count = non_null_values.len();
        let softmax_result = match T::DATA_TYPE {
            DataType::Float16 => {
                let buf = Buffer::from_vec(non_null_values);
                let input: &[f16] = buf.typed_data();
                let mut output = vec![f16::ZERO; non_null_count];
                kernel_f16::softmax_f16(input, &mut output);
                output.into_iter().map(|v| {
                    // Convert f16 back to T::Native
                    let f16_buf = Buffer::from_vec(vec![v]);
                    let as_native: &[T::Native] = f16_buf.typed_data();
                    as_native[0]
                }).collect::<Vec<_>>()
            }
            DataType::Float32 => {
                let buf = Buffer::from_vec(non_null_values);
                let input: &[f32] = buf.typed_data();
                let mut output = vec![0.0f32; non_null_count];

                if input.len() >= SIMD_THRESHOLD {
                    kernel_simd::softmax_f32(input, &mut output);
                } else {
                    kernel_naive::softmax_f32(input, &mut output);
                }

                output.into_iter().map(|v| {
                    let f32_buf = Buffer::from_vec(vec![v]);
                    let as_native: &[T::Native] = f32_buf.typed_data();
                    as_native[0]
                }).collect::<Vec<_>>()
            }
            DataType::Float64 => {
                let buf = Buffer::from_vec(non_null_values);
                let input: &[f64] = buf.typed_data();
                let mut output = vec![0.0f64; non_null_count];

                if input.len() >= SIMD_THRESHOLD {
                    kernel_simd::softmax_f64(input, &mut output);
                } else {
                    kernel_naive::softmax_f64(input, &mut output);
                }

                output.into_iter().map(|v| {
                    let f64_buf = Buffer::from_vec(vec![v]);
                    let as_native: &[T::Native] = f64_buf.typed_data();
                    as_native[0]
                }).collect::<Vec<_>>()
            }
            _ => {
                // Generic fallback
                let max_val = non_null_values
                    .iter()
                    .copied()
                    .fold(T::Native::neg_infinity(), |a, b| a.max(b));

                let mut sum = T::Native::zero();
                let exp_vals: Vec<T::Native> = non_null_values
                    .iter()
                    .map(|&x| {
                        let e = (x - max_val).exp();
                        sum += e;
                        e
                    })
                    .collect();

                exp_vals.into_iter().map(|e| e / sum).collect()
            }
        };

        // Map results back to original positions
        let mut result = vec![T::Native::zero(); n];
        let mut result_idx = 0;
        for i in 0..n {
            if nulls.is_valid(i) {
                result[i] = softmax_result[result_idx];
                result_idx += 1;
            }
        }

        Buffer::from_vec(result)
    } else {
        // No nulls - compute softmax on all values
        match T::DATA_TYPE {
            DataType::Float16 => {
                let buf = Buffer::from(values.inner().clone());
                let input: &[f16] = buf.typed_data();
                let mut output = vec![f16::ZERO; n];

                kernel_f16::softmax_f16(input, &mut output);

                Buffer::from_vec(output)
            }
            DataType::Float32 => {
                let buf = Buffer::from(values.inner().clone());
                let input: &[f32] = buf.typed_data();
                let mut output = vec![0.0f32; n];

                if n >= SIMD_THRESHOLD {
                    kernel_simd::softmax_f32(input, &mut output);
                } else {
                    kernel_naive::softmax_f32(input, &mut output);
                }

                Buffer::from_vec(output)
            }
            DataType::Float64 => {
                let buf = Buffer::from(values.inner().clone());
                let input: &[f64] = buf.typed_data();
                let mut output = vec![0.0f64; n];

                if n >= SIMD_THRESHOLD {
                    kernel_simd::softmax_f64(input, &mut output);
                } else {
                    kernel_naive::softmax_f64(input, &mut output);
                }

                Buffer::from_vec(output)
            }
            _ => {
                // Generic fallback for other float types
                let max_val = values
                    .iter()
                    .copied()
                    .fold(T::Native::neg_infinity(), |a, b| a.max(b));

                let mut sum = T::Native::zero();
                let exp_vals: Vec<T::Native> = values
                    .iter()
                    .map(|&x| {
                        let e = (x - max_val).exp();
                        sum += e;
                        e
                    })
                    .collect();

                let result: Vec<T::Native> = exp_vals.into_iter().map(|e| e / sum).collect();
                Buffer::from_vec(result)
            }
        }
    };

    // Preserve null mask from input array
    Ok(PrimitiveArray::new(result_buf.into(), array.nulls().cloned()))
}

/// Softmax over a specific axis of an N-D tensor.
///
/// Uses the outer/reduce/inner decomposition: for each (outer, inner) pair,
/// extracts the `dim_size` elements along the given axis, computes softmax
/// over them using the max-subtraction trick, and writes them back.
///
/// Supports negative axis values (e.g., -1 means last axis).
///
/// Supports Float16, Float32, and Float64 with optimized kernels:
/// - Float16: native kernel (computed via f32)
/// - Float32/Float64: SIMD kernels for axis dimension >= SIMD_THRESHOLD
/// - Other float types: generic fallback implementation
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

    // Dispatch based on type and dimension size
    match T::DATA_TYPE {
        DataType::Float16 => {
            // f16 path - work with ScalarBuffer for type safety
            let scalar_buf = ScalarBuffer::<f16>::from(
                input.data().typed_data::<f16>().to_vec()
            );
            let data_f16 = scalar_buf.as_ref();

            let mut out_vec: Vec<f16> = data_f16.to_vec();

            for o in 0..outer_size {
                for i in 0..inner_size {
                    // Extract slice along axis dimension
                    let mut slice_data = Vec::with_capacity(dim_size);
                    for d in 0..dim_size {
                        let idx = o * dim_size * inner_size + d * inner_size + i;
                        slice_data.push(data_f16[idx]);
                    }

                    // Compute softmax on this slice
                    let mut slice_out = vec![f16::ZERO; dim_size];
                    kernel_f16::softmax_f16(&slice_data, &mut slice_out);

                    // Write back
                    for d in 0..dim_size {
                        let idx = o * dim_size * inner_size + d * inner_size + i;
                        out_vec[idx] = slice_out[d];
                    }
                }
            }

            // Convert back to T::Native
            let out_scalar = ScalarBuffer::<f16>::from(out_vec);
            let out_buf = Buffer::from(out_scalar.into_inner());
            out = out_buf.typed_data::<T::Native>().to_vec();
        }
        DataType::Float32 if dim_size >= SIMD_THRESHOLD => {
            // SIMD path for f32 - work with ScalarBuffer
            let scalar_buf = ScalarBuffer::<f32>::from(
                input.data().typed_data::<f32>().to_vec()
            );
            let data_f32 = scalar_buf.as_ref();

            let mut out_vec: Vec<f32> = data_f32.to_vec();

            for o in 0..outer_size {
                for i in 0..inner_size {
                    // Extract slice along axis dimension
                    let mut slice_data = Vec::with_capacity(dim_size);
                    for d in 0..dim_size {
                        let idx = o * dim_size * inner_size + d * inner_size + i;
                        slice_data.push(data_f32[idx]);
                    }

                    // Compute softmax on this slice
                    let mut slice_out = vec![0.0f32; dim_size];
                    kernel_simd::softmax_f32(&slice_data, &mut slice_out);

                    // Write back
                    for d in 0..dim_size {
                        let idx = o * dim_size * inner_size + d * inner_size + i;
                        out_vec[idx] = slice_out[d];
                    }
                }
            }

            // Convert back to T::Native
            let out_scalar = ScalarBuffer::<f32>::from(out_vec);
            let out_buf = Buffer::from(out_scalar.into_inner());
            out = out_buf.typed_data::<T::Native>().to_vec();
        }
        DataType::Float64 if dim_size >= SIMD_THRESHOLD => {
            // SIMD path for f64 - work with ScalarBuffer
            let scalar_buf = ScalarBuffer::<f64>::from(
                input.data().typed_data::<f64>().to_vec()
            );
            let data_f64 = scalar_buf.as_ref();

            let mut out_vec: Vec<f64> = data_f64.to_vec();

            for o in 0..outer_size {
                for i in 0..inner_size {
                    // Extract slice along axis dimension
                    let mut slice_data = Vec::with_capacity(dim_size);
                    for d in 0..dim_size {
                        let idx = o * dim_size * inner_size + d * inner_size + i;
                        slice_data.push(data_f64[idx]);
                    }

                    // Compute softmax on this slice
                    let mut slice_out = vec![0.0f64; dim_size];
                    kernel_simd::softmax_f64(&slice_data, &mut slice_out);

                    // Write back
                    for d in 0..dim_size {
                        let idx = o * dim_size * inner_size + d * inner_size + i;
                        out_vec[idx] = slice_out[d];
                    }
                }
            }

            // Convert back to T::Native
            let out_scalar = ScalarBuffer::<f64>::from(out_vec);
            let out_buf = Buffer::from(out_scalar.into_inner());
            out = out_buf.typed_data::<T::Native>().to_vec();
        }
        _ => {
            // Naive path for all other cases
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
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(shape.to_vec()), None).map_err(KernelError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float16Array, Float32Array};
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::{Float16Type, Float32Type};

    #[test]
    fn test_softmax_f16() {
        // Test Float16 support
        let input_values = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let input = Float16Array::from(input_values);
        let output = softmax(&input).unwrap();

        // Check sum to 1
        let sum: f32 = output.values().iter().map(|&x| x.to_f32()).sum();
        assert!((sum - 1.0).abs() < 1e-3, "Sum is {}", sum);

        // Check ordering
        assert!(output.value(3) > output.value(2));
        assert!(output.value(2) > output.value(1));
        assert!(output.value(1) > output.value(0));
    }

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
    fn test_softmax_with_nulls() {
        // Null values should be propagated
        let input = Float32Array::from(vec![Some(1.0_f32), None, Some(3.0)]);
        let output = softmax(&input).unwrap();

        // Check that null is propagated
        assert!(!output.is_null(0));
        assert!(output.is_null(1));
        assert!(!output.is_null(2));

        // Check that non-null values sum to 1
        let sum: f32 = output.iter().flatten().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Sum is {}", sum);
    }

    #[test]
    fn test_softmax_empty() {
        // Empty array should be a no-op
        let input = Float32Array::from(Vec::<f32>::new());
        let output = softmax(&input).unwrap();
        assert_eq!(output.len(), 0);
    }

    #[test]
    fn test_softmax_simd_threshold() {
        // Test with size above SIMD threshold to trigger SIMD path
        let input: Vec<f32> = (0..2048).map(|i| (i as f32) * 0.01).collect();
        let array = Float32Array::from(input);
        let output = softmax(&array).unwrap();

        let sum: f32 = output.values().iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    // --- Tensor softmax tests ---

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    fn make_f16(data: Vec<f16>, shape: Vec<usize>) -> Tensor<'static, Float16Type> {
        let buffer = Buffer::from_vec(data);
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_softmax_tensor_f16() {
        // Test Float16 tensor support
        let input = make_f16(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
                f16::from_f32(5.0),
                f16::from_f32(6.0),
            ],
            vec![2, 3],
        );
        let out = softmax_tensor::<Float16Type>(&input, 1).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 3]);
        let data = out.data().typed_data::<f16>();
        // Each row should sum to 1
        let row0_sum: f32 = data[0..3].iter().map(|&x| x.to_f32()).sum();
        let row1_sum: f32 = data[3..6].iter().map(|&x| x.to_f32()).sum();
        assert!((row0_sum - 1.0).abs() < 1e-3);
        assert!((row1_sum - 1.0).abs() < 1e-3);
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

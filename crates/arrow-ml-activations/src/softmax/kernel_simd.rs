use std::simd::prelude::*;
use std::simd::StdFloat;

const SIMD_WIDTH_F32: usize = 8;
const SIMD_WIDTH_F64: usize = 4;

/// SIMD-optimized softmax for f32.
///
/// Uses 8-wide SIMD vectors for finding max, computing exp, and summing.
/// Falls back to scalar operations for remainder elements.
pub fn softmax_f32(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let n = input.len();

    // Find max using SIMD
    let mut max_vec = Simd::<f32, SIMD_WIDTH_F32>::splat(f32::NEG_INFINITY);
    let chunks = n / SIMD_WIDTH_F32;

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH_F32;
        // SAFETY: offset is guaranteed to be within bounds by chunks calculation
        let vals = unsafe {
            Simd::<f32, SIMD_WIDTH_F32>::from_slice(input.get_unchecked(offset..offset + SIMD_WIDTH_F32))
        };
        max_vec = max_vec.simd_max(vals);
    }

    // Reduce SIMD max to scalar
    let mut max_val = max_vec.reduce_max();

    // Handle remainder
    for i in (chunks * SIMD_WIDTH_F32)..n {
        if input[i] > max_val {
            max_val = input[i];
        }
    }

    let max_broadcast = Simd::<f32, SIMD_WIDTH_F32>::splat(max_val);

    // Compute exp(x - max) and sum using SIMD
    let mut sum_vec = Simd::<f32, SIMD_WIDTH_F32>::splat(0.0);

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH_F32;
        // SAFETY: offset is guaranteed to be within bounds
        unsafe {
            let vals = Simd::<f32, SIMD_WIDTH_F32>::from_slice(
                input.get_unchecked(offset..offset + SIMD_WIDTH_F32),
            );
            let exp_vals = (vals - max_broadcast).exp();
            exp_vals.copy_to_slice(output.get_unchecked_mut(offset..offset + SIMD_WIDTH_F32));
            sum_vec += exp_vals;
        }
    }

    // Reduce sum
    let mut sum = sum_vec.reduce_sum();

    // Handle remainder
    for i in (chunks * SIMD_WIDTH_F32)..n {
        let e = (input[i] - max_val).exp();
        output[i] = e;
        sum += e;
    }

    // Normalize using SIMD
    let inv_sum = 1.0 / sum;
    let inv_sum_vec = Simd::<f32, SIMD_WIDTH_F32>::splat(inv_sum);

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH_F32;
        // SAFETY: offset is guaranteed to be within bounds
        unsafe {
            let vals = Simd::<f32, SIMD_WIDTH_F32>::from_slice(
                output.get_unchecked(offset..offset + SIMD_WIDTH_F32),
            );
            let normalized = vals * inv_sum_vec;
            normalized.copy_to_slice(output.get_unchecked_mut(offset..offset + SIMD_WIDTH_F32));
        }
    }

    // Normalize remainder
    for i in (chunks * SIMD_WIDTH_F32)..n {
        output[i] *= inv_sum;
    }
}

/// SIMD-optimized softmax for f64.
///
/// Uses 4-wide SIMD vectors for finding max, computing exp, and summing.
/// Falls back to scalar operations for remainder elements.
pub fn softmax_f64(input: &[f64], output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());
    let n = input.len();

    // Find max using SIMD
    let mut max_vec = Simd::<f64, SIMD_WIDTH_F64>::splat(f64::NEG_INFINITY);
    let chunks = n / SIMD_WIDTH_F64;

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH_F64;
        // SAFETY: offset is guaranteed to be within bounds by chunks calculation
        let vals = unsafe {
            Simd::<f64, SIMD_WIDTH_F64>::from_slice(input.get_unchecked(offset..offset + SIMD_WIDTH_F64))
        };
        max_vec = max_vec.simd_max(vals);
    }

    // Reduce SIMD max to scalar
    let mut max_val = max_vec.reduce_max();

    // Handle remainder
    for i in (chunks * SIMD_WIDTH_F64)..n {
        if input[i] > max_val {
            max_val = input[i];
        }
    }

    let max_broadcast = Simd::<f64, SIMD_WIDTH_F64>::splat(max_val);

    // Compute exp(x - max) and sum using SIMD
    let mut sum_vec = Simd::<f64, SIMD_WIDTH_F64>::splat(0.0);

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH_F64;
        // SAFETY: offset is guaranteed to be within bounds
        unsafe {
            let vals = Simd::<f64, SIMD_WIDTH_F64>::from_slice(
                input.get_unchecked(offset..offset + SIMD_WIDTH_F64),
            );
            let exp_vals = (vals - max_broadcast).exp();
            exp_vals.copy_to_slice(output.get_unchecked_mut(offset..offset + SIMD_WIDTH_F64));
            sum_vec += exp_vals;
        }
    }

    // Reduce sum
    let mut sum = sum_vec.reduce_sum();

    // Handle remainder
    for i in (chunks * SIMD_WIDTH_F64)..n {
        let e = (input[i] - max_val).exp();
        output[i] = e;
        sum += e;
    }

    // Normalize using SIMD
    let inv_sum = 1.0 / sum;
    let inv_sum_vec = Simd::<f64, SIMD_WIDTH_F64>::splat(inv_sum);

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH_F64;
        // SAFETY: offset is guaranteed to be within bounds
        unsafe {
            let vals = Simd::<f64, SIMD_WIDTH_F64>::from_slice(
                output.get_unchecked(offset..offset + SIMD_WIDTH_F64),
            );
            let normalized = vals * inv_sum_vec;
            normalized.copy_to_slice(output.get_unchecked_mut(offset..offset + SIMD_WIDTH_F64));
        }
    }

    // Normalize remainder
    for i in (chunks * SIMD_WIDTH_F64)..n {
        output[i] *= inv_sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_f32_simd() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0f32; 8];
        softmax_f32(&input, &mut output);

        // Check sum to 1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check ordering
        for i in 0..7 {
            assert!(output[i + 1] > output[i]);
        }
    }

    #[test]
    fn test_softmax_f64_simd() {
        let input = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0f64; 8];
        softmax_f64(&input, &mut output);

        // Check sum to 1
        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Check ordering
        for i in 0..7 {
            assert!(output[i + 1] > output[i]);
        }
    }

    #[test]
    fn test_softmax_f32_non_aligned() {
        // Test with size not multiple of SIMD width
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 5];
        softmax_f32(&input, &mut output);

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_f32_large() {
        // Test numerical stability with large values
        let input = vec![1000.0f32, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0, 1007.0];
        let mut output = vec![0.0f32; 8];
        softmax_f32(&input, &mut output);

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}

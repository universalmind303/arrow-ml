use half::f16;

/// Naive softmax implementation for f16 (Float16).
///
/// Computes softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
/// using the max-subtraction trick for numerical stability.
///
/// Note: f16 operations are performed by converting to f32, computing,
/// and converting back to maintain numerical precision.
pub fn softmax_f16(input: &[f16], output: &mut [f16]) {
    debug_assert_eq!(input.len(), output.len());
    let n = input.len();

    // Convert to f32 for computation
    let input_f32: Vec<f32> = input.iter().map(|&x| x.to_f32()).collect();

    // Find max for numerical stability
    let mut max_val = f32::NEG_INFINITY;
    for &x in &input_f32 {
        if x > max_val {
            max_val = x;
        }
    }

    // Compute exp(x - max) and accumulate sum
    let mut sum = 0.0f32;
    let mut exp_vals = vec![0.0f32; n];
    for i in 0..n {
        let e = (input_f32[i] - max_val).exp();
        exp_vals[i] = e;
        sum += e;
    }

    // Normalize by sum and convert back to f16
    let inv_sum = 1.0 / sum;
    for i in 0..n {
        output[i] = f16::from_f32(exp_vals[i] * inv_sum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_f16() {
        let input = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let mut output = vec![f16::ZERO; 4];
        softmax_f16(&input, &mut output);

        // Check sum to 1
        let sum: f32 = output.iter().map(|&x| x.to_f32()).sum();
        assert!((sum - 1.0).abs() < 1e-3, "Sum is {}", sum);

        // Check ordering
        assert!(output[3] > output[2]);
        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
    }

    #[test]
    fn test_softmax_f16_uniform() {
        let input = vec![f16::from_f32(1.0); 4];
        let mut output = vec![f16::ZERO; 4];
        softmax_f16(&input, &mut output);

        // All should be 0.25
        for &val in &output {
            let val_f32 = val.to_f32();
            assert!((val_f32 - 0.25).abs() < 1e-3);
        }
    }
}

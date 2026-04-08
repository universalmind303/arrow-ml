/// Naive softmax implementation for f32.
///
/// Computes softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
/// using the max-subtraction trick for numerical stability.
pub fn softmax_f32(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let n = input.len();

    // Find max for numerical stability
    let mut max_val = f32::NEG_INFINITY;
    for &x in input {
        if x > max_val {
            max_val = x;
        }
    }

    // Compute exp(x - max) and accumulate sum
    let mut sum = 0.0f32;
    for i in 0..n {
        let e = (input[i] - max_val).exp();
        output[i] = e;
        sum += e;
    }

    // Normalize by sum
    let inv_sum = 1.0 / sum;
    for i in 0..n {
        output[i] *= inv_sum;
    }
}

/// Naive softmax implementation for f64.
///
/// Computes softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
/// using the max-subtraction trick for numerical stability.
pub fn softmax_f64(input: &[f64], output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());
    let n = input.len();

    // Find max for numerical stability
    let mut max_val = f64::NEG_INFINITY;
    for &x in input {
        if x > max_val {
            max_val = x;
        }
    }

    // Compute exp(x - max) and accumulate sum
    let mut sum = 0.0f64;
    for i in 0..n {
        let e = (input[i] - max_val).exp();
        output[i] = e;
        sum += e;
    }

    // Normalize by sum
    let inv_sum = 1.0 / sum;
    for i in 0..n {
        output[i] *= inv_sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_f32_naive() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        softmax_f32(&input, &mut output);

        // Check sum to 1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check ordering
        assert!(output[3] > output[2]);
        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
    }

    #[test]
    fn test_softmax_f64_naive() {
        let input = vec![1.0f64, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f64; 4];
        softmax_f64(&input, &mut output);

        // Check sum to 1
        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Check ordering
        assert!(output[3] > output[2]);
        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
    }

    #[test]
    fn test_softmax_f32_uniform() {
        let input = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 4];
        softmax_f32(&input, &mut output);

        // All should be 0.25
        for &val in &output {
            assert!((val - 0.25).abs() < 1e-6);
        }
    }
}

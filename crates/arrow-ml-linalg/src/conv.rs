use arrow::array::{ArrowPrimitiveType, PrimitiveArray};
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_ml_common::KernelError;
use arrow_ml_common::Result;
use num_traits::{Float, Zero};

/// Naive 2D convolution (direct loops, no SIMD).
///
/// Implements the ONNX Conv operator semantics for 2D spatial data.
///
/// # Arguments
///
/// * `input`    - 4D tensor in NCHW layout: (batch, in_channels, height, width)
/// * `weight`   - 4D tensor: (out_channels, in_channels/groups, kernel_h, kernel_w)
/// * `bias`     - Optional 1D bias array of length `out_channels`
/// * `padding`  - `[pad_h, pad_w]` — zero-padding applied to each spatial side
/// * `stride`   - `[stride_h, stride_w]`
/// * `dilation` - `[dilation_h, dilation_w]`
/// * `groups`   - Number of groups for grouped/depthwise convolution (default 1)
///
/// # Returns
///
/// 4D tensor in NCHW layout: (batch, out_channels, out_h, out_w)
pub fn conv2d<T>(
    input: &Tensor<'_, T>,
    weight: &Tensor<'_, T>,
    bias: Option<&PrimitiveArray<T>>,
    padding: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    // --- Extract and validate input shape (N, C, H, W) ---
    let in_shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("conv2d: input tensor has no shape".into()))?;
    if in_shape.len() != 4 {
        return Err(KernelError::InvalidArgument(format!(
            "conv2d: expected 4D input (NCHW), got {}D",
            in_shape.len()
        )));
    }
    let (batch, in_channels, in_h, in_w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);

    // --- Extract and validate weight shape (OC, IC/groups, kH, kW) ---
    let w_shape = weight
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("conv2d: weight tensor has no shape".into()))?;
    if w_shape.len() != 4 {
        return Err(KernelError::InvalidArgument(format!(
            "conv2d: expected 4D weight, got {}D",
            w_shape.len()
        )));
    }
    let (out_channels, ic_per_group, k_h, k_w) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);

    // --- Validate groups ---
    if groups == 0 {
        return Err(KernelError::InvalidArgument(
            "conv2d: groups must be >= 1".into(),
        ));
    }
    if in_channels % groups != 0 {
        return Err(KernelError::ShapeMismatch {
            operation: "conv2d",
            expected: format!("in_channels ({in_channels}) divisible by groups ({groups})"),
            actual: format!("in_channels % groups = {}", in_channels % groups),
        });
    }
    if out_channels % groups != 0 {
        return Err(KernelError::ShapeMismatch {
            operation: "conv2d",
            expected: format!("out_channels ({out_channels}) divisible by groups ({groups})"),
            actual: format!("out_channels % groups = {}", out_channels % groups),
        });
    }
    if in_channels / groups != ic_per_group {
        return Err(KernelError::ShapeMismatch {
            operation: "conv2d",
            expected: format!(
                "weight IC/group dim = {} (in_channels/groups)",
                in_channels / groups
            ),
            actual: format!("{ic_per_group}"),
        });
    }

    // --- Validate bias ---
    if let Some(b) = bias {
        if b.len() != out_channels {
            return Err(KernelError::ShapeMismatch {
                operation: "conv2d",
                expected: format!("bias length {out_channels}"),
                actual: format!("bias length {}", b.len()),
            });
        }
    }

    // --- Validate stride and dilation ---
    if stride[0] == 0 || stride[1] == 0 {
        return Err(KernelError::InvalidArgument(
            "conv2d: stride must be >= 1".into(),
        ));
    }
    if dilation[0] == 0 || dilation[1] == 0 {
        return Err(KernelError::InvalidArgument(
            "conv2d: dilation must be >= 1".into(),
        ));
    }

    // --- Compute output spatial dimensions ---
    let effective_k_h = (k_h - 1) * dilation[0] + 1;
    let effective_k_w = (k_w - 1) * dilation[1] + 1;

    // Guard against underflow: padded input must be at least as large as the effective kernel.
    let padded_h = in_h + 2 * padding[0];
    let padded_w = in_w + 2 * padding[1];
    if padded_h < effective_k_h || padded_w < effective_k_w {
        return Err(KernelError::InvalidArgument(format!(
            "conv2d: padded input ({}x{}) smaller than effective kernel ({}x{})",
            padded_h, padded_w, effective_k_h, effective_k_w
        )));
    }

    let out_h = (padded_h - effective_k_h) / stride[0] + 1;
    let out_w = (padded_w - effective_k_w) / stride[1] + 1;

    let oc_per_group = out_channels / groups;

    // --- Get raw data slices ---
    let in_data: &[T::Native] = input.data().typed_data();
    let w_data: &[T::Native] = weight.data().typed_data();
    let bias_vals = bias.map(|b| b.values());

    // --- Pre-compute strides for NCHW layout ---
    let in_stride_n = in_channels * in_h * in_w;
    let in_stride_c = in_h * in_w;
    // in_stride_h = in_w (implicit)

    let w_stride_oc = ic_per_group * k_h * k_w;
    let w_stride_ic = k_h * k_w;
    // w_stride_kh = k_w (implicit)

    let out_stride_n = out_channels * out_h * out_w;
    let out_stride_c = out_h * out_w;
    // out_stride_h = out_w (implicit)

    // --- Allocate output ---
    let out_len = batch * out_channels * out_h * out_w;
    let mut out = vec![T::Native::zero(); out_len];

    // --- Naive conv2d: 7 nested loops ---
    let zero = T::Native::zero();

    for n in 0..batch {
        for g in 0..groups {
            for oc_local in 0..oc_per_group {
                let oc = g * oc_per_group + oc_local;

                // Start with bias if present
                let bias_val = match bias_vals {
                    Some(bv) => bv[oc],
                    None => zero,
                };

                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = bias_val;

                        for ic in 0..ic_per_group {
                            let in_c = g * ic_per_group + ic;

                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let ih = oh * stride[0] + kh * dilation[0];
                                    let iw = ow * stride[1] + kw * dilation[1];

                                    // ih/iw are in padded coordinates; convert
                                    // to original input coordinates.
                                    let ih_orig = ih as isize - padding[0] as isize;
                                    let iw_orig = iw as isize - padding[1] as isize;

                                    if ih_orig >= 0
                                        && ih_orig < in_h as isize
                                        && iw_orig >= 0
                                        && iw_orig < in_w as isize
                                    {
                                        let ih_orig = ih_orig as usize;
                                        let iw_orig = iw_orig as usize;

                                        let in_idx = n * in_stride_n
                                            + in_c * in_stride_c
                                            + ih_orig * in_w
                                            + iw_orig;
                                        let w_idx =
                                            oc * w_stride_oc + ic * w_stride_ic + kh * k_w + kw;

                                        acc = acc + in_data[in_idx] * w_data[w_idx];
                                    }
                                }
                            }
                        }

                        let out_idx = n * out_stride_n + oc * out_stride_c + oh * out_w + ow;
                        out[out_idx] = acc;
                    }
                }
            }
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(vec![batch, out_channels, out_h, out_w]), None)
        .map_err(KernelError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float32Array;
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::Float32Type;

    /// Helper: create an N-dimensional f32 tensor from flat data + shape.
    fn make_f32_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
        let buffer = ScalarBuffer::<f32>::from(data).into_inner();
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    /// Shorthand for no-bias, stride-1, no-padding, no-dilation, 1-group conv.
    fn simple_conv2d(
        input: &Tensor<'_, Float32Type>,
        weight: &Tensor<'_, Float32Type>,
    ) -> Result<Tensor<'static, Float32Type>> {
        conv2d::<Float32Type>(input, weight, None, [0, 0], [1, 1], [1, 1], 1)
    }

    // -----------------------------------------------------------------------
    // Test: 1x1 convolution (acts as a per-pixel linear transform)
    // -----------------------------------------------------------------------
    #[test]
    fn test_conv2d_1x1_identity() {
        // input: 1 batch, 1 channel, 3x3
        // weight: 1 output channel, 1 input channel, 1x1 kernel with value 1.0
        // Expect output == input.
        #[rustfmt::skip]
        let input = make_f32_tensor(
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ],
            vec![1, 1, 3, 3],
        );
        let weight = make_f32_tensor(vec![1.0], vec![1, 1, 1, 1]);

        let out = simple_conv2d(&input, &weight).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1, 3, 3]);
        let data = out.data().typed_data::<f32>();
        for (i, &val) in data.iter().enumerate().take(9) {
            assert!(
                (val - (i as f32 + 1.0)).abs() < 1e-6,
                "mismatch at {i}: got {} expected {}",
                val,
                i as f32 + 1.0
            );
        }
    }

    #[test]
    fn test_conv2d_1x1_scaling() {
        // 1x1 conv with weight=2.0 should double every element
        #[rustfmt::skip]
        let input = make_f32_tensor(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 1, 2, 2],
        );
        let weight = make_f32_tensor(vec![2.0], vec![1, 1, 1, 1]);

        let out = simple_conv2d(&input, &weight).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[2.0, 4.0, 6.0, 8.0]);
    }

    // -----------------------------------------------------------------------
    // Test: 3x3 convolution with padding
    // -----------------------------------------------------------------------
    #[test]
    fn test_conv2d_3x3_no_padding() {
        // 1 batch, 1 channel, 4x4 input, 3x3 kernel of all-ones -> sum pooling
        // Output should be 2x2
        #[rustfmt::skip]
        let input = make_f32_tensor(
            vec![
                1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0,
            ],
            vec![1, 1, 4, 4],
        );
        let weight = make_f32_tensor(vec![1.0; 9], vec![1, 1, 3, 3]);

        let out = simple_conv2d(&input, &weight).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1, 2, 2]);
        let data = out.data().typed_data::<f32>();
        // Each output element sums a 3x3 window of 1s = 9.0
        for v in data {
            assert!((*v - 9.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_conv2d_3x3_with_padding() {
        // 1 batch, 1 channel, 3x3 input, 3x3 kernel of all-ones, padding=[1,1]
        // Same-padding: output should also be 3x3.
        #[rustfmt::skip]
        let input = make_f32_tensor(
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
            ],
            vec![1, 1, 3, 3],
        );
        let weight = make_f32_tensor(vec![1.0; 9], vec![1, 1, 3, 3]);

        let out = conv2d::<Float32Type>(&input, &weight, None, [1, 1], [1, 1], [1, 1], 1).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1, 3, 3]);

        let data = out.data().typed_data::<f32>();
        // Corner (0,0): sums 1+2+4+5 = 12
        assert!((data[0] - 12.0).abs() < 1e-6);
        // Center (1,1): sums all 9 values = 45
        assert!((data[4] - 45.0).abs() < 1e-6);
        // Corner (2,2): sums 5+6+8+9 = 28
        assert!((data[8] - 28.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test: stride > 1
    // -----------------------------------------------------------------------
    #[test]
    fn test_conv2d_stride_2() {
        // 1 batch, 1 channel, 4x4 input, 1x1 kernel (identity), stride=[2,2]
        // Should pick every other element: output 2x2
        #[rustfmt::skip]
        let input = make_f32_tensor(
            vec![
                 1.0,  2.0,  3.0,  4.0,
                 5.0,  6.0,  7.0,  8.0,
                 9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![1, 1, 4, 4],
        );
        let weight = make_f32_tensor(vec![1.0], vec![1, 1, 1, 1]);

        let out = conv2d::<Float32Type>(&input, &weight, None, [0, 0], [2, 2], [1, 1], 1).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1, 2, 2]);
        let data = out.data().typed_data::<f32>();
        // Picks (0,0)=1, (0,2)=3, (2,0)=9, (2,2)=11
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 3.0).abs() < 1e-6);
        assert!((data[2] - 9.0).abs() < 1e-6);
        assert!((data[3] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_conv2d_3x3_stride_2() {
        // 1 batch, 1 channel, 5x5 input, 3x3 all-ones kernel, stride=[2,2], no padding
        // out_h = (5-3)/2 + 1 = 2, out_w = 2
        let input = make_f32_tensor(vec![1.0; 25], vec![1, 1, 5, 5]);
        let weight = make_f32_tensor(vec![1.0; 9], vec![1, 1, 3, 3]);

        let out = conv2d::<Float32Type>(&input, &weight, None, [0, 0], [2, 2], [1, 1], 1).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1, 2, 2]);
        let data = out.data().typed_data::<f32>();
        for v in data {
            assert!((*v - 9.0).abs() < 1e-6);
        }
    }

    // -----------------------------------------------------------------------
    // Test: with bias
    // -----------------------------------------------------------------------
    #[test]
    fn test_conv2d_with_bias() {
        // 1x1 identity kernel + bias of 10.0 per output channel
        #[rustfmt::skip]
        let input = make_f32_tensor(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 1, 2, 2],
        );
        let weight = make_f32_tensor(vec![1.0], vec![1, 1, 1, 1]);
        let bias = Float32Array::from(vec![10.0]);

        let out =
            conv2d::<Float32Type>(&input, &weight, Some(&bias), [0, 0], [1, 1], [1, 1], 1).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_conv2d_multi_oc_with_bias() {
        // 1 batch, 1 in_channel, 2x2 input
        // 2 output channels: weight[0] = all-ones, weight[1] = all-twos
        // bias = [0.5, -0.5]
        let input = make_f32_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        // Weight shape: (2, 1, 1, 1) -- two 1x1 filters over 1 input channel
        let weight = make_f32_tensor(vec![1.0, 2.0], vec![2, 1, 1, 1]);
        let bias = Float32Array::from(vec![0.5, -0.5]);

        let out =
            conv2d::<Float32Type>(&input, &weight, Some(&bias), [0, 0], [1, 1], [1, 1], 1).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 2, 2, 2]);
        let data = out.data().typed_data::<f32>();
        // OC 0: input * 1.0 + 0.5 = [1.5, 2.5, 3.5, 4.5]
        assert!((data[0] - 1.5).abs() < 1e-6);
        assert!((data[1] - 2.5).abs() < 1e-6);
        assert!((data[2] - 3.5).abs() < 1e-6);
        assert!((data[3] - 4.5).abs() < 1e-6);
        // OC 1: input * 2.0 - 0.5 = [1.5, 3.5, 5.5, 7.5]
        assert!((data[4] - 1.5).abs() < 1e-6);
        assert!((data[5] - 3.5).abs() < 1e-6);
        assert!((data[6] - 5.5).abs() < 1e-6);
        assert!((data[7] - 7.5).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test: grouped convolution (depthwise)
    // -----------------------------------------------------------------------
    #[test]
    fn test_conv2d_depthwise() {
        // 1 batch, 2 channels, 3x3 input
        // Depthwise: groups=2, out_channels=2
        // Weight shape: (2, 1, 1, 1) -- each channel gets its own 1x1 filter
        // Channel 0 filter = 1.0, Channel 1 filter = 3.0
        #[rustfmt::skip]
        let input = make_f32_tensor(
            vec![
                // Channel 0
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
                // Channel 1
                10.0, 20.0, 30.0,
                40.0, 50.0, 60.0,
                70.0, 80.0, 90.0,
            ],
            vec![1, 2, 3, 3],
        );
        // Weight: (out_channels=2, ic_per_group=1, kH=1, kW=1)
        let weight = make_f32_tensor(vec![1.0, 3.0], vec![2, 1, 1, 1]);

        let out = conv2d::<Float32Type>(&input, &weight, None, [0, 0], [1, 1], [1, 1], 2).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 2, 3, 3]);
        let data = out.data().typed_data::<f32>();

        // Channel 0: input * 1.0
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[4] - 5.0).abs() < 1e-6);
        assert!((data[8] - 9.0).abs() < 1e-6);

        // Channel 1: input * 3.0
        assert!((data[9] - 30.0).abs() < 1e-6);
        assert!((data[13] - 150.0).abs() < 1e-6);
        assert!((data[17] - 270.0).abs() < 1e-6);
    }

    #[test]
    fn test_conv2d_depthwise_3x3() {
        // 1 batch, 2 channels, 4x4 input
        // Depthwise with 3x3 kernel, padding=1 (same)
        // Channel 0 kernel = all 1s, Channel 1 kernel = all 0s except center = 1
        #[rustfmt::skip]
        let input = make_f32_tensor(
            vec![
                // Channel 0: all ones
                1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0,
                // Channel 1: sequential
                1.0,  2.0,  3.0,  4.0,
                5.0,  6.0,  7.0,  8.0,
                9.0, 10.0, 11.0, 12.0,
               13.0, 14.0, 15.0, 16.0,
            ],
            vec![1, 2, 4, 4],
        );
        // Weight shape: (2, 1, 3, 3)
        #[rustfmt::skip]
        let weight = make_f32_tensor(
            vec![
                // Filter for channel 0: all ones (sum pooling)
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                // Filter for channel 1: identity (center=1)
                0.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 0.0,
            ],
            vec![2, 1, 3, 3],
        );

        let out = conv2d::<Float32Type>(&input, &weight, None, [1, 1], [1, 1], [1, 1], 2).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 2, 4, 4]);
        let data = out.data().typed_data::<f32>();

        // Channel 0 at (1,1): full 3x3 window of 1s = 9.0
        assert!((data[5] - 9.0).abs() < 1e-6);
        // Channel 0 at (0,0): corner, only 4 elements visible = 4.0
        assert!((data[0] - 4.0).abs() < 1e-6);

        // Channel 1: identity kernel preserves input values
        // data[16..32] should match input channel 1
        let ch1_start = 16;
        let expected_ch1 = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        for i in 0..16 {
            assert!(
                (data[ch1_start + i] - expected_ch1[i]).abs() < 1e-6,
                "ch1 mismatch at {i}: got {} expected {}",
                data[ch1_start + i],
                expected_ch1[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: batched input
    // -----------------------------------------------------------------------
    #[test]
    fn test_conv2d_batched() {
        // 2 batches, 1 channel, 2x2 input, 1x1 kernel = 2.0
        let input = make_f32_tensor(
            vec![
                1.0, 2.0, 3.0, 4.0, // batch 0
                5.0, 6.0, 7.0, 8.0, // batch 1
            ],
            vec![2, 1, 2, 2],
        );
        let weight = make_f32_tensor(vec![2.0], vec![1, 1, 1, 1]);

        let out = simple_conv2d(&input, &weight).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 1, 2, 2]);
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    }

    // -----------------------------------------------------------------------
    // Test: dilation
    // -----------------------------------------------------------------------
    #[test]
    fn test_conv2d_dilation() {
        // 1 batch, 1 channel, 5x5 input
        // 3x3 kernel with dilation=2 -> effective kernel 5x5
        // So output with no padding = (5 - 5)/1 + 1 = 1x1
        // All-ones kernel: sums the 9 dilated positions
        #[rustfmt::skip]
        let input = make_f32_tensor(
            vec![
                1.0, 0.0, 2.0, 0.0, 3.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                4.0, 0.0, 5.0, 0.0, 6.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                7.0, 0.0, 8.0, 0.0, 9.0,
            ],
            vec![1, 1, 5, 5],
        );
        let weight = make_f32_tensor(vec![1.0; 9], vec![1, 1, 3, 3]);

        let out = conv2d::<Float32Type>(&input, &weight, None, [0, 0], [1, 1], [2, 2], 1).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1, 1, 1]);
        let data = out.data().typed_data::<f32>();
        // Dilated kernel hits positions (0,0),(0,2),(0,4),(2,0),(2,2),(2,4),(4,0),(4,2),(4,4)
        // = 1+2+3+4+5+6+7+8+9 = 45
        assert!((data[0] - 45.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test: multi-channel (non-grouped)
    // -----------------------------------------------------------------------
    #[test]
    fn test_conv2d_multi_channel_input() {
        // 1 batch, 2 in_channels, 2x2 input
        // 1 output channel, kernel 1x1 => weight shape (1, 2, 1, 1)
        // weight = [0.5, 0.5] -> averages the two channels per pixel
        let input = make_f32_tensor(
            vec![
                // Channel 0
                1.0, 2.0, 3.0, 4.0, // Channel 1
                10.0, 20.0, 30.0, 40.0,
            ],
            vec![1, 2, 2, 2],
        );
        let weight = make_f32_tensor(vec![0.5, 0.5], vec![1, 2, 1, 1]);

        let out = simple_conv2d(&input, &weight).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1, 2, 2]);
        let data = out.data().typed_data::<f32>();
        // (0.5*1 + 0.5*10, 0.5*2 + 0.5*20, 0.5*3 + 0.5*30, 0.5*4 + 0.5*40)
        assert!((data[0] - 5.5).abs() < 1e-6);
        assert!((data[1] - 11.0).abs() < 1e-6);
        assert!((data[2] - 16.5).abs() < 1e-6);
        assert!((data[3] - 22.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test: error cases
    // -----------------------------------------------------------------------
    #[test]
    fn test_conv2d_invalid_groups() {
        // in_channels=3 is not divisible by groups=2
        let input = make_f32_tensor(vec![1.0; 12], vec![1, 3, 2, 2]);
        // weight with ic_per_group=1 (matches groups=2 expectation for 2 in_channels,
        // but in_channels=3 is not divisible by 2)
        let weight = make_f32_tensor(vec![1.0; 2], vec![2, 1, 1, 1]);
        let result = conv2d::<Float32Type>(&input, &weight, None, [0, 0], [1, 1], [1, 1], 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv2d_bias_length_mismatch() {
        let input = make_f32_tensor(vec![1.0; 4], vec![1, 1, 2, 2]);
        let weight = make_f32_tensor(vec![1.0], vec![1, 1, 1, 1]);
        let bad_bias = Float32Array::from(vec![1.0, 2.0]); // length 2, but out_channels=1
        let result =
            conv2d::<Float32Type>(&input, &weight, Some(&bad_bias), [0, 0], [1, 1], [1, 1], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv2d_wrong_input_dim() {
        // 3D instead of 4D
        let input = make_f32_tensor(vec![1.0; 8], vec![2, 2, 2]);
        let weight = make_f32_tensor(vec![1.0], vec![1, 1, 1, 1]);
        let result = simple_conv2d(&input, &weight);
        assert!(result.is_err());
    }
}

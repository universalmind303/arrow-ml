use arrow::array::{ArrowPrimitiveType, PrimitiveArray};
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_kernels_common::{KernelError, Result};
use num_traits::{Float, Zero};

/// Naive 2D transposed convolution (deconvolution) with direct loops.
///
/// Implements the ONNX ConvTranspose operator semantics for 2D spatial data.
/// This is the gradient (adjoint) of a standard convolution with respect to its
/// input, commonly used for learned upsampling in decoder networks.
///
/// # Arguments
///
/// * `input`          - 4D tensor in NCHW layout: (batch, in_channels, height, width)
/// * `weight`         - 4D tensor: (in_channels, out_channels/groups, kernel_h, kernel_w)
///                      NOTE: transposed relative to conv2d weights
/// * `bias`           - Optional 1D bias array of length `out_channels`
/// * `padding`        - `[pad_h, pad_w]` — padding subtracted from the output spatial dims
/// * `output_padding` - `[opad_h, opad_w]` — extra size added to one side of the output
/// * `stride`         - `[stride_h, stride_w]`
/// * `dilation`       - `[dilation_h, dilation_w]`
/// * `groups`         - Number of groups for grouped/depthwise transposed convolution
///
/// # Output shape
///
/// ```text
/// out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (k_h - 1) + output_pad_h + 1
/// out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (k_w - 1) + output_pad_w + 1
/// ```
///
/// # Returns
///
/// 4D tensor in NCHW layout: (batch, out_channels, out_h, out_w)
pub fn conv_transpose2d<T>(
    input: &Tensor<'_, T>,
    weight: &Tensor<'_, T>,
    bias: Option<&PrimitiveArray<T>>,
    padding: [usize; 2],
    output_padding: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    // --- Extract and validate input shape (N, C, H, W) ---
    let in_shape = input.shape().ok_or_else(|| {
        KernelError::InvalidArgument("conv_transpose2d: input has no shape".into())
    })?;
    if in_shape.len() != 4 {
        return Err(KernelError::InvalidArgument(format!(
            "conv_transpose2d: expected 4D input, got {}D",
            in_shape.len()
        )));
    }
    let (batch, in_channels, in_h, in_w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);

    // --- Extract and validate weight shape (IC, OC/groups, kH, kW) ---
    let w_shape = weight.shape().ok_or_else(|| {
        KernelError::InvalidArgument("conv_transpose2d: weight has no shape".into())
    })?;
    if w_shape.len() != 4 {
        return Err(KernelError::InvalidArgument(format!(
            "conv_transpose2d: expected 4D weight, got {}D",
            w_shape.len()
        )));
    }
    // Weight: (in_channels, oc_per_group, k_h, k_w)
    let (w_ic, oc_per_group, k_h, k_w) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);

    // --- Validate groups ---
    if groups == 0 {
        return Err(KernelError::InvalidArgument(
            "conv_transpose2d: groups must be >= 1".into(),
        ));
    }
    if in_channels % groups != 0 || w_ic != in_channels {
        return Err(KernelError::ShapeMismatch {
            operation: "conv_transpose2d",
            expected: format!("weight IC = in_channels = {in_channels}"),
            actual: format!("weight IC = {w_ic}"),
        });
    }
    let ic_per_group = in_channels / groups;
    let out_channels = oc_per_group * groups;

    // --- Validate bias ---
    if let Some(b) = bias {
        if b.len() != out_channels {
            return Err(KernelError::ShapeMismatch {
                operation: "conv_transpose2d",
                expected: format!("bias length {out_channels}"),
                actual: format!("bias length {}", b.len()),
            });
        }
    }

    // --- Validate stride and dilation ---
    if stride[0] == 0 || stride[1] == 0 {
        return Err(KernelError::InvalidArgument(
            "conv_transpose2d: stride must be >= 1".into(),
        ));
    }
    if dilation[0] == 0 || dilation[1] == 0 {
        return Err(KernelError::InvalidArgument(
            "conv_transpose2d: dilation must be >= 1".into(),
        ));
    }

    // --- Compute output spatial dimensions ---
    let out_h =
        (in_h - 1) * stride[0] + dilation[0] * (k_h - 1) + output_padding[0] + 1 - 2 * padding[0];
    let out_w =
        (in_w - 1) * stride[1] + dilation[1] * (k_w - 1) + output_padding[1] + 1 - 2 * padding[1];

    // --- Get raw data slices ---
    let in_data: &[T::Native] = input.data().typed_data();
    let w_data: &[T::Native] = weight.data().typed_data();
    let bias_vals = bias.map(|b| b.values());

    // --- Allocate output and initialise with bias ---
    let out_len = batch * out_channels * out_h * out_w;
    let mut out = vec![T::Native::zero(); out_len];

    if let Some(ref bv) = bias_vals {
        for n in 0..batch {
            for oc in 0..out_channels {
                let base = n * out_channels * out_h * out_w + oc * out_h * out_w;
                for i in 0..out_h * out_w {
                    out[base + i] = bv[oc];
                }
            }
        }
    }

    // --- Scatter: for each input position, add contributions to output ---
    // Weight layout: (in_channels, oc_per_group, k_h, k_w)
    let w_stride_ic = oc_per_group * k_h * k_w;
    let w_stride_oc = k_h * k_w;

    for n in 0..batch {
        for g in 0..groups {
            for ic_local in 0..ic_per_group {
                let ic = g * ic_per_group + ic_local;
                for ih in 0..in_h {
                    for iw in 0..in_w {
                        let in_val = in_data
                            [n * in_channels * in_h * in_w + ic * in_h * in_w + ih * in_w + iw];

                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let oh_raw = ih * stride[0] + kh * dilation[0];
                                let ow_raw = iw * stride[1] + kw * dilation[1];

                                // Apply padding
                                if oh_raw < padding[0] || ow_raw < padding[1] {
                                    continue;
                                }
                                let oh = oh_raw - padding[0];
                                let ow = ow_raw - padding[1];
                                if oh >= out_h || ow >= out_w {
                                    continue;
                                }

                                for oc_local in 0..oc_per_group {
                                    let oc = g * oc_per_group + oc_local;
                                    let w_idx =
                                        ic * w_stride_ic + oc_local * w_stride_oc + kh * k_w + kw;
                                    let out_idx = n * out_channels * out_h * out_w
                                        + oc * out_h * out_w
                                        + oh * out_w
                                        + ow;
                                    out[out_idx] = out[out_idx] + in_val * w_data[w_idx];
                                }
                            }
                        }
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

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_conv_transpose2d_1x1_identity() {
        // 1x1 kernel with stride 1 should be identity
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let weight = make_f32(vec![1.0], vec![1, 1, 1, 1]);
        let out = conv_transpose2d::<Float32Type>(
            &input,
            &weight,
            None,
            [0, 0],
            [0, 0],
            [1, 1],
            [1, 1],
            1,
        )
        .unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1, 2, 2]);
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_conv_transpose2d_stride2() {
        // 1x1 kernel with stride 2 upsamples: 2x2 -> 3x3
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let weight = make_f32(vec![1.0], vec![1, 1, 1, 1]);
        let out = conv_transpose2d::<Float32Type>(
            &input,
            &weight,
            None,
            [0, 0],
            [0, 0],
            [2, 2],
            [1, 1],
            1,
        )
        .unwrap();
        // out_h = (2-1)*2 + 1*(1-1) + 0 + 1 - 0 = 3
        assert_eq!(out.shape().unwrap(), &vec![1, 1, 3, 3]);
        let data = out.data().typed_data::<f32>();
        // Input at (0,0)=1 maps to output (0,0)
        // Input at (0,1)=2 maps to output (0,2)
        // Input at (1,0)=3 maps to output (2,0)
        // Input at (1,1)=4 maps to output (2,2)
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[2] - 2.0).abs() < 1e-6);
        assert!((data[6] - 3.0).abs() < 1e-6);
        assert!((data[8] - 4.0).abs() < 1e-6);
        // Zeros in between
        assert!((data[1] - 0.0).abs() < 1e-6);
        assert!((data[4] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_conv_transpose2d_with_bias() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let weight = make_f32(vec![1.0], vec![1, 1, 1, 1]);
        let bias = Float32Array::from(vec![5.0]);
        let out = conv_transpose2d::<Float32Type>(
            &input,
            &weight,
            Some(&bias),
            [0, 0],
            [0, 0],
            [1, 1],
            [1, 1],
            1,
        )
        .unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_conv_transpose2d_3x3_kernel_stride2() {
        // 1 batch, 1 channel, 1x1 input, 3x3 kernel, stride 2
        // out_h = (1-1)*2 + 1*(3-1) + 0 + 1 = 3
        let input = make_f32(vec![1.0], vec![1, 1, 1, 1]);
        let weight = make_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![1, 1, 3, 3],
        );
        let out = conv_transpose2d::<Float32Type>(
            &input,
            &weight,
            None,
            [0, 0],
            [0, 0],
            [2, 2],
            [1, 1],
            1,
        )
        .unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1, 3, 3]);
        let data = out.data().typed_data::<f32>();
        // Single input value 1.0 scattered through 3x3 kernel
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_conv_transpose2d_output_shape() {
        // Standard test: 2x2 input, 3x3 kernel, stride 2, padding 1
        // out_h = (2-1)*2 + 1*(3-1) + 0 + 1 - 2*1 = 2+2+1-2 = 3
        let input = make_f32(vec![0.0; 4], vec![1, 1, 2, 2]);
        let weight = make_f32(vec![0.0; 9], vec![1, 1, 3, 3]);
        let out = conv_transpose2d::<Float32Type>(
            &input,
            &weight,
            None,
            [1, 1],
            [0, 0],
            [2, 2],
            [1, 1],
            1,
        )
        .unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1, 3, 3]);
    }
}

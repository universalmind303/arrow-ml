use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_kernels_common::KernelError;
use arrow_kernels_common::Result;
use num_traits::{Float, Zero};

/// Extracts NCHW shape from a 4D tensor, returning an error if not 4D.
fn shape_nchw<T: ArrowPrimitiveType>(
    tensor: &Tensor<'_, T>,
    operation: &'static str,
) -> Result<(usize, usize, usize, usize)> {
    let shape = tensor
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument(format!("{operation}: tensor has no shape")))?;
    if shape.len() != 4 {
        return Err(KernelError::InvalidArgument(format!(
            "{operation}: expected 4D NCHW tensor, got {}D",
            shape.len()
        )));
    }
    Ok((shape[0], shape[1], shape[2], shape[3]))
}

/// Computes the output spatial dimension for a pooling operation.
///
/// out = floor((input + 2*pad - kernel) / stride) + 1
fn pool_output_size(input: usize, kernel: usize, stride: usize, pad: usize) -> Result<usize> {
    let numerator = input + 2 * pad;
    if numerator < kernel {
        return Err(KernelError::InvalidArgument(format!(
            "pooling: input size {input} + 2*padding {pad} < kernel size {kernel}"
        )));
    }
    Ok((numerator - kernel) / stride + 1)
}

/// 2D max pooling over an NCHW tensor.
///
/// For each spatial window of size `kernel_size`, takes the maximum value.
///
/// - `input`: 4D tensor in NCHW layout (batch, channels, height, width).
/// - `kernel_size`: `[kH, kW]` pooling window dimensions.
/// - `stride`: `[sH, sW]` stride in each spatial dimension.
/// - `padding`: `[pH, pW]` zero-padding added to both sides of each spatial dimension.
///
/// Returns an NCHW tensor with spatial dimensions:
///   out_h = floor((H + 2*pH - kH) / sH) + 1
///   out_w = floor((W + 2*pW - kW) / sW) + 1
pub fn max_pool2d<T>(
    input: &Tensor<'_, T>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let (n, c, h, w) = shape_nchw(input, "max_pool2d")?;
    let [kh, kw] = kernel_size;
    let [sh, sw] = stride;
    let [ph, pw] = padding;

    if kh == 0 || kw == 0 {
        return Err(KernelError::InvalidArgument(
            "max_pool2d: kernel size must be > 0".to_string(),
        ));
    }
    if sh == 0 || sw == 0 {
        return Err(KernelError::InvalidArgument(
            "max_pool2d: stride must be > 0".to_string(),
        ));
    }

    let out_h = pool_output_size(h, kh, sh, ph)?;
    let out_w = pool_output_size(w, kw, sw, pw)?;

    let data: &[T::Native] = input.data().typed_data();
    let mut out = Vec::with_capacity(n * c * out_h * out_w);

    for batch in 0..n {
        for ch in 0..c {
            let base = batch * c * h * w + ch * h * w;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = T::Native::neg_infinity();
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = oh * sh + ki;
                            let iw = ow * sw + kj;
                            // Check if we are in the padded region.
                            if ih >= ph && ih < h + ph && iw >= pw && iw < w + pw {
                                let val = data[base + (ih - ph) * w + (iw - pw)];
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    out.push(max_val);
                }
            }
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(vec![n, c, out_h, out_w]), None).map_err(KernelError::from)
}

/// 2D average pooling over an NCHW tensor.
///
/// For each spatial window of size `kernel_size`, computes the arithmetic mean.
/// Padded positions contribute zero and are included in the divisor (count_include_pad=true),
/// matching the default ONNX AveragePool behavior.
///
/// - `input`: 4D tensor in NCHW layout (batch, channels, height, width).
/// - `kernel_size`: `[kH, kW]` pooling window dimensions.
/// - `stride`: `[sH, sW]` stride in each spatial dimension.
/// - `padding`: `[pH, pW]` zero-padding added to both sides of each spatial dimension.
///
/// Returns an NCHW tensor with spatial dimensions:
///   out_h = floor((H + 2*pH - kH) / sH) + 1
///   out_w = floor((W + 2*pW - kW) / sW) + 1
pub fn avg_pool2d<T>(
    input: &Tensor<'_, T>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let (n, c, h, w) = shape_nchw(input, "avg_pool2d")?;
    let [kh, kw] = kernel_size;
    let [sh, sw] = stride;
    let [ph, pw] = padding;

    if kh == 0 || kw == 0 {
        return Err(KernelError::InvalidArgument(
            "avg_pool2d: kernel size must be > 0".to_string(),
        ));
    }
    if sh == 0 || sw == 0 {
        return Err(KernelError::InvalidArgument(
            "avg_pool2d: stride must be > 0".to_string(),
        ));
    }

    let out_h = pool_output_size(h, kh, sh, ph)?;
    let out_w = pool_output_size(w, kw, sw, pw)?;

    let data: &[T::Native] = input.data().typed_data();
    let pool_area: T::Native =
        <T::Native as num_traits::NumCast>::from(kh * kw).unwrap();
    let mut out = Vec::with_capacity(n * c * out_h * out_w);

    for batch in 0..n {
        for ch in 0..c {
            let base = batch * c * h * w + ch * h * w;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = T::Native::zero();
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = oh * sh + ki;
                            let iw = ow * sw + kj;
                            if ih >= ph && ih < h + ph && iw >= pw && iw < w + pw {
                                sum = sum + data[base + (ih - ph) * w + (iw - pw)];
                            }
                            // Padded positions contribute 0 (already zero-init in sum).
                        }
                    }
                    out.push(sum / pool_area);
                }
            }
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(vec![n, c, out_h, out_w]), None).map_err(KernelError::from)
}

/// Global average pooling: reduces each (H, W) plane to a single scalar.
///
/// Input: NCHW tensor of shape (N, C, H, W).
/// Output: NCHW tensor of shape (N, C, 1, 1), where each value is the mean
/// over the corresponding spatial plane.
pub fn global_avg_pool<T>(input: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let (n, c, h, w) = shape_nchw(input, "global_avg_pool")?;
    let spatial = h * w;
    if spatial == 0 {
        return Err(KernelError::InvalidArgument(
            "global_avg_pool: spatial dimensions must be > 0".to_string(),
        ));
    }

    let data: &[T::Native] = input.data().typed_data();
    let count: T::Native = <T::Native as num_traits::NumCast>::from(spatial).unwrap();
    let mut out = Vec::with_capacity(n * c);

    for batch in 0..n {
        for ch in 0..c {
            let base = batch * c * h * w + ch * h * w;
            let mut sum = T::Native::zero();
            for i in 0..spatial {
                sum = sum + data[base + i];
            }
            out.push(sum / count);
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(vec![n, c, 1, 1]), None).map_err(KernelError::from)
}

/// Global max pooling: reduces each (H, W) plane to a single maximum value.
///
/// Input: NCHW tensor of shape (N, C, H, W).
/// Output: NCHW tensor of shape (N, C, 1, 1), where each value is the maximum
/// over the corresponding spatial plane.
pub fn global_max_pool<T>(input: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let (n, c, h, w) = shape_nchw(input, "global_max_pool")?;
    let spatial = h * w;
    if spatial == 0 {
        return Err(KernelError::InvalidArgument(
            "global_max_pool: spatial dimensions must be > 0".to_string(),
        ));
    }

    let data: &[T::Native] = input.data().typed_data();
    let mut out = Vec::with_capacity(n * c);

    for batch in 0..n {
        for ch in 0..c {
            let base = batch * c * h * w + ch * h * w;
            let mut max_val = T::Native::neg_infinity();
            for i in 0..spatial {
                let val = data[base + i];
                if val > max_val {
                    max_val = val;
                }
            }
            out.push(max_val);
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(vec![n, c, 1, 1]), None).map_err(KernelError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::Float32Type;

    /// Helper: create an NCHW f32 tensor from flat data.
    fn make_nchw(
        data: Vec<f32>,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> Tensor<'static, Float32Type> {
        assert_eq!(data.len(), n * c * h * w, "data length mismatch");
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(vec![n, c, h, w]), None).unwrap()
    }

    // ----- max_pool2d tests -----

    #[test]
    fn test_max_pool2d_basic() {
        // 1x1x4x4 input, kernel 2x2, stride 1, no padding
        // Input (single channel):
        //  1  2  3  4
        //  5  6  7  8
        //  9 10 11 12
        // 13 14 15 16
        #[rustfmt::skip]
        let data = vec![
             1.0,  2.0,  3.0,  4.0,
             5.0,  6.0,  7.0,  8.0,
             9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = make_nchw(data, 1, 1, 4, 4);
        let out = max_pool2d::<Float32Type>(&input, [2, 2], [1, 1], [0, 0]).unwrap();

        let shape = out.shape().unwrap();
        assert_eq!(shape, &vec![1, 1, 3, 3]);

        let result: &[f32] = out.data().typed_data();
        #[rustfmt::skip]
        let expected = [
             6.0,  7.0,  8.0,
            10.0, 11.0, 12.0,
            14.0, 15.0, 16.0,
        ];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "max_pool2d basic: mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_max_pool2d_stride2() {
        // 1x1x4x4 input, kernel 2x2, stride 2, no padding -> 2x2 output
        #[rustfmt::skip]
        let data = vec![
             1.0,  2.0,  3.0,  4.0,
             5.0,  6.0,  7.0,  8.0,
             9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = make_nchw(data, 1, 1, 4, 4);
        let out = max_pool2d::<Float32Type>(&input, [2, 2], [2, 2], [0, 0]).unwrap();

        let shape = out.shape().unwrap();
        assert_eq!(shape, &vec![1, 1, 2, 2]);

        let result: &[f32] = out.data().typed_data();
        // Top-left 2x2: max(1,2,5,6)=6
        // Top-right 2x2: max(3,4,7,8)=8
        // Bottom-left 2x2: max(9,10,13,14)=14
        // Bottom-right 2x2: max(11,12,15,16)=16
        assert!((result[0] - 6.0).abs() < 1e-6);
        assert!((result[1] - 8.0).abs() < 1e-6);
        assert!((result[2] - 14.0).abs() < 1e-6);
        assert!((result[3] - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_pool2d_with_padding() {
        // 1x1x2x2 input, kernel 2x2, stride 1, padding 1 -> 3x3 output
        // Input:
        //  1  2
        //  3  4
        //
        // With 1-padding the effective input becomes:
        //  0  0  0  0
        //  0  1  2  0
        //  0  3  4  0
        //  0  0  0  0
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = make_nchw(data, 1, 1, 2, 2);
        let out = max_pool2d::<Float32Type>(&input, [2, 2], [1, 1], [1, 1]).unwrap();

        let shape = out.shape().unwrap();
        assert_eq!(shape, &vec![1, 1, 3, 3]);

        let result: &[f32] = out.data().typed_data();
        // Row 0: max(0,0,0,1)=1, max(0,0,1,2)=2, max(0,0,2,0)=2
        // Row 1: max(0,1,0,3)=3, max(1,2,3,4)=4, max(2,0,4,0)=4
        // Row 2: max(0,3,0,0)=3, max(3,4,0,0)=4, max(4,0,0,0)=4
        #[rustfmt::skip]
        let expected = [
            1.0, 2.0, 2.0,
            3.0, 4.0, 4.0,
            3.0, 4.0, 4.0,
        ];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "max_pool2d padded: mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_max_pool2d_multi_batch_channel() {
        // 2 batches, 2 channels, 2x2 spatial, kernel 2x2, stride 1, no padding -> 1x1
        // Batch 0, Ch 0: [[1,2],[3,4]] -> max=4
        // Batch 0, Ch 1: [[5,6],[7,8]] -> max=8
        // Batch 1, Ch 0: [[9,10],[11,12]] -> max=12
        // Batch 1, Ch 1: [[13,14],[15,16]] -> max=16
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let input = make_nchw(data, 2, 2, 2, 2);
        let out = max_pool2d::<Float32Type>(&input, [2, 2], [1, 1], [0, 0]).unwrap();

        let shape = out.shape().unwrap();
        assert_eq!(shape, &vec![2, 2, 1, 1]);

        let result: &[f32] = out.data().typed_data();
        assert!((result[0] - 4.0).abs() < 1e-6);
        assert!((result[1] - 8.0).abs() < 1e-6);
        assert!((result[2] - 12.0).abs() < 1e-6);
        assert!((result[3] - 16.0).abs() < 1e-6);
    }

    // ----- avg_pool2d tests -----

    #[test]
    fn test_avg_pool2d_basic() {
        // 1x1x4x4 input, kernel 2x2, stride 2, no padding
        #[rustfmt::skip]
        let data = vec![
             1.0,  2.0,  3.0,  4.0,
             5.0,  6.0,  7.0,  8.0,
             9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = make_nchw(data, 1, 1, 4, 4);
        let out = avg_pool2d::<Float32Type>(&input, [2, 2], [2, 2], [0, 0]).unwrap();

        let shape = out.shape().unwrap();
        assert_eq!(shape, &vec![1, 1, 2, 2]);

        let result: &[f32] = out.data().typed_data();
        // Top-left 2x2: avg(1,2,5,6)=3.5
        // Top-right 2x2: avg(3,4,7,8)=5.5
        // Bottom-left 2x2: avg(9,10,13,14)=11.5
        // Bottom-right 2x2: avg(11,12,15,16)=13.5
        assert!((result[0] - 3.5).abs() < 1e-6);
        assert!((result[1] - 5.5).abs() < 1e-6);
        assert!((result[2] - 11.5).abs() < 1e-6);
        assert!((result[3] - 13.5).abs() < 1e-6);
    }

    #[test]
    fn test_avg_pool2d_stride1() {
        // 1x1x3x3 input, kernel 2x2, stride 1, no padding -> 2x2
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let input = make_nchw(data, 1, 1, 3, 3);
        let out = avg_pool2d::<Float32Type>(&input, [2, 2], [1, 1], [0, 0]).unwrap();

        let shape = out.shape().unwrap();
        assert_eq!(shape, &vec![1, 1, 2, 2]);

        let result: &[f32] = out.data().typed_data();
        // avg(1,2,4,5)=3.0, avg(2,3,5,6)=4.0, avg(4,5,7,8)=6.0, avg(5,6,8,9)=7.0
        assert!((result[0] - 3.0).abs() < 1e-6);
        assert!((result[1] - 4.0).abs() < 1e-6);
        assert!((result[2] - 6.0).abs() < 1e-6);
        assert!((result[3] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_avg_pool2d_with_padding() {
        // 1x1x2x2 input, kernel 2x2, stride 1, padding 1 -> 3x3
        // count_include_pad=true: divisor is always kh*kw=4
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = make_nchw(data, 1, 1, 2, 2);
        let out = avg_pool2d::<Float32Type>(&input, [2, 2], [1, 1], [1, 1]).unwrap();

        let shape = out.shape().unwrap();
        assert_eq!(shape, &vec![1, 1, 3, 3]);

        let result: &[f32] = out.data().typed_data();
        // Padded effective input:
        //  0  0  0  0
        //  0  1  2  0
        //  0  3  4  0
        //  0  0  0  0
        // Each window divided by 4:
        // (0+0+0+1)/4=0.25, (0+0+1+2)/4=0.75, (0+0+2+0)/4=0.5
        // (0+1+0+3)/4=1.0,  (1+2+3+4)/4=2.5,  (2+0+4+0)/4=1.5
        // (0+3+0+0)/4=0.75, (3+4+0+0)/4=1.75, (4+0+0+0)/4=1.0
        #[rustfmt::skip]
        let expected = [
            0.25, 0.75, 0.5,
            1.0,  2.5,  1.5,
            0.75, 1.75, 1.0,
        ];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "avg_pool2d padded: mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    // ----- global_avg_pool tests -----

    #[test]
    fn test_global_avg_pool() {
        // 1x2x2x2: two channels, each 2x2
        // Ch 0: [[1,2],[3,4]] -> mean=2.5
        // Ch 1: [[5,6],[7,8]] -> mean=6.5
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = make_nchw(data, 1, 2, 2, 2);
        let out = global_avg_pool::<Float32Type>(&input).unwrap();

        let shape = out.shape().unwrap();
        assert_eq!(shape, &vec![1, 2, 1, 1]);

        let result: &[f32] = out.data().typed_data();
        assert!((result[0] - 2.5).abs() < 1e-6);
        assert!((result[1] - 6.5).abs() < 1e-6);
    }

    #[test]
    fn test_global_avg_pool_batch() {
        // 2 batches, 1 channel, 3x3
        // Batch 0: 1..9 -> mean = 5.0
        // Batch 1: 10..18 -> mean = 14.0
        let data: Vec<f32> = (1..=18).map(|x| x as f32).collect();
        let input = make_nchw(data, 2, 1, 3, 3);
        let out = global_avg_pool::<Float32Type>(&input).unwrap();

        let shape = out.shape().unwrap();
        assert_eq!(shape, &vec![2, 1, 1, 1]);

        let result: &[f32] = out.data().typed_data();
        assert!((result[0] - 5.0).abs() < 1e-6);
        assert!((result[1] - 14.0).abs() < 1e-6);
    }

    // ----- global_max_pool tests -----

    #[test]
    fn test_global_max_pool() {
        // 1x2x2x2: two channels
        // Ch 0: [[1,2],[3,4]] -> max=4
        // Ch 1: [[-1,6],[7,0]] -> max=7
        let data = vec![1.0, 2.0, 3.0, 4.0, -1.0, 6.0, 7.0, 0.0];
        let input = make_nchw(data, 1, 2, 2, 2);
        let out = global_max_pool::<Float32Type>(&input).unwrap();

        let shape = out.shape().unwrap();
        assert_eq!(shape, &vec![1, 2, 1, 1]);

        let result: &[f32] = out.data().typed_data();
        assert!((result[0] - 4.0).abs() < 1e-6);
        assert!((result[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_global_max_pool_batch() {
        // 2 batches, 1 channel, 2x3
        // Batch 0: [1,5,3,2,4,6] -> max=6
        // Batch 1: [10,8,9,7,12,11] -> max=12
        let data = vec![1.0, 5.0, 3.0, 2.0, 4.0, 6.0, 10.0, 8.0, 9.0, 7.0, 12.0, 11.0];
        let input = make_nchw(data, 2, 1, 2, 3);
        let out = global_max_pool::<Float32Type>(&input).unwrap();

        let shape = out.shape().unwrap();
        assert_eq!(shape, &vec![2, 1, 1, 1]);

        let result: &[f32] = out.data().typed_data();
        assert!((result[0] - 6.0).abs() < 1e-6);
        assert!((result[1] - 12.0).abs() < 1e-6);
    }

    // ----- output shape verification tests -----

    #[test]
    fn test_pool_output_shapes() {
        // 1x1x7x7 input, kernel 3x3, stride 2, padding 1
        // out = floor((7 + 2 - 3) / 2) + 1 = floor(6/2) + 1 = 4
        let data = vec![0.0; 7 * 7];
        let input = make_nchw(data, 1, 1, 7, 7);

        let out = max_pool2d::<Float32Type>(&input, [3, 3], [2, 2], [1, 1]).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1, 4, 4]);

        let data = vec![0.0; 7 * 7];
        let input = make_nchw(data, 1, 1, 7, 7);
        let out = avg_pool2d::<Float32Type>(&input, [3, 3], [2, 2], [1, 1]).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1, 4, 4]);
    }

    #[test]
    fn test_pool_asymmetric_kernel() {
        // 1x1x4x6, kernel 2x3, stride 1x1, no padding
        // out_h = (4 - 2)/1 + 1 = 3
        // out_w = (6 - 3)/1 + 1 = 4
        let data = vec![0.0; 4 * 6];
        let input = make_nchw(data, 1, 1, 4, 6);
        let out = max_pool2d::<Float32Type>(&input, [2, 3], [1, 1], [0, 0]).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1, 3, 4]);
    }

    #[test]
    fn test_not_4d_errors() {
        // 2D tensor should fail.
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(vec![1.0, 2.0, 3.0, 4.0]).into_inner());
        let input: Tensor<'static, Float32Type> =
            Tensor::new_row_major(buffer, Some(vec![2, 2]), None).unwrap();
        assert!(max_pool2d::<Float32Type>(&input, [1, 1], [1, 1], [0, 0]).is_err());
        assert!(avg_pool2d::<Float32Type>(&input, [1, 1], [1, 1], [0, 0]).is_err());
        assert!(global_avg_pool::<Float32Type>(&input).is_err());
        assert!(global_max_pool::<Float32Type>(&input).is_err());
    }

    #[test]
    fn test_global_pool_single_element() {
        // 1x1x1x1 input
        let input = make_nchw(vec![42.0], 1, 1, 1, 1);
        let avg = global_avg_pool::<Float32Type>(&input).unwrap();
        let max = global_max_pool::<Float32Type>(&input).unwrap();

        assert!((avg.data().typed_data::<f32>()[0] - 42.0).abs() < 1e-6);
        assert!((max.data().typed_data::<f32>()[0] - 42.0).abs() < 1e-6);
    }
}

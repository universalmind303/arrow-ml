use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_ml_common::{KernelError, Result};
use num_traits::{Float, NumCast, One, ToPrimitive};

/// Helper to cast a `usize` to `T::Native` via `NumCast`.
#[inline]
fn from_usize<N: NumCast>(v: usize) -> N {
    <N as NumCast>::from(v).unwrap()
}

/// Helper to cast `T::Native` (a float) back to `usize`.
#[inline]
fn as_usize<N: ToPrimitive>(v: N) -> usize {
    v.to_usize().unwrap()
}

/// Resize a 4D NCHW tensor using nearest-neighbor interpolation.
pub fn resize_nearest<T>(
    input: &Tensor<'_, T>,
    output_h: usize,
    output_w: usize,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let shape = input.shape().ok_or_else(|| {
        KernelError::InvalidArgument("resize_nearest: tensor has no shape".into())
    })?;
    if shape.len() != 4 {
        return Err(KernelError::InvalidArgument(format!(
            "resize_nearest: expected 4D NCHW tensor, got {}D",
            shape.len()
        )));
    }
    let (n, c, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);

    if output_h == 0 || output_w == 0 {
        return Err(KernelError::InvalidArgument(
            "resize_nearest: output dimensions must be > 0".into(),
        ));
    }

    let data: &[T::Native] = input.data().typed_data();
    let mut out = Vec::with_capacity(n * c * output_h * output_w);

    let scale_h: T::Native = from_usize::<T::Native>(in_h) / from_usize::<T::Native>(output_h);
    let scale_w: T::Native = from_usize::<T::Native>(in_w) / from_usize::<T::Native>(output_w);

    for batch in 0..n {
        for ch in 0..c {
            let base = batch * c * in_h * in_w + ch * in_h * in_w;
            for oh in 0..output_h {
                for ow in 0..output_w {
                    // Asymmetric coordinate mapping (floor-based)
                    let ih_f = from_usize::<T::Native>(oh) * scale_h;
                    let iw_f = from_usize::<T::Native>(ow) * scale_w;
                    let ih = as_usize(ih_f.floor()).min(in_h - 1);
                    let iw = as_usize(iw_f.floor()).min(in_w - 1);
                    out.push(data[base + ih * in_w + iw]);
                }
            }
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(vec![n, c, output_h, output_w]), None)
        .map_err(KernelError::from)
}

/// Resize a 4D NCHW tensor using bilinear interpolation.
pub fn resize_bilinear<T>(
    input: &Tensor<'_, T>,
    output_h: usize,
    output_w: usize,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let shape = input.shape().ok_or_else(|| {
        KernelError::InvalidArgument("resize_bilinear: tensor has no shape".into())
    })?;
    if shape.len() != 4 {
        return Err(KernelError::InvalidArgument(format!(
            "resize_bilinear: expected 4D NCHW tensor, got {}D",
            shape.len()
        )));
    }
    let (n, c, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);

    if output_h == 0 || output_w == 0 {
        return Err(KernelError::InvalidArgument(
            "resize_bilinear: output dimensions must be > 0".into(),
        ));
    }

    let data: &[T::Native] = input.data().typed_data();
    let mut out = Vec::with_capacity(n * c * output_h * output_w);

    let scale_h: T::Native = from_usize::<T::Native>(in_h) / from_usize::<T::Native>(output_h);
    let scale_w: T::Native = from_usize::<T::Native>(in_w) / from_usize::<T::Native>(output_w);
    let one = <T::Native as One>::one();

    for batch in 0..n {
        for ch in 0..c {
            let base = batch * c * in_h * in_w + ch * in_h * in_w;
            for oh in 0..output_h {
                for ow in 0..output_w {
                    let src_h: T::Native = from_usize::<T::Native>(oh) * scale_h;
                    let src_w: T::Native = from_usize::<T::Native>(ow) * scale_w;

                    let h0_f = src_h.floor();
                    let w0_f = src_w.floor();
                    let h_frac = src_h - h0_f;
                    let w_frac = src_w - w0_f;

                    let h0 = as_usize(h0_f).min(in_h - 1);
                    let h1 = (h0 + 1).min(in_h - 1);
                    let w0 = as_usize(w0_f).min(in_w - 1);
                    let w1 = (w0 + 1).min(in_w - 1);

                    let v00 = data[base + h0 * in_w + w0];
                    let v01 = data[base + h0 * in_w + w1];
                    let v10 = data[base + h1 * in_w + w0];
                    let v11 = data[base + h1 * in_w + w1];

                    let val = v00 * (one - h_frac) * (one - w_frac)
                        + v01 * (one - h_frac) * w_frac
                        + v10 * h_frac * (one - w_frac)
                        + v11 * h_frac * w_frac;
                    out.push(val);
                }
            }
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(vec![n, c, output_h, output_w]), None)
        .map_err(KernelError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::Float32Type;

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_resize_nearest_upsample() {
        // 1x1x2x2 -> 1x1x4x4
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let out = resize_nearest::<Float32Type>(&input, 4, 4).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1, 4, 4]);
        let data = out.data().typed_data::<f32>();
        // Top-left quadrant should be 1.0, top-right 2.0, etc.
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 1.0).abs() < 1e-6);
        assert!((data[2] - 2.0).abs() < 1e-6);
        assert!((data[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_resize_nearest_downsample() {
        // 1x1x4x4 -> 1x1x2x2
        let data_in: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let input = make_f32(data_in, vec![1, 1, 4, 4]);
        let out = resize_nearest::<Float32Type>(&input, 2, 2).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1, 2, 2]);
        let data = out.data().typed_data::<f32>();
        // With floor: (0,0)->(0,0)=1, (0,1)->(0,2)=3, (1,0)->(2,0)=9, (1,1)->(2,2)=11
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 3.0).abs() < 1e-6);
        assert!((data[2] - 9.0).abs() < 1e-6);
        assert!((data[3] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_resize_nearest_identity() {
        // Same size should be identity
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let out = resize_nearest::<Float32Type>(&input, 2, 2).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_resize_bilinear_identity() {
        // Same size should be identity
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let out = resize_bilinear::<Float32Type>(&input, 2, 2).unwrap();
        let data = out.data().typed_data::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
        assert!((data[2] - 3.0).abs() < 1e-5);
        assert!((data[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_resize_bilinear_upsample() {
        // 1x1x2x2 -> 1x1x3x3, should interpolate
        let input = make_f32(vec![0.0, 1.0, 1.0, 0.0], vec![1, 1, 2, 2]);
        let out = resize_bilinear::<Float32Type>(&input, 3, 3).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1, 3, 3]);
        // Just verify no NaN/Inf
        let data = out.data().typed_data::<f32>();
        for v in data {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_resize_nearest_multi_channel() {
        // 1 batch, 2 channels, 2x2 -> 4x4
        let input = make_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1, 2, 2, 2],
        );
        let out = resize_nearest::<Float32Type>(&input, 4, 4).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 2, 4, 4]);
    }

    #[test]
    fn test_resize_not_4d() {
        let input = make_f32(vec![1.0, 2.0], vec![2]);
        assert!(resize_nearest::<Float32Type>(&input, 4, 4).is_err());
        assert!(resize_bilinear::<Float32Type>(&input, 4, 4).is_err());
    }
}

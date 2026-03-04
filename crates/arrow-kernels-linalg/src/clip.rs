use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_kernels_common::{KernelError, Result};

/// Clamp tensor values to [min, max] range.
///
/// Either min or max can be None (unbounded on that side).
pub fn clip<T>(
    input: &Tensor<'_, T>,
    min: Option<T::Native>,
    max: Option<T::Native>,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd + Copy,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("clip: tensor has no shape".into()))?
        .to_vec();

    let data: &[T::Native] = input.data().typed_data();
    let mut out = Vec::with_capacity(data.len());

    for &v in data {
        let mut val = v;
        if let Some(lo) = min {
            if val < lo {
                val = lo;
            }
        }
        if let Some(hi) = max {
            if val > hi {
                val = hi;
            }
        }
        out.push(val);
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(shape), None).map_err(KernelError::from)
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
    fn test_clip_both() {
        let input = make_f32(vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], vec![6]);
        let out = clip(&input, Some(0.0), Some(2.0)).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[0.0, 0.0, 0.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn test_clip_min_only() {
        let input = make_f32(vec![-2.0, -1.0, 0.0, 1.0], vec![4]);
        let out = clip(&input, Some(0.0), None).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_clip_max_only() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = clip(&input, None, Some(2.5)).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[1.0, 2.0, 2.5, 2.5]);
    }

    #[test]
    fn test_clip_preserves_shape() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = clip(&input, Some(2.0), Some(5.0)).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 3]);
    }
}

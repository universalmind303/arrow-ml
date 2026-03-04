use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_kernels_common::{KernelError, Result};
use num_traits::{Float, One, Zero};

/// Reduce by sum along specified axes.
pub fn reduce_sum<T>(
    input: &Tensor<'_, T>,
    axes: &[i64],
    keepdims: bool,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    reduce_impl(input, axes, keepdims, "reduce_sum", |vals| {
        vals.iter()
            .copied()
            .fold(<T::Native as Zero>::zero(), |a, b| a + b)
    })
}

/// Reduce by mean along specified axes.
pub fn reduce_mean<T>(
    input: &Tensor<'_, T>,
    axes: &[i64],
    keepdims: bool,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    reduce_impl(input, axes, keepdims, "reduce_mean", |vals| {
        let sum = vals
            .iter()
            .copied()
            .fold(<T::Native as Zero>::zero(), |a, b| a + b);
        let n = <T::Native as num_traits::NumCast>::from(vals.len()).unwrap();
        sum / n
    })
}

/// Reduce by max along specified axes.
pub fn reduce_max<T>(
    input: &Tensor<'_, T>,
    axes: &[i64],
    keepdims: bool,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    reduce_impl(input, axes, keepdims, "reduce_max", |vals| {
        vals.iter()
            .copied()
            .fold(T::Native::neg_infinity(), |a, b| if b > a { b } else { a })
    })
}

/// Reduce by min along specified axes.
pub fn reduce_min<T>(
    input: &Tensor<'_, T>,
    axes: &[i64],
    keepdims: bool,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    reduce_impl(input, axes, keepdims, "reduce_min", |vals| {
        vals.iter()
            .copied()
            .fold(T::Native::infinity(), |a, b| if b < a { b } else { a })
    })
}

/// Reduce by product along specified axes.
pub fn reduce_prod<T>(
    input: &Tensor<'_, T>,
    axes: &[i64],
    keepdims: bool,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    reduce_impl(input, axes, keepdims, "reduce_prod", |vals| {
        vals.iter()
            .copied()
            .fold(<T::Native as One>::one(), |a, b| a * b)
    })
}

fn reduce_impl<T, F>(
    input: &Tensor<'_, T>,
    axes: &[i64],
    keepdims: bool,
    op: &str,
    reducer: F,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
    F: Fn(&[T::Native]) -> T::Native,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument(format!("{op}: tensor has no shape")))?;
    let ndim = shape.len();

    let resolved = normalize_axes(axes, ndim, op)?;

    // Reduce one axis at a time, from highest to lowest (so indices remain valid)
    let mut sorted = resolved.clone();
    sorted.sort_unstable();
    sorted.dedup();

    let mut current_data: Vec<T::Native> = input.data().typed_data::<T::Native>().to_vec();
    let mut current_shape = shape.to_vec();

    for &ax in sorted.iter().rev() {
        let (new_data, new_shape) =
            reduce_single_axis::<T, F>(&current_data, &current_shape, ax, &reducer);
        current_data = new_data;
        current_shape = new_shape;
    }

    // Build final shape
    let final_shape = if keepdims {
        let mut s = shape.to_vec();
        for &ax in &sorted {
            s[ax] = 1;
        }
        s
    } else {
        current_shape
    };

    // Handle scalar result
    let final_shape = if final_shape.is_empty() {
        vec![1]
    } else {
        final_shape
    };

    let buf = Buffer::from_vec(current_data);
    Tensor::new_row_major(buf, Some(final_shape), None).map_err(KernelError::from)
}

fn normalize_axes(axes: &[i64], ndim: usize, op: &str) -> Result<Vec<usize>> {
    axes.iter()
        .map(|&a| {
            let r = if a < 0 { ndim as i64 + a } else { a };
            if r < 0 || r >= ndim as i64 {
                Err(KernelError::InvalidArgument(format!(
                    "{op}: axis {a} out of range for {ndim}D tensor"
                )))
            } else {
                Ok(r as usize)
            }
        })
        .collect()
}

fn reduce_single_axis<T, F>(
    data: &[T::Native],
    shape: &[usize],
    axis: usize,
    reducer: &F,
) -> (Vec<T::Native>, Vec<usize>)
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
    F: Fn(&[T::Native]) -> T::Native,
{
    let ndim = shape.len();
    let outer_size: usize = shape[..axis].iter().product();
    let reduce_size = shape[axis];
    let inner_size: usize = shape[axis + 1..].iter().product();

    let mut out = Vec::with_capacity(outer_size * inner_size);
    let mut tmp = Vec::with_capacity(reduce_size);

    for o in 0..outer_size {
        for i in 0..inner_size {
            tmp.clear();
            for r in 0..reduce_size {
                tmp.push(data[o * reduce_size * inner_size + r * inner_size + i]);
            }
            out.push(reducer(&tmp));
        }
    }

    let mut new_shape: Vec<usize> = (0..ndim).filter(|&d| d != axis).map(|d| shape[d]).collect();
    if new_shape.is_empty() {
        new_shape.push(1);
    }

    (out, new_shape)
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
    fn test_reduce_sum_axis0() {
        // [[1,2,3],[4,5,6]] -> sum axis 0 -> [5,7,9]
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_sum(&input, &[0], false).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![3]);
        assert_eq!(out.data().typed_data::<f32>(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_reduce_sum_axis1() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_sum(&input, &[1], false).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2]);
        assert_eq!(out.data().typed_data::<f32>(), &[6.0, 15.0]);
    }

    #[test]
    fn test_reduce_sum_keepdims() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_sum(&input, &[0], true).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 3]);
        assert_eq!(out.data().typed_data::<f32>(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_reduce_sum_all_axes() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_sum(&input, &[0, 1], false).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[21.0]);
    }

    #[test]
    fn test_reduce_mean_axis0() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_mean(&input, &[0], false).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_reduce_max_axis1() {
        let input = make_f32(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]);
        let out = reduce_max(&input, &[1], false).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[5.0, 6.0]);
    }

    #[test]
    fn test_reduce_min_axis0() {
        let input = make_f32(vec![3.0, 1.0, 5.0, 2.0, 4.0, 0.0], vec![2, 3]);
        let out = reduce_min(&input, &[0], false).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[2.0, 1.0, 0.0]);
    }

    #[test]
    fn test_reduce_negative_axis() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_sum(&input, &[-1], false).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[6.0, 15.0]);
    }

    #[test]
    fn test_reduce_3d() {
        // 2x2x3 tensor, reduce axis 1
        let input = make_f32(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 2, 3],
        );
        let out = reduce_sum(&input, &[1], false).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 3]);
        assert_eq!(
            out.data().typed_data::<f32>(),
            &[5.0, 7.0, 9.0, 17.0, 19.0, 21.0]
        );
    }

    #[test]
    fn test_reduce_keepdims_all() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_sum(&input, &[0, 1], true).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1]);
        assert_eq!(out.data().typed_data::<f32>(), &[21.0]);
    }

    #[test]
    fn test_reduce_prod_axis1() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_prod(&input, &[1], false).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2]);
        let data = out.data().typed_data::<f32>();
        assert!((data[0] - 6.0).abs() < 1e-6); // 1*2*3
        assert!((data[1] - 120.0).abs() < 1e-4); // 4*5*6
    }

    #[test]
    fn test_reduce_prod_axis0() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_prod(&input, &[0], false).unwrap();
        let data = out.data().typed_data::<f32>();
        assert!((data[0] - 4.0).abs() < 1e-6); // 1*4
        assert!((data[1] - 10.0).abs() < 1e-5); // 2*5
        assert!((data[2] - 18.0).abs() < 1e-5); // 3*6
    }
}

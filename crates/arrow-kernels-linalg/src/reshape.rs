use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_kernels_common::{KernelError, Result};

/// Reshape a tensor to a new shape. Supports one -1 dimension (inferred).
///
/// Total elements must match. Data is copied with the new shape.
pub fn reshape<T>(
    input: &Tensor<'_, T>,
    new_shape: &[i64],
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let data: &[T::Native] = input.data().typed_data();
    let total = data.len();

    let resolved = resolve_shape(new_shape, total)?;
    let buf = Buffer::from_vec(data.to_vec());
    Tensor::new_row_major(buf, Some(resolved), None).map_err(KernelError::from)
}

/// Flatten a tensor from `axis` to the end into a single dimension.
///
/// E.g., shape [2, 3, 4] with axis=1 -> [2, 12].
/// Supports negative axis indexing.
pub fn flatten<T>(
    input: &Tensor<'_, T>,
    axis: i64,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("flatten: tensor has no shape".into()))?;
    let ndim = shape.len();
    let ax = resolve_axis(axis, ndim, "flatten")?;

    let mut new_shape: Vec<usize> = shape[..ax].to_vec();
    let tail: usize = shape[ax..].iter().product();
    new_shape.push(tail);

    let data: &[T::Native] = input.data().typed_data();
    let buf = Buffer::from_vec(data.to_vec());
    Tensor::new_row_major(buf, Some(new_shape), None).map_err(KernelError::from)
}

/// Remove dimensions of size 1.
///
/// If `axes` is None, removes all size-1 dims.
/// If `axes` is Some, removes only specified axes (which must be size 1).
pub fn squeeze<T>(
    input: &Tensor<'_, T>,
    axes: Option<&[i64]>,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("squeeze: tensor has no shape".into()))?;
    let ndim = shape.len();

    let new_shape: Vec<usize> = match axes {
        None => shape.iter().copied().filter(|&d| d != 1).collect(),
        Some(ax) => {
            let resolved: Vec<usize> = ax
                .iter()
                .map(|&a| resolve_axis(a, ndim, "squeeze"))
                .collect::<Result<_>>()?;
            for &r in &resolved {
                if shape[r] != 1 {
                    return Err(KernelError::InvalidArgument(format!(
                        "squeeze: dim[{r}] = {} is not 1",
                        shape[r]
                    )));
                }
            }
            shape
                .iter()
                .enumerate()
                .filter(|(i, _)| !resolved.contains(i))
                .map(|(_, &d)| d)
                .collect()
        }
    };

    // Tensor requires at least 1 dimension
    let new_shape = if new_shape.is_empty() {
        vec![1]
    } else {
        new_shape
    };

    let data: &[T::Native] = input.data().typed_data();
    let buf = Buffer::from_vec(data.to_vec());
    Tensor::new_row_major(buf, Some(new_shape), None).map_err(KernelError::from)
}

/// Insert dimensions of size 1 at the specified positions.
///
/// Axes refer to positions in the *output* shape. Supports negative indexing.
pub fn unsqueeze<T>(
    input: &Tensor<'_, T>,
    axes: &[i64],
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("unsqueeze: tensor has no shape".into()))?;
    let out_ndim = shape.len() + axes.len();

    let mut resolved: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a >= 0 {
                Ok(a as usize)
            } else {
                let pos = out_ndim as i64 + a;
                if pos < 0 {
                    Err(KernelError::InvalidArgument(format!(
                        "unsqueeze: axis {a} out of range for output ndim {out_ndim}"
                    )))
                } else {
                    Ok(pos as usize)
                }
            }
        })
        .collect::<Result<_>>()?;

    resolved.sort();
    // Check for duplicates
    for w in resolved.windows(2) {
        if w[0] == w[1] {
            return Err(KernelError::InvalidArgument(format!(
                "unsqueeze: duplicate axis {}",
                w[0]
            )));
        }
    }

    let mut new_shape = Vec::with_capacity(out_ndim);
    let mut in_idx = 0;
    for out_idx in 0..out_ndim {
        if resolved.contains(&out_idx) {
            new_shape.push(1);
        } else {
            new_shape.push(shape[in_idx]);
            in_idx += 1;
        }
    }

    let data: &[T::Native] = input.data().typed_data();
    let buf = Buffer::from_vec(data.to_vec());
    Tensor::new_row_major(buf, Some(new_shape), None).map_err(KernelError::from)
}

fn resolve_axis(axis: i64, ndim: usize, op: &str) -> Result<usize> {
    let nd = ndim as i64;
    let a = if axis < 0 { nd + axis } else { axis };
    if a < 0 || a >= nd {
        return Err(KernelError::InvalidArgument(format!(
            "{op}: axis {axis} out of range for {ndim}D tensor"
        )));
    }
    Ok(a as usize)
}

fn resolve_shape(new_shape: &[i64], total: usize) -> Result<Vec<usize>> {
    let mut neg_idx = None;
    let mut product = 1usize;

    for (i, &d) in new_shape.iter().enumerate() {
        if d == -1 {
            if neg_idx.is_some() {
                return Err(KernelError::InvalidArgument(
                    "reshape: at most one -1 dimension allowed".into(),
                ));
            }
            neg_idx = Some(i);
        } else if d < -1 {
            return Err(KernelError::InvalidArgument(format!(
                "reshape: invalid dimension {d}"
            )));
        } else {
            product *= d as usize;
        }
    }

    let mut resolved: Vec<usize> = new_shape.iter().map(|&d| d as usize).collect();

    if let Some(idx) = neg_idx {
        if product == 0 || total % product != 0 {
            return Err(KernelError::ShapeMismatch {
                operation: "reshape",
                expected: format!("total elements {total} divisible by {product}"),
                actual: format!("remainder {}", if product == 0 { total } else { total % product }),
            });
        }
        resolved[idx] = total / product;
    } else {
        let resolved_total: usize = resolved.iter().product();
        if resolved_total != total {
            return Err(KernelError::ShapeMismatch {
                operation: "reshape",
                expected: format!("{total} elements"),
                actual: format!("{resolved_total} elements"),
            });
        }
    }

    Ok(resolved)
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
    fn test_reshape_2d_to_1d() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reshape(&input, &[6]).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![6]);
    }

    #[test]
    fn test_reshape_1d_to_2d() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);
        let out = reshape(&input, &[2, 3]).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 3]);
    }

    #[test]
    fn test_reshape_with_neg1() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);
        let out = reshape(&input, &[2, -1]).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 3]);
    }

    #[test]
    fn test_reshape_incompatible() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);
        assert!(reshape(&input, &[4]).is_err());
    }

    #[test]
    fn test_flatten_axis0() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = flatten(&input, 0).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![6]);
    }

    #[test]
    fn test_flatten_axis1() {
        let input = make_f32((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]);
        let out = flatten(&input, 1).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 12]);
    }

    #[test]
    fn test_flatten_negative_axis() {
        let input = make_f32((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]);
        let out = flatten(&input, -2).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 12]);
    }

    #[test]
    fn test_squeeze_all() {
        let input = make_f32(vec![5.0], vec![1, 1, 1]);
        let out = squeeze(&input, None).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1]); // at least 1 dim
    }

    #[test]
    fn test_squeeze_specific() {
        let input = make_f32(vec![1.0, 2.0, 3.0], vec![1, 3, 1]);
        let out = squeeze(&input, Some(&[0, 2])).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![3]);
    }

    #[test]
    fn test_squeeze_not_size1() {
        let input = make_f32(vec![1.0, 2.0, 3.0], vec![1, 3]);
        assert!(squeeze(&input, Some(&[1])).is_err());
    }

    #[test]
    fn test_unsqueeze() {
        let input = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let out = unsqueeze(&input, &[0, 2]).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 3, 1]);
    }

    #[test]
    fn test_unsqueeze_negative() {
        let input = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let out = unsqueeze(&input, &[-1]).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![3, 1]);
    }

    #[test]
    fn test_unsqueeze_duplicate_error() {
        let input = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(unsqueeze(&input, &[0, 0]).is_err());
    }
}

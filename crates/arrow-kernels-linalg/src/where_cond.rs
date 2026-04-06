use arrow::array::{ArrowPrimitiveType, BooleanArray};
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_kernels_common::{KernelError, Result};

use crate::broadcast::broadcast_shapes;

/// Element-wise ternary: if condition[i] then x[i] else y[i].
///
/// x and y are broadcast to a common shape. Condition length must match the
/// total element count of the broadcast output.
pub fn where_cond<T>(
    condition: &BooleanArray,
    x: &Tensor<'_, T>,
    y: &Tensor<'_, T>,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let shape_x = x
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("where_cond: tensor x has no shape".into()))?;
    let shape_y = y
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("where_cond: tensor y has no shape".into()))?;

    let x_data: &[T::Native] = x.data().typed_data();
    let y_data: &[T::Native] = y.data().typed_data();

    // Fast path: same shape (original behavior)
    if shape_x == shape_y {
        let total: usize = shape_x.iter().product();
        if condition.len() != total {
            return Err(KernelError::ShapeMismatch {
                operation: "where_cond",
                expected: format!("condition length {total}"),
                actual: format!("condition length {}", condition.len()),
            });
        }

        let mut out = Vec::with_capacity(total);
        for i in 0..total {
            if condition.value(i) {
                out.push(x_data[i]);
            } else {
                out.push(y_data[i]);
            }
        }
        let buf = Buffer::from_vec(out);
        return Tensor::new_row_major(buf, Some(shape_x.to_vec()), None)
            .map_err(KernelError::from);
    }

    // Broadcast path
    let (out_shape, x_strides, y_strides) =
        broadcast_shapes(shape_x, shape_y, "where_cond")?;
    let total: usize = out_shape.iter().product();
    let ndim = out_shape.len();

    if condition.len() != total {
        return Err(KernelError::ShapeMismatch {
            operation: "where_cond",
            expected: format!("condition length {total}"),
            actual: format!("condition length {}", condition.len()),
        });
    }

    let mut out = Vec::with_capacity(total);
    let mut coords = vec![0usize; ndim];

    for i in 0..total {
        let mut x_flat = 0;
        let mut y_flat = 0;
        for d in 0..ndim {
            x_flat += coords[d] * x_strides[d];
            y_flat += coords[d] * y_strides[d];
        }

        if condition.value(i) {
            out.push(x_data[x_flat]);
        } else {
            out.push(y_data[y_flat]);
        }

        for d in (0..ndim).rev() {
            coords[d] += 1;
            if coords[d] < out_shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(out_shape), None).map_err(KernelError::from)
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
    fn test_where_basic() {
        let cond = BooleanArray::from(vec![true, false, true, false]);
        let x = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let y = make_f32(vec![10.0, 20.0, 30.0, 40.0], vec![4]);
        let out = where_cond::<Float32Type>(&cond, &x, &y).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[1.0, 20.0, 3.0, 40.0]);
    }

    #[test]
    fn test_where_all_true() {
        let cond = BooleanArray::from(vec![true, true, true]);
        let x = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let y = make_f32(vec![10.0, 20.0, 30.0], vec![3]);
        let out = where_cond::<Float32Type>(&cond, &x, &y).unwrap();
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_where_shape_mismatch() {
        let cond = BooleanArray::from(vec![true, false]);
        let x = make_f32(vec![1.0, 2.0], vec![2]);
        let y = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        // [2] and [3] are not broadcast-compatible
        assert!(where_cond::<Float32Type>(&cond, &x, &y).is_err());
    }

    #[test]
    fn test_where_preserves_shape() {
        let cond = BooleanArray::from(vec![true, false, true, false, true, false]);
        let x = make_f32(vec![1.0; 6], vec![2, 3]);
        let y = make_f32(vec![0.0; 6], vec![2, 3]);
        let out = where_cond::<Float32Type>(&cond, &x, &y).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 3]);
    }

    #[test]
    fn test_where_broadcast_scalar_y() {
        // x: [3], y: [1] (scalar broadcast)
        let cond = BooleanArray::from(vec![true, false, true]);
        let x = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let y = make_f32(vec![99.0], vec![1]);
        let out = where_cond::<Float32Type>(&cond, &x, &y).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[1.0, 99.0, 3.0]);
    }

    #[test]
    fn test_where_broadcast_row_col() {
        // x: [3,1], y: [1,3] -> output [3,3]
        let cond = BooleanArray::from(vec![
            true, false, false, false, true, false, false, false, true,
        ]);
        let x = make_f32(vec![10.0, 20.0, 30.0], vec![3, 1]);
        let y = make_f32(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let out = where_cond::<Float32Type>(&cond, &x, &y).unwrap();
        assert_eq!(out.shape().unwrap(), &[3, 3]);
        let data = out.data().typed_data::<f32>();
        assert_eq!(
            data,
            &[
                10.0, 2.0, 3.0, // row 0: cond=[T,F,F], x=10, y=[1,2,3]
                1.0, 20.0, 3.0, // row 1: cond=[F,T,F], x=20, y=[1,2,3]
                1.0, 2.0, 30.0, // row 2: cond=[F,F,T], x=30, y=[1,2,3]
            ]
        );
    }
}

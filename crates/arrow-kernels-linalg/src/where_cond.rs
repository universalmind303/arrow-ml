use arrow::array::{ArrowPrimitiveType, BooleanArray};
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_kernels_common::{KernelError, Result};

/// Element-wise ternary: if condition[i] then x[i] else y[i].
/// All inputs must have the same shape (flat length).
pub fn where_cond<T>(
    condition: &BooleanArray,
    x: &Tensor<'_, T>,
    y: &Tensor<'_, T>,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let shape = x
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("where_cond: tensor x has no shape".into()))?
        .to_vec();
    let shape_y = y
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("where_cond: tensor y has no shape".into()))?;

    if shape != *shape_y {
        return Err(KernelError::ShapeMismatch {
            operation: "where_cond",
            expected: format!("{:?}", shape),
            actual: format!("{:?}", shape_y),
        });
    }

    let total: usize = shape.iter().product();
    if condition.len() != total {
        return Err(KernelError::ShapeMismatch {
            operation: "where_cond",
            expected: format!("condition length {total}"),
            actual: format!("condition length {}", condition.len()),
        });
    }

    let x_data: &[T::Native] = x.data().typed_data();
    let y_data: &[T::Native] = y.data().typed_data();

    let mut out = Vec::with_capacity(total);
    for i in 0..total {
        if condition.value(i) {
            out.push(x_data[i]);
        } else {
            out.push(y_data[i]);
        }
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
}

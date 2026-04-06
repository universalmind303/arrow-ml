use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_kernels_common::{KernelError, Result};

/// Compute broadcast-compatible output shape from two input shapes.
///
/// Returns `(output_shape, a_strides, b_strides)` where strides are 0 for broadcast dims.
/// Follows NumPy broadcasting rules: dimensions are compared from the right,
/// a dimension of size 1 can broadcast to any size, missing dims are treated as 1.
pub fn broadcast_shapes(
    shape_a: &[usize],
    shape_b: &[usize],
    operation: &'static str,
) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let ndim = shape_a.len().max(shape_b.len());

    let mut padded_a = vec![1usize; ndim];
    let offset_a = ndim - shape_a.len();
    for (i, &d) in shape_a.iter().enumerate() {
        padded_a[offset_a + i] = d;
    }

    let mut padded_b = vec![1usize; ndim];
    let offset_b = ndim - shape_b.len();
    for (i, &d) in shape_b.iter().enumerate() {
        padded_b[offset_b + i] = d;
    }

    let mut out_shape = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let a = padded_a[i];
        let b = padded_b[i];
        if a == b {
            out_shape.push(a);
        } else if a == 1 {
            out_shape.push(b);
        } else if b == 1 {
            out_shape.push(a);
        } else {
            return Err(KernelError::ShapeMismatch {
                operation,
                expected: format!("{:?}", shape_a),
                actual: format!("{:?}", shape_b),
            });
        }
    }

    let a_strides = compute_broadcast_strides(&padded_a, ndim);
    let b_strides = compute_broadcast_strides(&padded_b, ndim);

    Ok((out_shape, a_strides, b_strides))
}

fn compute_broadcast_strides(padded_shape: &[usize], ndim: usize) -> Vec<usize> {
    let mut strides = vec![0usize; ndim];
    let mut stride = 1usize;
    for i in (0..ndim).rev() {
        if padded_shape[i] == 1 {
            strides[i] = 0;
        } else {
            strides[i] = stride;
        }
        stride *= padded_shape[i];
    }
    strides
}

/// Apply a binary operation element-wise with NumPy-style broadcasting.
///
/// Three type parameters allow the output type to differ from the inputs,
/// so the same function handles arithmetic (T,T -> T) and comparison (T,T -> UInt8).
pub fn broadcast_binary_op<A, B, C, F>(
    a: &Tensor<'_, A>,
    b: &Tensor<'_, B>,
    op: F,
    op_name: &'static str,
) -> Result<Tensor<'static, C>>
where
    A: ArrowPrimitiveType,
    A::Native: Copy,
    B: ArrowPrimitiveType,
    B::Native: Copy,
    C: ArrowPrimitiveType,
    C::Native: Copy,
    F: Fn(A::Native, B::Native) -> C::Native,
{
    let shape_a = a
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument(format!("{op_name}: tensor a has no shape")))?;
    let shape_b = b
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument(format!("{op_name}: tensor b has no shape")))?;

    let data_a: &[A::Native] = a.data().typed_data();
    let data_b: &[B::Native] = b.data().typed_data();

    // Fast path: same shape — flat zip, no stride math
    if shape_a == shape_b {
        let out: Vec<C::Native> = data_a
            .iter()
            .zip(data_b.iter())
            .map(|(&x, &y)| op(x, y))
            .collect();
        let buf = Buffer::from_vec(out);
        return Tensor::new_row_major(buf, Some(shape_a.to_vec()), None)
            .map_err(KernelError::from);
    }

    // Fast path: scalar broadcast (one input is a single element)
    if data_a.len() == 1 {
        let scalar = data_a[0];
        let out: Vec<C::Native> = data_b.iter().map(|&y| op(scalar, y)).collect();
        let buf = Buffer::from_vec(out);
        return Tensor::new_row_major(buf, Some(shape_b.to_vec()), None)
            .map_err(KernelError::from);
    }
    if data_b.len() == 1 {
        let scalar = data_b[0];
        let out: Vec<C::Native> = data_a.iter().map(|&x| op(x, scalar)).collect();
        let buf = Buffer::from_vec(out);
        return Tensor::new_row_major(buf, Some(shape_a.to_vec()), None)
            .map_err(KernelError::from);
    }

    // General path: strided coordinate iteration
    let (out_shape, a_strides, b_strides) = broadcast_shapes(shape_a, shape_b, op_name)?;
    let total: usize = out_shape.iter().product();
    let ndim = out_shape.len();

    let mut out = Vec::with_capacity(total);
    let mut coords = vec![0usize; ndim];

    for _ in 0..total {
        let mut a_flat = 0;
        let mut b_flat = 0;
        for d in 0..ndim {
            a_flat += coords[d] * a_strides[d];
            b_flat += coords[d] * b_strides[d];
        }
        out.push(op(data_a[a_flat], data_b[b_flat]));

        // Increment coords (rightmost first)
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
    use arrow::datatypes::{Float32Type, UInt8Type};

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_broadcast_shapes_same() {
        let (out, a_s, b_s) = broadcast_shapes(&[2, 3], &[2, 3], "test").unwrap();
        assert_eq!(out, vec![2, 3]);
        assert_eq!(a_s, vec![3, 1]);
        assert_eq!(b_s, vec![3, 1]);
    }

    #[test]
    fn test_broadcast_shapes_row_col() {
        let (out, a_s, b_s) = broadcast_shapes(&[3, 1], &[1, 4], "test").unwrap();
        assert_eq!(out, vec![3, 4]);
        assert_eq!(a_s, vec![1, 0]); // col: stride on dim0, broadcast on dim1
        assert_eq!(b_s, vec![0, 1]); // row: broadcast on dim0, stride on dim1
    }

    #[test]
    fn test_broadcast_shapes_rank_mismatch() {
        let (out, a_s, b_s) = broadcast_shapes(&[4], &[2, 3, 4], "test").unwrap();
        assert_eq!(out, vec![2, 3, 4]);
        // a gets padded to [1, 1, 4]
        assert_eq!(a_s, vec![0, 0, 1]);
        assert_eq!(b_s, vec![12, 4, 1]);
    }

    #[test]
    fn test_broadcast_shapes_incompatible() {
        assert!(broadcast_shapes(&[3], &[4], "test").is_err());
        assert!(broadcast_shapes(&[2, 3], &[2, 4], "test").is_err());
    }

    #[test]
    fn test_binary_op_same_shape() {
        let a = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let b = make_f32(vec![4.0, 5.0, 6.0], vec![3]);
        let out: Tensor<Float32Type> =
            broadcast_binary_op(&a, &b, |x, y| x + y, "add").unwrap();
        assert_eq!(out.shape().unwrap(), &[3]);
        assert_eq!(out.data().typed_data::<f32>(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_binary_op_scalar_broadcast() {
        let a = make_f32(vec![2.0], vec![1]);
        let b = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let out: Tensor<Float32Type> =
            broadcast_binary_op(&a, &b, |x, y| x * y, "mul").unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_binary_op_row_col_broadcast() {
        // [3,1] + [1,4] -> [3,4]
        let a = make_f32(vec![10.0, 20.0, 30.0], vec![3, 1]);
        let b = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let out: Tensor<Float32Type> =
            broadcast_binary_op(&a, &b, |x, y| x + y, "add").unwrap();
        assert_eq!(out.shape().unwrap(), &[3, 4]);
        let data = out.data().typed_data::<f32>();
        assert_eq!(
            data,
            &[
                11.0, 12.0, 13.0, 14.0, // 10 + [1,2,3,4]
                21.0, 22.0, 23.0, 24.0, // 20 + [1,2,3,4]
                31.0, 32.0, 33.0, 34.0, // 30 + [1,2,3,4]
            ]
        );
    }

    #[test]
    fn test_binary_op_rank_extension() {
        // [4] + [2,4] -> [2,4]
        let a = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = make_f32(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], vec![2, 4]);
        let out: Tensor<Float32Type> =
            broadcast_binary_op(&a, &b, |x, y| x + y, "add").unwrap();
        assert_eq!(out.shape().unwrap(), &[2, 4]);
        let data = out.data().typed_data::<f32>();
        assert_eq!(data, &[11.0, 22.0, 33.0, 44.0, 51.0, 62.0, 73.0, 84.0]);
    }

    #[test]
    fn test_binary_op_3d_broadcast() {
        // [1,3,1] + [2,1,4] -> [2,3,4]
        let a = make_f32(vec![1.0, 2.0, 3.0], vec![1, 3, 1]);
        let b = make_f32(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], vec![2, 1, 4]);
        let out: Tensor<Float32Type> =
            broadcast_binary_op(&a, &b, |x, y| x + y, "add").unwrap();
        assert_eq!(out.shape().unwrap(), &[2, 3, 4]);
        let data = out.data().typed_data::<f32>();
        // batch 0: b=[10,20,30,40]
        assert_eq!(&data[0..4], &[11.0, 21.0, 31.0, 41.0]); // a=1
        assert_eq!(&data[4..8], &[12.0, 22.0, 32.0, 42.0]); // a=2
        assert_eq!(&data[8..12], &[13.0, 23.0, 33.0, 43.0]); // a=3
        // batch 1: b=[50,60,70,80]
        assert_eq!(&data[12..16], &[51.0, 61.0, 71.0, 81.0]);
        assert_eq!(&data[16..20], &[52.0, 62.0, 72.0, 82.0]);
        assert_eq!(&data[20..24], &[53.0, 63.0, 73.0, 83.0]);
    }

    #[test]
    fn test_binary_op_incompatible() {
        let a = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let b = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let result: std::result::Result<Tensor<Float32Type>, _> =
            broadcast_binary_op(&a, &b, |x, y| x + y, "add");
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_op_cross_type() {
        // Comparison: f32 x f32 -> u8
        let a = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let b = make_f32(vec![2.0, 2.0, 2.0], vec![3]);
        let out: Tensor<UInt8Type> = broadcast_binary_op(
            &a,
            &b,
            |x, y| if x > y { 1u8 } else { 0u8 },
            "greater",
        )
        .unwrap();
        assert_eq!(out.data().typed_data::<u8>(), &[0, 0, 1]);
    }
}

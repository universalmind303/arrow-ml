use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_kernels_common::{KernelError, Result};

/// N-D constant-value padding following ONNX Pad semantics.
///
/// `pads` format (ONNX convention): [begin_0, begin_1, ..., end_0, end_1, ...]
pub fn pad<T>(
    input: &Tensor<'_, T>,
    pads: &[usize],
    constant_value: T::Native,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let in_shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("pad: tensor has no shape".into()))?;
    let ndims = in_shape.len();

    if pads.len() != 2 * ndims {
        return Err(KernelError::InvalidArgument(format!(
            "pad: pads length must be 2 * ndims ({}), got {}",
            2 * ndims,
            pads.len()
        )));
    }

    let mut out_shape = Vec::with_capacity(ndims);
    for i in 0..ndims {
        out_shape.push(pads[i] + in_shape[i] + pads[ndims + i]);
    }

    let out_len: usize = out_shape.iter().product();
    let mut out = vec![constant_value; out_len];

    let data: &[T::Native] = input.data().typed_data();
    let in_len: usize = in_shape.iter().product();

    let in_strides = row_major_strides(in_shape);
    let out_strides = row_major_strides(&out_shape);

    let mut nd_index = vec![0usize; ndims];
    for flat_in in 0..in_len {
        flat_to_nd(flat_in, &in_strides, &mut nd_index);

        let mut flat_out = 0;
        for d in 0..ndims {
            flat_out += (nd_index[d] + pads[d]) * out_strides[d];
        }

        out[flat_out] = data[flat_in];
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(out_shape), None).map_err(KernelError::from)
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let ndims = shape.len();
    let mut strides = vec![1usize; ndims];
    for i in (0..ndims.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn flat_to_nd(mut flat: usize, strides: &[usize], out: &mut [usize]) {
    for (d, &s) in strides.iter().enumerate() {
        out[d] = flat / s;
        flat %= s;
    }
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
    fn test_pad_1d_symmetric() {
        let input = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let result = pad::<Float32Type>(&input, &[2, 2], 0.0).unwrap();
        assert_eq!(result.shape().unwrap(), &[7]);
        assert_eq!(
            result.data().typed_data::<f32>(),
            &[0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_pad_1d_asymmetric() {
        let input = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let result = pad::<Float32Type>(&input, &[1, 3], 0.0).unwrap();
        assert_eq!(
            result.data().typed_data::<f32>(),
            &[0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_pad_2d_symmetric() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = pad::<Float32Type>(&input, &[1, 1, 1, 1], 0.0).unwrap();
        assert_eq!(result.shape().unwrap(), &[4, 5]);
        #[rustfmt::skip]
        let expected = [
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 2.0, 3.0, 0.0,
            0.0, 4.0, 5.0, 6.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(result.data().typed_data::<f32>(), &expected);
    }

    #[test]
    fn test_pad_nonzero_constant() {
        let input = make_f32(vec![5.0, 6.0], vec![2]);
        let result = pad::<Float32Type>(&input, &[1, 1], -1.0).unwrap();
        assert_eq!(result.data().typed_data::<f32>(), &[-1.0, 5.0, 6.0, -1.0]);
    }

    #[test]
    fn test_pad_no_padding() {
        let input = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let result = pad::<Float32Type>(&input, &[0, 0], 0.0).unwrap();
        assert_eq!(result.data().typed_data::<f32>(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_pad_invalid_pads_length() {
        let input = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(pad::<Float32Type>(&input, &[1, 1, 1, 1], 0.0).is_err());
    }
}

use arrow::array::{ArrowPrimitiveType, UInt32Array};
use arrow::buffer::Buffer;
use arrow::tensor::Tensor;
use arrow_ml_common::KernelError;
use arrow_ml_common::Result;

/// Embedding lookup: gather rows from a weight matrix by index.
///
/// `weights`: 2D tensor of shape (vocab_size, embed_dim) — the embedding table.
/// `indices`: 1D array of token IDs (u32).
///
/// Returns a 2D tensor of shape (len(indices), embed_dim).
pub fn embedding<T: ArrowPrimitiveType>(
    weights: &Tensor<'_, T>,
    indices: &UInt32Array,
) -> Result<Tensor<'static, T>>
where
    T::Native: Copy,
{
    let shape = weights
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("embedding: weights has no shape".into()))?;
    if shape.len() != 2 {
        return Err(KernelError::InvalidArgument(format!(
            "embedding: expected 2D weight tensor, got {}D",
            shape.len()
        )));
    }
    let (vocab_size, embed_dim) = (shape[0], shape[1]);
    let data: &[T::Native] = weights.data().typed_data();
    let seq_len = indices.len();

    let mut out = Vec::with_capacity(seq_len * embed_dim);

    for i in 0..seq_len {
        let idx = indices.value(i) as usize;
        if idx >= vocab_size {
            return Err(KernelError::InvalidArgument(format!(
                "embedding: index {idx} out of range for vocab_size {vocab_size}"
            )));
        }
        let row_start = idx * embed_dim;
        out.extend_from_slice(&data[row_start..row_start + embed_dim]);
    }

    let buf = Buffer::from_vec(out);
    Tensor::new_row_major(buf, Some(vec![seq_len, embed_dim]), None).map_err(KernelError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::Float32Type;

    fn make_f32_2d(data: Vec<f32>, rows: usize, cols: usize) -> Tensor<'static, Float32Type> {
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(vec![rows, cols]), None).unwrap()
    }

    #[test]
    fn test_embedding_basic() {
        // 4-word vocab, 3-dim embeddings
        #[rustfmt::skip]
        let weights = make_f32_2d(vec![
            0.1, 0.2, 0.3,   // word 0
            1.1, 1.2, 1.3,   // word 1
            2.1, 2.2, 2.3,   // word 2
            3.1, 3.2, 3.3,   // word 3
        ], 4, 3);

        let indices = UInt32Array::from(vec![2, 0, 3]);
        let out = embedding(&weights, &indices).unwrap();

        assert_eq!(out.shape().unwrap(), &vec![3, 3]);
        let data = out.data().typed_data::<f32>();
        // row 0 = word 2
        assert!((data[0] - 2.1).abs() < 1e-6);
        assert!((data[1] - 2.2).abs() < 1e-6);
        assert!((data[2] - 2.3).abs() < 1e-6);
        // row 1 = word 0
        assert!((data[3] - 0.1).abs() < 1e-6);
        // row 2 = word 3
        assert!((data[6] - 3.1).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_out_of_range() {
        let weights = make_f32_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let indices = UInt32Array::from(vec![5]); // out of range
        assert!(embedding(&weights, &indices).is_err());
    }

    #[test]
    fn test_embedding_repeated() {
        let weights = make_f32_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let indices = UInt32Array::from(vec![0, 0, 1, 0]);
        let out = embedding(&weights, &indices).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![4, 2]);
        let data = out.data().typed_data::<f32>();
        assert_eq!(&data[0..2], &[1.0, 2.0]);
        assert_eq!(&data[2..4], &[1.0, 2.0]);
        assert_eq!(&data[4..6], &[3.0, 4.0]);
        assert_eq!(&data[6..8], &[1.0, 2.0]);
    }
}

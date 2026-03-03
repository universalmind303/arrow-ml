mod kernel_f32;
mod kernel_f64;
mod packing;

use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::datatypes::DataType;
use arrow::tensor::Tensor;
use arrow_kernels_common::BackendRegistry;
use arrow_kernels_common::KernelError;
use arrow_kernels_common::Result;
use num_traits::Zero;
use std::ops::{AddAssign, Mul};

/// Casts a Tensor's buffer to a typed slice.
pub(crate) fn tensor_as_slice<'a, T: ArrowPrimitiveType>(
    tensor: &'a Tensor<'_, T>,
) -> &'a [T::Native] {
    let buf = tensor.data();
    let ptr = buf.as_ptr() as *const T::Native;
    let len = buf.len() / std::mem::size_of::<T::Native>();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// Extracts 2D shape (rows, cols) from a Tensor, returning an error if not 2D.
fn shape_2d<T: ArrowPrimitiveType>(
    tensor: &Tensor<'_, T>,
    operation: &'static str,
) -> Result<(usize, usize)> {
    let shape = tensor
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument(format!("{operation}: tensor has no shape")))?;
    if shape.len() != 2 {
        return Err(KernelError::InvalidArgument(format!(
            "{operation}: expected 2D tensor, got {ndim}D",
            ndim = shape.len()
        )));
    }
    Ok((shape[0], shape[1]))
}

/// Minimum dimension to use SIMD path (below this, packing overhead dominates).
const SIMD_THRESHOLD: usize = 128;

/// Minimum dimension to use a GPU backend (below this, dispatch overhead dominates).
const GPU_THRESHOLD: usize = 256;

// ---------------------------------------------------------------------------
// Private f32 implementation
// ---------------------------------------------------------------------------

fn naive_matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
    c
}

fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // Try GPU backend for large matrices
    if m >= GPU_THRESHOLD || n >= GPU_THRESHOLD || k >= GPU_THRESHOLD {
        let registry = BackendRegistry::global();
        let mut c = vec![0.0f32; m * n];
        if registry.try_matmul_f32(a, b, &mut c, m, k, n).is_some() {
            return c;
        }
    }

    if m >= SIMD_THRESHOLD || n >= SIMD_THRESHOLD || k >= SIMD_THRESHOLD {
        kernel_f32::gemm(a, b, m, k, n)
    } else {
        naive_matmul_f32(a, b, m, k, n)
    }
}

fn matvec_f32(a: &[f32], x: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; m];
    for i in 0..m {
        let mut sum = 0.0f32;
        for j in 0..n {
            sum += a[i * n + j] * x[j];
        }
        result[i] = sum;
    }
    result
}

// ---------------------------------------------------------------------------
// Private f64 implementation
// ---------------------------------------------------------------------------

fn naive_matmul_f64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
    c
}

fn matmul_f64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    // Try GPU backend for large matrices
    if m >= GPU_THRESHOLD || n >= GPU_THRESHOLD || k >= GPU_THRESHOLD {
        let registry = BackendRegistry::global();
        let mut c = vec![0.0f64; m * n];
        if registry.try_matmul_f64(a, b, &mut c, m, k, n).is_some() {
            return c;
        }
    }

    if m >= SIMD_THRESHOLD || n >= SIMD_THRESHOLD || k >= SIMD_THRESHOLD {
        kernel_f64::gemm(a, b, m, k, n)
    } else {
        naive_matmul_f64(a, b, m, k, n)
    }
}

fn matvec_f64(a: &[f64], x: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut result = vec![0.0f64; m];
    for i in 0..m {
        let mut sum = 0.0f64;
        for j in 0..n {
            sum += a[i * n + j] * x[j];
        }
        result[i] = sum;
    }
    result
}

// ---------------------------------------------------------------------------
// Generic naive implementations (all other numeric types)
// ---------------------------------------------------------------------------

fn naive_matmul_generic<N>(a: &[N], b: &[N], m: usize, k: usize, n: usize) -> Vec<N>
where
    N: Zero + Copy + Mul<Output = N> + AddAssign,
{
    let mut c = vec![N::zero(); m * n];
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
    c
}

fn naive_matvec_generic<N>(a: &[N], x: &[N], m: usize, n: usize) -> Vec<N>
where
    N: Zero + Copy + Mul<Output = N> + AddAssign,
{
    let mut result = vec![N::zero(); m];
    for i in 0..m {
        let mut sum = N::zero();
        for j in 0..n {
            sum += a[i * n + j] * x[j];
        }
        result[i] = sum;
    }
    result
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Validates matmul shapes and returns (m, k, n).
fn validate_matmul_shapes<T: ArrowPrimitiveType>(
    a: &Tensor<'_, T>,
    b: &Tensor<'_, T>,
) -> Result<(usize, usize, usize)> {
    let (m, k1) = shape_2d(a, "matmul")?;
    let (k2, n) = shape_2d(b, "matmul")?;
    if k1 != k2 {
        return Err(KernelError::ShapeMismatch {
            operation: "matmul",
            expected: format!("A columns ({k1}) == B rows"),
            actual: format!("B rows = {k2}"),
        });
    }
    Ok((m, k1, n))
}

/// Wraps raw bytes into a `Buffer` and constructs a row-major 2D `Tensor<T>`.
fn buf_to_tensor_2d<T: ArrowPrimitiveType>(
    buf: Buffer,
    m: usize,
    n: usize,
) -> Result<Tensor<'static, T>> {
    Tensor::new_row_major(buf, Some(vec![m, n]), None).map_err(KernelError::from)
}

/// Wraps raw bytes into a `Buffer` and constructs a row-major 1D `Tensor<T>`.
fn buf_to_tensor_1d<T: ArrowPrimitiveType>(buf: Buffer, len: usize) -> Result<Tensor<'static, T>> {
    Tensor::new_row_major(buf, Some(vec![len]), None).map_err(KernelError::from)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Matrix multiplication: C = A * B.
///
/// A must be (m x k), B must be (k x n). Both must be 2D row-major tensors.
/// Returns a row-major (m x n) tensor.
///
/// Supports `Float32` and `Float64` data types. Uses a tiled + packed + SIMD
/// micro-kernel for large matrices, falls back to a naive loop for small ones.
pub fn matmul<T: ArrowPrimitiveType>(
    a: &Tensor<'_, T>,
    b: &Tensor<'_, T>,
) -> Result<Tensor<'static, T>>
where
    T::Native: Zero + Copy + Mul<Output = T::Native> + AddAssign,
{
    let (m, k, n) = validate_matmul_shapes(a, b)?;

    let result_buf = match T::DATA_TYPE {
        DataType::Float32 => {
            let result = matmul_f32(a.data().typed_data(), b.data().typed_data(), m, k, n);
            Buffer::from_vec(result)
        }
        DataType::Float64 => {
            let result = matmul_f64(a.data().typed_data(), b.data().typed_data(), m, k, n);
            Buffer::from_vec(result)
        }
        _ => {
            let a_slice: &[T::Native] = a.data().typed_data();
            let b_slice: &[T::Native] = b.data().typed_data();
            let result = naive_matmul_generic(a_slice, b_slice, m, k, n);
            Buffer::from_vec(result)
        }
    };

    buf_to_tensor_2d::<T>(result_buf, m, n)
}

/// Matrix-vector multiplication: y = A * x.
///
/// A must be (m x n), x must be a 1D tensor of length n.
/// Returns a 1D tensor of length m.
///
/// Supports `Float32` and `Float64` data types.
pub fn matvec<T: ArrowPrimitiveType>(
    a: &Tensor<'_, T>,
    x: &Tensor<'_, T>,
) -> Result<Tensor<'static, T>>
where
    T::Native: Zero + Copy + Mul<Output = T::Native> + AddAssign,
{
    let (m, n) = shape_2d(a, "matvec")?;

    let x_shape = x
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("matvec: vector has no shape".to_string()))?;
    if x_shape.len() != 1 {
        return Err(KernelError::InvalidArgument(format!(
            "matvec: expected 1D vector, got {}D",
            x_shape.len()
        )));
    }
    if x_shape[0] != n {
        return Err(KernelError::ShapeMismatch {
            operation: "matvec",
            expected: format!("vector length {n}"),
            actual: format!("length {}", x_shape[0]),
        });
    }

    let a_buf = a.data();
    let x_buf = x.data();

    let result_buf = match T::DATA_TYPE {
        DataType::Float32 => {
            let result = matvec_f32(a_buf.typed_data(), x_buf.typed_data(), m, n);
            Buffer::from_vec(result)
        }
        DataType::Float64 => {
            let result = matvec_f64(a_buf.typed_data(), x_buf.typed_data(), m, n);
            Buffer::from_vec(result)
        }
        _ => {
            let result = naive_matvec_generic::<T::Native>(a_buf.typed_data(), x_buf.typed_data(), m, n);
            Buffer::from_vec(result)
        }
    };

    buf_to_tensor_1d::<T>(result_buf, m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::{Float32Type, Float64Type};

    fn make_f32_2d(data: Vec<f32>, rows: usize, cols: usize) -> Tensor<'static, Float32Type> {
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(vec![rows, cols]), None).unwrap()
    }

    fn make_f64_2d(data: Vec<f64>, rows: usize, cols: usize) -> Tensor<'static, Float64Type> {
        let buffer = Buffer::from(ScalarBuffer::<f64>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(vec![rows, cols]), None).unwrap()
    }

    fn make_f32_1d(data: Vec<f32>) -> Tensor<'static, Float32Type> {
        let len = data.len();
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(vec![len]), None).unwrap()
    }

    /// Reference naive matmul for verification.
    fn reference_matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let a_val = a[i * k + p];
                for j in 0..n {
                    c[i * n + j] += a_val * b[p * n + j];
                }
            }
        }
        c
    }

    // ---- f32 matmul tests ----

    #[test]
    fn test_matmul_2x3_times_3x2() {
        let a = make_f32_2d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = make_f32_2d(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
        let c = matmul(&a, &b).unwrap();

        assert_eq!(c.shape().unwrap(), &vec![2, 2]);
        let data = c.data().typed_data::<f32>();
        assert!((data[0] - 58.0).abs() < 1e-5);
        assert!((data[1] - 64.0).abs() < 1e-5);
        assert!((data[2] - 139.0).abs() < 1e-5);
        assert!((data[3] - 154.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let a = make_f32_2d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = make_f32_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn test_matmul_identity() {
        let identity = make_f32_2d(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let a = make_f32_2d(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = matmul(&identity, &a).unwrap();
        let data = c.data().typed_data::<f32>();
        assert!((data[0] - 5.0).abs() < 1e-5);
        assert!((data[1] - 6.0).abs() < 1e-5);
        assert!((data[2] - 7.0).abs() < 1e-5);
        assert!((data[3] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_1x1() {
        let a = make_f32_2d(vec![3.0], 1, 1);
        let b = make_f32_2d(vec![7.0], 1, 1);
        let c = matmul(&a, &b).unwrap();
        let data = c.data().typed_data::<f32>();
        assert!((data[0] - 21.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_64x64() {
        let (m, k, n) = (64, 64, 64);
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let a = make_f32_2d(a_data.clone(), m, k);
        let b = make_f32_2d(b_data.clone(), k, n);
        let c = matmul(&a, &b).unwrap();

        let c_ref = reference_matmul_f32(&a_data, &b_data, m, k, n);
        let c_data = c.data().typed_data::<f32>();
        for i in 0..m * n {
            assert!(
                (c_data[i] - c_ref[i]).abs() < 1e-1,
                "mismatch at {i}: got {} expected {}",
                c_data[i],
                c_ref[i]
            );
        }
    }

    #[test]
    fn test_matmul_non_aligned() {
        let (m, k, n) = (7, 13, 11);
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let a = make_f32_2d(a_data.clone(), m, k);
        let b = make_f32_2d(b_data.clone(), k, n);
        let c = matmul(&a, &b).unwrap();

        let c_ref = reference_matmul_f32(&a_data, &b_data, m, k, n);
        let c_data = c.data().typed_data::<f32>();
        for i in 0..m * n {
            assert!(
                (c_data[i] - c_ref[i]).abs() < 1e-2,
                "mismatch at {i}: got {} expected {}",
                c_data[i],
                c_ref[i]
            );
        }
    }

    #[test]
    fn test_matmul_256x256() {
        let (m, k, n) = (256, 256, 256);
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 100) as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 100) as f32) * 0.01).collect();
        let a = make_f32_2d(a_data.clone(), m, k);
        let b = make_f32_2d(b_data.clone(), k, n);
        let c = matmul(&a, &b).unwrap();

        let c_ref = reference_matmul_f32(&a_data, &b_data, m, k, n);
        let c_data = c.data().typed_data::<f32>();
        for i in 0..m * n {
            assert!(
                (c_data[i] - c_ref[i]).abs() < 0.5,
                "mismatch at {i}: got {} expected {}",
                c_data[i],
                c_ref[i]
            );
        }
    }

    // ---- f64 matmul tests ----

    #[test]
    fn test_matmul_f64_small() {
        let a = make_f64_2d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = make_f64_2d(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
        let c = matmul(&a, &b).unwrap();

        assert_eq!(c.shape().unwrap(), &vec![2, 2]);
        let data = c.data().typed_data::<f64>();
        assert!((data[0] - 58.0).abs() < 1e-10);
        assert!((data[1] - 64.0).abs() < 1e-10);
        assert!((data[2] - 139.0).abs() < 1e-10);
        assert!((data[3] - 154.0).abs() < 1e-10);
    }

    #[test]
    fn test_matmul_f64_16x16() {
        let (m, k, n) = (16, 16, 16);
        let a_data: Vec<f64> = (0..m * k).map(|i| (i as f64) * 0.01).collect();
        let b_data: Vec<f64> = (0..k * n).map(|i| (i as f64) * 0.01).collect();
        let a = make_f64_2d(a_data.clone(), m, k);
        let b = make_f64_2d(b_data.clone(), k, n);
        let c = matmul(&a, &b).unwrap();

        let mut c_ref = vec![0.0f64; m * n];
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    c_ref[i * n + j] += a_data[i * k + p] * b_data[p * n + j];
                }
            }
        }
        let c_data = c.data().typed_data::<f64>();
        for i in 0..m * n {
            assert!(
                (c_data[i] - c_ref[i]).abs() < 1e-6,
                "f64 mismatch at {i}: got {} expected {}",
                c_data[i],
                c_ref[i]
            );
        }
    }

    // ---- matvec tests ----

    #[test]
    fn test_matvec_f32() {
        let a = make_f32_2d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let x = make_f32_1d(vec![1.0, 1.0, 1.0]);
        let y = matvec(&a, &x).unwrap();

        assert_eq!(y.shape().unwrap(), &vec![2]);
        let data = tensor_as_slice(&y);
        assert!((data[0] - 6.0).abs() < 1e-5);
        assert!((data[1] - 15.0).abs() < 1e-5);
    }

    // ---- integer type matmul tests ----

    fn make_tensor_2d<T: ArrowPrimitiveType>(
        data: Vec<T::Native>,
        rows: usize,
        cols: usize,
    ) -> Tensor<'static, T> {
        let buffer = Buffer::from_vec(data);
        Tensor::new_row_major(buffer, Some(vec![rows, cols]), None).unwrap()
    }

    fn make_tensor_1d<T: ArrowPrimitiveType>(data: Vec<T::Native>) -> Tensor<'static, T> {
        let len = data.len();
        let buffer = Buffer::from_vec(data);
        Tensor::new_row_major(buffer, Some(vec![len]), None).unwrap()
    }

    #[test]
    fn test_matmul_i32() {
        use arrow::datatypes::Int32Type;
        // [1 2 3] * [7  8 ]   [58  64 ]
        // [4 5 6]   [9  10] = [139 154]
        //            [11 12]
        let a = make_tensor_2d::<Int32Type>(vec![1, 2, 3, 4, 5, 6], 2, 3);
        let b = make_tensor_2d::<Int32Type>(vec![7, 8, 9, 10, 11, 12], 3, 2);
        let c = matmul(&a, &b).unwrap();

        assert_eq!(c.shape().unwrap(), &vec![2, 2]);
        let data: &[i32] = c.data().typed_data();
        assert_eq!(data, &[58, 64, 139, 154]);
    }

    #[test]
    fn test_matmul_u8() {
        use arrow::datatypes::UInt8Type;
        // Small values to avoid overflow: [1 2] * [3 4] = [5  8 ]
        //                                 [0 1]   [1 2]   [1  2 ]
        let a = make_tensor_2d::<UInt8Type>(vec![1, 2, 0, 1], 2, 2);
        let b = make_tensor_2d::<UInt8Type>(vec![3, 4, 1, 2], 2, 2);
        let c = matmul(&a, &b).unwrap();

        let data: &[u8] = c.data().typed_data();
        assert_eq!(data, &[5, 8, 1, 2]);
    }

    #[test]
    fn test_matmul_i16() {
        use arrow::datatypes::Int16Type;
        let a = make_tensor_2d::<Int16Type>(vec![1, 2, 3, 4, 5, 6], 2, 3);
        let b = make_tensor_2d::<Int16Type>(vec![7, 8, 9, 10, 11, 12], 3, 2);
        let c = matmul(&a, &b).unwrap();

        let data: &[i16] = c.data().typed_data();
        assert_eq!(data, &[58, 64, 139, 154]);
    }

    #[test]
    fn test_matvec_i32() {
        use arrow::datatypes::Int32Type;
        let a = make_tensor_2d::<Int32Type>(vec![1, 2, 3, 4, 5, 6], 2, 3);
        let x = make_tensor_1d::<Int32Type>(vec![1, 1, 1]);
        let y = matvec(&a, &x).unwrap();

        assert_eq!(y.shape().unwrap(), &vec![2]);
        let data: &[i32] = y.data().typed_data();
        assert_eq!(data, &[6, 15]);
    }
}

mod host_tensor;
mod kernel_f32;
mod kernel_f64;
mod packing;

use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::datatypes::DataType;
use arrow::tensor::Tensor;
use arrow_ml_common::backend::Backend;
use arrow_ml_common::device_tensor::{dtype, AmDeviceType};
use arrow_ml_common::kernels::matmul::MatmulKernel;
use arrow_ml_common::BackendRegistry;
use arrow_ml_common::KernelError;
use arrow_ml_common::Result;
use host_tensor::OwnedHostTensor;
use num_traits::{One, Zero};
use std::cell::RefCell;
use std::ops::{Add, AddAssign, Mul};

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
    // Try GPU backend for large matrices.
    if m >= GPU_THRESHOLD || n >= GPU_THRESHOLD || k >= GPU_THRESHOLD {
        if let Some(c) = try_backend_matmul_f32(a, b, m, k, n) {
            return c;
        }
    }

    if m >= SIMD_THRESHOLD || n >= SIMD_THRESHOLD || k >= SIMD_THRESHOLD {
        kernel_f32::gemm(a, b, m, k, n)
    } else {
        naive_matmul_f32(a, b, m, k, n)
    }
}

/// Attempt the matmul through the highest-priority backend that supports
/// f32 on the CPU device. Returns `None` if no backend supports it or the
/// backend errored out — caller is expected to fall through to SIMD.
///
/// Uses a thread-local handle cache so the kernel is opened once per thread
/// and amortized across many invocations.
fn try_backend_matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Option<Vec<f32>> {
    let backend: &'static Backend = BackendRegistry::global().best_matmul()?;
    let ops = backend.matmul.as_ref()?;
    if !ops.supports_dtype(dtype::FLOAT32, AmDeviceType::Cpu as i32) {
        return None;
    }

    MATMUL_F32_KERNEL.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            match MatmulKernel::open(backend, dtype::FLOAT32, AmDeviceType::Cpu as i32) {
                Ok(k) => *slot = Some(k),
                Err(_) => return None,
            }
        }
        let kernel = slot.as_ref().expect("just-initialized");

        let a_tensor = OwnedHostTensor::from_f32_slice(a, m, k);
        let b_tensor = OwnedHostTensor::from_f32_slice(b, k, n);
        let mut c_tensor = OwnedHostTensor::from_f32_vec(vec![0.0f32; m * n], m, n);

        let invoke_result = unsafe {
            kernel.invoke(a_tensor.as_ffi(), b_tensor.as_ffi(), c_tensor.as_ffi_mut())
        };
        if invoke_result.is_err() {
            return None;
        }

        Some(c_tensor.as_f32_slice().to_vec())
    })
}

thread_local! {
    /// Per-thread cached `MatmulKernel` handle for f32 on the CPU device.
    /// Opened lazily on first use, never closed (lives until thread exit,
    /// at which point `Drop` fires `am_matmul_close`).
    static MATMUL_F32_KERNEL: RefCell<Option<MatmulKernel<'static>>> = const { RefCell::new(None) };
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

fn naive_gemm_64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
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

fn gemm_f64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    // GPU branch: probe for f64 support before paying the open/dispatch cost.
    // The Metal backend reports `supports_dtype(FLOAT64, _) == 0` so on
    // macOS this short-circuits straight to SIMD with no open call.
    if m >= GPU_THRESHOLD || n >= GPU_THRESHOLD || k >= GPU_THRESHOLD {
        if let Some(backend) = BackendRegistry::global().best_matmul() {
            if let Some(ops) = backend.matmul.as_ref() {
                if ops.supports_dtype(dtype::FLOAT64, AmDeviceType::Cpu as i32) {
                    // No backend currently exports f64 matmul; this branch
                    // is plumbing for when one does. Skip silently —
                    // implementation lands with the first f64-capable
                    // backend.
                    let _ = (a, b, m, k, n);
                }
            }
        }
    }

    if m >= SIMD_THRESHOLD || n >= SIMD_THRESHOLD || k >= SIMD_THRESHOLD {
        kernel_f64::gemm(a, b, m, k, n)
    } else {
        naive_gemm_64(a, b, m, k, n)
    }
}

fn gemv_f64(a: &[f64], x: &[f64], m: usize, n: usize) -> Vec<f64> {
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

fn naive_gemm_generic<N>(a: &[N], b: &[N], m: usize, k: usize, n: usize) -> Vec<N>
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

fn naive_gemv_generic<N>(a: &[N], x: &[N], m: usize, n: usize) -> Vec<N>
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

/// Applies α/β scaling in-place: out[i] = alpha * out[i] + beta * existing[i].
/// No-op when alpha == 1 and existing is None (the common matmul/matvec case).
fn apply_alpha_beta<N: One + Copy + Mul<Output = N> + Add<Output = N> + PartialEq>(
    out: &mut [N],
    alpha: N,
    beta: N,
    existing: Option<&[N]>,
) {
    if alpha == N::one() && existing.is_none() {
        return;
    }
    for i in 0..out.len() {
        let mut val = alpha * out[i];
        if let Some(c_data) = existing {
            val = val + beta * c_data[i];
        }
        out[i] = val;
    }
}

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

/// General matrix multiplication: C = α * A * B + β * C.
///
/// A must be (m x k), B must be (k x n). Both must be 2D row-major tensors.
/// If `c` is provided, it must be (m x n) and is scaled by `beta` then accumulated into.
/// If `c` is `None`, `beta` is ignored and the result is just α * A * B.
///
/// Returns a row-major (m x n) tensor.
///
/// Uses optimized SIMD/GPU paths for f32/f64, naive fallback for other types.
pub fn gemm<T: ArrowPrimitiveType>(
    alpha: T::Native,
    a: &Tensor<'_, T>,
    b: &Tensor<'_, T>,
    beta: T::Native,
    c: Option<&Tensor<'_, T>>,
) -> Result<Tensor<'static, T>>
where
    T::Native: Zero + One + Copy + Mul<Output = T::Native> + Add<Output = T::Native> + AddAssign,
{
    let (m, k, n) = validate_matmul_shapes(a, b)?;

    if let Some(c_tensor) = c {
        let (cm, cn) = shape_2d(c_tensor, "gemm")?;
        if cm != m || cn != n {
            return Err(KernelError::ShapeMismatch {
                operation: "gemm",
                expected: format!("C shape ({m}, {n})"),
                actual: format!("({cm}, {cn})"),
            });
        }
    }

    // Compute A*B, then apply alpha/beta scaling.
    // Each arm uses concrete types to avoid associated-type unification issues.
    let result_buf = match T::DATA_TYPE {
        DataType::Float32 => {
            let mut ab = matmul_f32(a.data().typed_data(), b.data().typed_data(), m, k, n);
            let c_f32: Option<&[f32]> = c.map(|t| t.data().typed_data());
            // SAFETY: T::DATA_TYPE == Float32 guarantees T::Native == f32
            let alpha_f32: f32 = unsafe { *(&alpha as *const T::Native as *const f32) };
            let beta_f32: f32 = unsafe { *(&beta as *const T::Native as *const f32) };
            apply_alpha_beta(&mut ab, alpha_f32, beta_f32, c_f32);
            Buffer::from_vec(ab)
        }
        DataType::Float64 => {
            let mut ab = gemm_f64(a.data().typed_data(), b.data().typed_data(), m, k, n);
            let c_f64: Option<&[f64]> = c.map(|t| t.data().typed_data());
            let alpha_f64: f64 = unsafe { *(&alpha as *const T::Native as *const f64) };
            let beta_f64: f64 = unsafe { *(&beta as *const T::Native as *const f64) };
            apply_alpha_beta(&mut ab, alpha_f64, beta_f64, c_f64);
            Buffer::from_vec(ab)
        }
        _ => {
            let mut ab: Vec<T::Native> =
                naive_gemm_generic(a.data().typed_data(), b.data().typed_data(), m, k, n);
            let c_slice: Option<&[T::Native]> = c.map(|t| t.data().typed_data());
            apply_alpha_beta(&mut ab, alpha, beta, c_slice);
            Buffer::from_vec(ab)
        }
    };

    buf_to_tensor_2d::<T>(result_buf, m, n)
}

/// Matrix multiplication: C = A * B.
///
/// Convenience wrapper around [`gemm`] with α=1, β=0.
///
/// A must be (m x k), B must be (k x n). Both must be 2D row-major tensors.
/// Returns a row-major (m x n) tensor.
pub fn matmul<T: ArrowPrimitiveType>(
    a: &Tensor<'_, T>,
    b: &Tensor<'_, T>,
) -> Result<Tensor<'static, T>>
where
    T::Native: Zero + One + Copy + Mul<Output = T::Native> + Add<Output = T::Native> + AddAssign,
{
    gemm(T::Native::one(), a, b, T::Native::zero(), None)
}

/// General matrix-vector multiplication: y = α * A * x + β * y.
///
/// A must be (m x n), x must be a 1D tensor of length n.
/// If `y` is provided, it must be 1D of length m and is scaled by `beta` then accumulated into.
/// If `y` is `None`, `beta` is ignored and the result is just α * A * x.
///
/// Returns a 1D tensor of length m.
pub fn gemv<T: ArrowPrimitiveType>(
    alpha: T::Native,
    a: &Tensor<'_, T>,
    x: &Tensor<'_, T>,
    beta: T::Native,
    y: Option<&Tensor<'_, T>>,
) -> Result<Tensor<'static, T>>
where
    T::Native: Zero + One + Copy + Mul<Output = T::Native> + Add<Output = T::Native> + AddAssign,
{
    let (m, n) = shape_2d(a, "gemv")?;

    let x_shape = x
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("gemv: vector has no shape".to_string()))?;
    if x_shape.len() != 1 {
        return Err(KernelError::InvalidArgument(format!(
            "gemv: expected 1D vector, got {}D",
            x_shape.len()
        )));
    }
    if x_shape[0] != n {
        return Err(KernelError::ShapeMismatch {
            operation: "gemv",
            expected: format!("vector length {n}"),
            actual: format!("length {}", x_shape[0]),
        });
    }

    if let Some(y_tensor) = y {
        let y_shape = y_tensor.shape().ok_or_else(|| {
            KernelError::InvalidArgument("gemv: y vector has no shape".to_string())
        })?;
        if y_shape.len() != 1 || y_shape[0] != m {
            return Err(KernelError::ShapeMismatch {
                operation: "gemv",
                expected: format!("y length {m}"),
                actual: format!(
                    "{}D length {}",
                    y_shape.len(),
                    y_shape.first().unwrap_or(&0)
                ),
            });
        }
    }

    let a_buf = a.data();
    let x_buf = x.data();

    let result_buf = match T::DATA_TYPE {
        DataType::Float32 => {
            let mut ax = matvec_f32(a_buf.typed_data(), x_buf.typed_data(), m, n);
            let y_f32: Option<&[f32]> = y.map(|t| t.data().typed_data());
            let alpha_f32: f32 = unsafe { *(&alpha as *const T::Native as *const f32) };
            let beta_f32: f32 = unsafe { *(&beta as *const T::Native as *const f32) };
            apply_alpha_beta(&mut ax, alpha_f32, beta_f32, y_f32);
            Buffer::from_vec(ax)
        }
        DataType::Float64 => {
            let mut ax = gemv_f64(a_buf.typed_data(), x_buf.typed_data(), m, n);
            let y_f64: Option<&[f64]> = y.map(|t| t.data().typed_data());
            let alpha_f64: f64 = unsafe { *(&alpha as *const T::Native as *const f64) };
            let beta_f64: f64 = unsafe { *(&beta as *const T::Native as *const f64) };
            apply_alpha_beta(&mut ax, alpha_f64, beta_f64, y_f64);
            Buffer::from_vec(ax)
        }
        _ => {
            let mut ax: Vec<T::Native> =
                naive_gemv_generic(a_buf.typed_data(), x_buf.typed_data(), m, n);
            let y_slice: Option<&[T::Native]> = y.map(|t| t.data().typed_data());
            apply_alpha_beta(&mut ax, alpha, beta, y_slice);
            Buffer::from_vec(ax)
        }
    };

    buf_to_tensor_1d::<T>(result_buf, m)
}

/// Matrix-vector multiplication: y = A * x.
///
/// Convenience wrapper around [`gemv`] with α=1, β=0.
///
/// A must be (m x n), x must be a 1D tensor of length n.
/// Returns a 1D tensor of length m.
pub fn matvec<T: ArrowPrimitiveType>(
    a: &Tensor<'_, T>,
    x: &Tensor<'_, T>,
) -> Result<Tensor<'static, T>>
where
    T::Native: Zero + One + Copy + Mul<Output = T::Native> + Add<Output = T::Native> + AddAssign,
{
    gemv(T::Native::one(), a, x, T::Native::zero(), None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::{Float32Type, Float64Type};

    fn make_f32_2d(data: Vec<f32>, rows: usize, cols: usize) -> Tensor<'static, Float32Type> {
        let buffer = ScalarBuffer::<f32>::from(data).into_inner();
        Tensor::new_row_major(buffer, Some(vec![rows, cols]), None).unwrap()
    }

    fn make_f64_2d(data: Vec<f64>, rows: usize, cols: usize) -> Tensor<'static, Float64Type> {
        let buffer = ScalarBuffer::<f64>::from(data).into_inner();
        Tensor::new_row_major(buffer, Some(vec![rows, cols]), None).unwrap()
    }

    fn make_f32_1d(data: Vec<f32>) -> Tensor<'static, Float32Type> {
        let len = data.len();
        let buffer = ScalarBuffer::<f32>::from(data).into_inner();
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
        let data = y.data().typed_data::<f32>();
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
        let a = make_tensor_2d::<Int32Type>(vec![1i32, 2, 3, 4, 5, 6], 2, 3);
        let b = make_tensor_2d::<Int32Type>(vec![7i32, 8, 9, 10, 11, 12], 3, 2);
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

    // ---- gemm tests ----

    #[test]
    fn test_gemm_alpha_only() {
        // C = 2.0 * A * B (no existing C)
        let a = make_f32_2d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = make_f32_2d(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
        let c = gemm(2.0f32, &a, &b, 0.0, None).unwrap();

        let data = c.data().typed_data::<f32>();
        // AB = [58, 64, 139, 154], scaled by 2
        assert!((data[0] - 116.0).abs() < 1e-4);
        assert!((data[1] - 128.0).abs() < 1e-4);
        assert!((data[2] - 278.0).abs() < 1e-4);
        assert!((data[3] - 308.0).abs() < 1e-4);
    }

    #[test]
    fn test_gemm_alpha_beta() {
        // C = 1.0 * A * B + 0.5 * C_existing
        let a = make_f32_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = make_f32_2d(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        let c_existing = make_f32_2d(vec![100.0, 200.0, 300.0, 400.0], 2, 2);

        // AB = [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
        // result = 1.0 * AB + 0.5 * C = [19+50, 22+100, 43+150, 50+200]
        //        = [69, 122, 193, 250]
        let c = gemm(1.0f32, &a, &b, 0.5, Some(&c_existing)).unwrap();
        let data = c.data().typed_data::<f32>();
        assert!((data[0] - 69.0).abs() < 1e-4);
        assert!((data[1] - 122.0).abs() < 1e-4);
        assert!((data[2] - 193.0).abs() < 1e-4);
        assert!((data[3] - 250.0).abs() < 1e-4);
    }

    #[test]
    fn test_gemm_identity_is_matmul() {
        // gemm with α=1, β=0, no C should match matmul
        let a = make_f32_2d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = make_f32_2d(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
        let c_gemm = gemm(1.0f32, &a, &b, 0.0, None).unwrap();
        let c_matmul = matmul(&a, &b).unwrap();

        let d1 = c_gemm.data().typed_data::<f32>();
        let d2 = c_matmul.data().typed_data::<f32>();
        for i in 0..4 {
            assert!((d1[i] - d2[i]).abs() < 1e-6);
        }
    }

    // ---- gemv tests ----

    #[test]
    fn test_gemv_alpha_only() {
        // y = 3.0 * A * x
        let a = make_f32_2d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let x = make_f32_1d(vec![1.0, 1.0, 1.0]);
        let y = gemv(3.0f32, &a, &x, 0.0, None).unwrap();

        // Ax = [6, 15], scaled by 3 = [18, 45]
        let data: &[f32] = y.data().typed_data();
        assert!((data[0] - 18.0).abs() < 1e-5);
        assert!((data[1] - 45.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemv_alpha_beta() {
        // y = 2.0 * A * x + 0.5 * y_existing
        let a = make_f32_2d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let x = make_f32_1d(vec![1.0, 1.0, 1.0]);
        let y_existing = make_tensor_1d::<Float32Type>(vec![100.0, 200.0]);

        // Ax = [6, 15]
        // result = 2*[6,15] + 0.5*[100,200] = [12+50, 30+100] = [62, 130]
        let y = gemv(2.0f32, &a, &x, 0.5, Some(&y_existing)).unwrap();
        let data: &[f32] = y.data().typed_data();
        assert!((data[0] - 62.0).abs() < 1e-5);
        assert!((data[1] - 130.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemv_identity_is_matvec() {
        let a = make_f32_2d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let x = make_f32_1d(vec![1.0, 1.0, 1.0]);
        let y_gemv = gemv(1.0f32, &a, &x, 0.0, None).unwrap();
        let y_matvec = matvec(&a, &x).unwrap();

        let d1: &[f32] = y_gemv.data().typed_data();
        let d2: &[f32] = y_matvec.data().typed_data();
        for i in 0..2 {
            assert!((d1[i] - d2[i]).abs() < 1e-6);
        }
    }
}

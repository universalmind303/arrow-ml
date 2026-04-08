//! Matmul kernel ABI.
//!
//! A backend exports the matmul kernel as a four-symbol family:
//!
//! - `am_matmul_supports(dtype, device_type) -> i32` — pre-dispatch capability probe
//! - `am_matmul_open(dtype, device_type, *out_handle) -> i32` — create a reusable handle
//! - `am_matmul_invoke(handle, *a, *b, *c) -> i32` — compute C = A @ B
//! - `am_matmul_close(handle)` — destroy the handle
//!
//! The symbol family is loaded as a unit by [`MatmulOps::load`]. If any
//! symbol is missing, the whole vtable is `None` and the backend doesn't
//! support matmul.

use crate::device_tensor::FFI_TensorArray;
use libloading::Library;

/// Opaque matmul kernel handle.
///
/// Owned by the backend, created by `am_matmul_open`, destroyed by
/// `am_matmul_close`. The per-kernel typed handle (vs. a generic
/// `*mut c_void`) means the type system catches "passed a softmax handle
/// to matmul" mismatches as link errors at the FFI boundary.
#[repr(C)]
pub struct AmMatmulKernel {
    _opaque: [u8; 0],
}

// ----- C ABI symbols (function pointer types) -----

/// Returns `1` if this backend supports matmul for the given dtype on the
/// given device, `0` if not. Negative values are ABI errors.
pub type AmMatmulSupportsFn = unsafe extern "C" fn(dtype: i32, device_type: i32) -> i32;

/// Open a matmul kernel for a specific dtype/device.
///
/// The handle is reusable across many invocations with varying shapes.
///
/// On success, writes `*out_handle` and returns `AM_OK`. On failure
/// (unsupported dtype, OOM, etc.), writes NULL and returns a negative
/// error code.
pub type AmMatmulOpenFn = unsafe extern "C" fn(
    dtype: i32,
    device_type: i32,
    out_handle: *mut *mut AmMatmulKernel,
) -> i32;

/// Compute `c = a @ b`.
///
/// Shapes:
/// - `a` is rank-2 with shape `[m, k]`
/// - `b` is rank-2 with shape `[k, n]`
/// - `c` is rank-2 with shape `[m, n]`, pre-allocated by the caller
///
/// All three tensors must match the dtype/device the kernel was opened with.
/// Synchronous in v2 — when this returns `AM_OK`, `c` is ready to read.
pub type AmMatmulInvokeFn = unsafe extern "C" fn(
    handle: *mut AmMatmulKernel,
    a: *const FFI_TensorArray,
    b: *const FFI_TensorArray,
    c: *mut FFI_TensorArray,
) -> i32;

/// Destroy a matmul kernel handle. Frees backend-owned resources (cached
/// buffers, pipeline state references, etc.).
pub type AmMatmulCloseFn = unsafe extern "C" fn(handle: *mut AmMatmulKernel);

// ----- Vtable -----

/// The matmul ABI symbol family. Loaded as a unit — if any symbol is
/// missing the whole vtable is `None` and the backend doesn't export matmul.
#[derive(Copy, Clone)]
pub struct MatmulOps {
    pub supports: AmMatmulSupportsFn,
    pub open: AmMatmulOpenFn,
    pub invoke: AmMatmulInvokeFn,
    pub close: AmMatmulCloseFn,
}

// SAFETY: function pointers obtained from a loaded shared library are
// Send + Sync. The library handle that backs them is kept alive by the
// owning `Backend` for the lifetime of the vtable.
unsafe impl Send for MatmulOps {}
unsafe impl Sync for MatmulOps {}

impl MatmulOps {
    /// Look up the matmul symbol family in `lib`. Returns `None` if any
    /// symbol is missing — the backend simply doesn't support matmul.
    pub(crate) fn load(lib: &Library) -> Option<Self> {
        unsafe {
            Some(MatmulOps {
                supports: *lib.get(b"am_matmul_supports\0").ok()?,
                open: *lib.get(b"am_matmul_open\0").ok()?,
                invoke: *lib.get(b"am_matmul_invoke\0").ok()?,
                close: *lib.get(b"am_matmul_close\0").ok()?,
            })
        }
    }

    /// Quick capability probe — does this backend support matmul for this
    /// dtype on this device? No allocation, no handle open. Use this before
    /// dispatching to skip backends that don't support the dtype the caller
    /// has.
    pub fn supports_dtype(&self, dtype: i32, device_type: i32) -> bool {
        unsafe { (self.supports)(dtype, device_type) == 1 }
    }
}

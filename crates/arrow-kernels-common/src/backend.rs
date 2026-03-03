//! C ABI contract for dynamically loaded backend libraries.
//!
//! Every backend is a cdylib that exports `extern "C"` functions with
//! well-known names. The runtime loads them via `libloading` and calls
//! through the function pointers stored here.

use std::ffi::c_char;

// ---------------------------------------------------------------------------
// ABI result codes
// ---------------------------------------------------------------------------

/// Operation succeeded.
pub const AK_OK: i32 = 0;
/// The backend does not implement this kernel.
pub const AK_ERR_UNSUPPORTED: i32 = -1;
/// A GPU / device error occurred.
pub const AK_ERR_GPU: i32 = -2;
/// Invalid argument (bad dimensions, null pointer, etc.).
pub const AK_ERR_INVALID: i32 = -3;

// ---------------------------------------------------------------------------
// Function-pointer types matching the C ABI
// ---------------------------------------------------------------------------

/// Returns a static C string identifying the backend (e.g. `"metal\0"`).
pub type AkBackendNameFn = unsafe extern "C" fn() -> *const c_char;

/// Returns a priority value; higher means more preferred.
pub type AkBackendPriorityFn = unsafe extern "C" fn() -> u32;

/// `C = A * B`  for f32.  All pointers are row-major, caller-allocated.
/// Returns `AK_OK` on success.
pub type AkMatmulF32Fn =
    unsafe extern "C" fn(a: *const f32, b: *const f32, c: *mut f32, m: u32, k: u32, n: u32) -> i32;

/// `C = A * B`  for f64.
pub type AkMatmulF64Fn =
    unsafe extern "C" fn(a: *const f64, b: *const f64, c: *mut f64, m: u32, k: u32, n: u32) -> i32;

// ---------------------------------------------------------------------------
// Loaded backend handle
// ---------------------------------------------------------------------------

/// A successfully loaded backend library with its function pointers.
pub struct Backend {
    /// Keep the library handle alive so the function pointers remain valid.
    _lib: libloading::Library,
    /// Human-readable name (e.g. "metal", "cuda").
    pub name: String,
    /// Priority — higher is more preferred.
    pub priority: u32,
    /// Optional kernel entry points. `None` means the backend doesn't support it.
    pub matmul_f32: Option<AkMatmulF32Fn>,
    pub matmul_f64: Option<AkMatmulF64Fn>,
}

// SAFETY: The function pointers come from a loaded shared library whose
// handle (`_lib`) is kept alive for the lifetime of this struct.  The
// pointers themselves are plain `extern "C"` fn pointers which are Send+Sync.
unsafe impl Send for Backend {}
unsafe impl Sync for Backend {}

impl Backend {
    /// Try to load a backend from the shared library at `path`.
    ///
    /// Returns `None` if the library can't be opened or doesn't export
    /// the mandatory `ak_backend_name` / `ak_backend_priority` symbols.
    pub fn load(path: &std::path::Path) -> Option<Self> {
        let lib = unsafe { libloading::Library::new(path) }.ok()?;

        // Mandatory symbols
        let name_fn: AkBackendNameFn = unsafe {
            *lib.get::<AkBackendNameFn>(b"ak_backend_name\0").ok()?
        };
        let priority_fn: AkBackendPriorityFn = unsafe {
            *lib.get::<AkBackendPriorityFn>(b"ak_backend_priority\0").ok()?
        };

        let name = unsafe {
            let ptr = name_fn();
            std::ffi::CStr::from_ptr(ptr)
                .to_string_lossy()
                .into_owned()
        };
        let priority = unsafe { priority_fn() };

        // Optional kernel symbols
        let matmul_f32: Option<AkMatmulF32Fn> = unsafe {
            lib.get::<AkMatmulF32Fn>(b"ak_matmul_f32\0")
                .ok()
                .map(|s| *s)
        };
        let matmul_f64: Option<AkMatmulF64Fn> = unsafe {
            lib.get::<AkMatmulF64Fn>(b"ak_matmul_f64\0")
                .ok()
                .map(|s| *s)
        };

        Some(Backend {
            _lib: lib,
            name,
            priority,
            matmul_f32,
            matmul_f64,
        })
    }
}

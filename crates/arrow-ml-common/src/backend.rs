//! C ABI contract for dynamically loaded backend libraries.
//!
//! Every backend is a cdylib that exports `extern "C"` functions with
//! well-known names. The runtime loads them via `libloading` and calls
//! through the function pointers stored here.

use std::ffi::c_char;

// ---------------------------------------------------------------------------
// ABI version
// ---------------------------------------------------------------------------

/// The current backend ABI version. Bump on any breaking change to function
/// signatures, struct layouts, or symbol names. Backends must export
/// `am_backend_abi_version` returning this exact value or they will be
/// rejected by the loader.
pub const ARROW_ML_BACKEND_ABI_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// ABI result codes
// ---------------------------------------------------------------------------

/// Operation succeeded.
pub const AM_OK: i32 = 0;
/// The backend does not implement this kernel.
pub const AM_ERR_UNSUPPORTED: i32 = -1;
/// A GPU / device error occurred.
pub const AM_ERR_GPU: i32 = -2;
/// Invalid argument (bad dimensions, null pointer, etc.).
pub const AM_ERR_INVALID: i32 = -3;

// ---------------------------------------------------------------------------
// Function-pointer types matching the C ABI
// ---------------------------------------------------------------------------

/// Returns the ABI version this backend was built against. Must equal
/// [`ARROW_ML_BACKEND_ABI_VERSION`] for the loader to accept the backend.
pub type AmBackendAbiVersionFn = unsafe extern "C" fn() -> u32;

/// Returns a static C string identifying the backend (e.g. `"metal\0"`).
pub type AmBackendNameFn = unsafe extern "C" fn() -> *const c_char;

/// Returns a priority value; higher means more preferred.
pub type AmBackendPriorityFn = unsafe extern "C" fn() -> u32;

/// `C = A * B`  for f32.  All pointers are row-major, caller-allocated.
/// Returns `AM_OK` on success.
pub type AmMatmulF32Fn =
    unsafe extern "C" fn(a: *const f32, b: *const f32, c: *mut f32, m: u32, k: u32, n: u32) -> i32;

/// `C = A * B`  for f64.
pub type AmMatmulF64Fn =
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
    pub matmul_f32: Option<AmMatmulF32Fn>,
    pub matmul_f64: Option<AmMatmulF64Fn>,
}

// SAFETY: The function pointers come from a loaded shared library whose
// handle (`_lib`) is kept alive for the lifetime of this struct.  The
// pointers themselves are plain `extern "C"` fn pointers which are Send+Sync.
unsafe impl Send for Backend {}
unsafe impl Sync for Backend {}

impl Backend {
    /// Try to load a backend from the shared library at `path`.
    ///
    /// Returns `None` if the library can't be opened, doesn't export the
    /// mandatory `am_backend_abi_version` / `am_backend_name` /
    /// `am_backend_priority` symbols, or reports an ABI version that
    /// doesn't match [`ARROW_ML_BACKEND_ABI_VERSION`].
    pub fn load(path: &std::path::Path) -> Option<Self> {
        let lib = unsafe { libloading::Library::new(path) }.ok()?;

        // ABI version check first — refuse anything that doesn't match.
        let abi_fn: AmBackendAbiVersionFn =
            match unsafe { lib.get::<AmBackendAbiVersionFn>(b"am_backend_abi_version\0") } {
                Ok(sym) => *sym,
                Err(_) => {
                    eprintln!(
                        "[arrow-ml] backend at {} is missing am_backend_abi_version — skipping",
                        path.display()
                    );
                    return None;
                }
            };
        let reported = unsafe { abi_fn() };
        if reported != ARROW_ML_BACKEND_ABI_VERSION {
            eprintln!(
                "[arrow-ml] backend at {} reports ABI v{}, expected v{} — skipping",
                path.display(),
                reported,
                ARROW_ML_BACKEND_ABI_VERSION
            );
            return None;
        }

        // Mandatory identification symbols
        let name_fn: AmBackendNameFn =
            unsafe { *lib.get::<AmBackendNameFn>(b"am_backend_name\0").ok()? };
        let priority_fn: AmBackendPriorityFn = unsafe {
            *lib.get::<AmBackendPriorityFn>(b"am_backend_priority\0")
                .ok()?
        };

        let name = unsafe {
            let ptr = name_fn();
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        };
        let priority = unsafe { priority_fn() };

        // Optional kernel symbols
        let matmul_f32: Option<AmMatmulF32Fn> = unsafe {
            lib.get::<AmMatmulF32Fn>(b"am_matmul_f32\0")
                .ok()
                .map(|s| *s)
        };
        let matmul_f64: Option<AmMatmulF64Fn> = unsafe {
            lib.get::<AmMatmulF64Fn>(b"am_matmul_f64\0")
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

//! C ABI contract for dynamically loaded backend libraries.
//!
//! Every backend is a `cdylib` that exports `extern "C"` functions with
//! well-known names. The runtime loads them via `libloading` and calls
//! through the function pointers stored here.
//!
//! See `crates/arrow-ml-common/src/kernels/` for the per-kernel ABI symbol
//! families. Each kernel has its own four-symbol family
//! (`am_<kernel>_supports/open/invoke/close`) loaded as a unit.

use crate::kernels::matmul::MatmulOps;
use std::ffi::c_char;

// ---------------------------------------------------------------------------
// ABI version
// ---------------------------------------------------------------------------

/// Experimental backend ABI version. While the contract is iterating,
/// breaking changes happen *in place* at version 1 — the version number
/// will only be bumped once the shape stabilizes. Backends must export
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
/// The backend doesn't support the requested dtype.
pub const AM_ERR_UNSUPPORTED_DTYPE: i32 = -4;
/// A tensor is on a device this kernel handle wasn't opened for.
pub const AM_ERR_DEVICE_MISMATCH: i32 = -5;

// ---------------------------------------------------------------------------
// Mandatory backend identification ABI
// ---------------------------------------------------------------------------

/// Returns the ABI version this backend was built against. Must equal
/// [`ARROW_ML_BACKEND_ABI_VERSION`] for the loader to accept the backend.
pub type AmBackendAbiVersionFn = unsafe extern "C" fn() -> u32;

/// Returns a static C string identifying the backend (e.g. `"metal\0"`).
pub type AmBackendNameFn = unsafe extern "C" fn() -> *const c_char;

/// Returns a priority value; higher means more preferred.
pub type AmBackendPriorityFn = unsafe extern "C" fn() -> u32;

/// Returns the device type this backend dispatches on as an
/// [`crate::device_tensor::AmDeviceType`] integer value.
pub type AmBackendDeviceTypeFn = unsafe extern "C" fn() -> i32;

/// Optional thread-local last-error retrieval. Returns a NUL-terminated
/// UTF-8 string describing the last error from any ABI call on the current
/// thread, or NULL if no error has been recorded. The pointer is valid
/// until the next ABI call on this thread.
pub type AmLastErrorFn = unsafe extern "C" fn() -> *const c_char;

// ---------------------------------------------------------------------------
// Loaded backend handle
// ---------------------------------------------------------------------------

/// A successfully loaded backend library with its identification metadata
/// and per-kernel vtables.
///
/// Each `Option<KernelOps>` field represents one kernel. `Some(_)` means
/// the backend exported the full symbol family for that kernel; `None`
/// means at least one symbol was missing and the backend cannot be used
/// for that kernel.
pub struct Backend {
    /// Keep the library handle alive so the function pointers remain valid.
    _lib: libloading::Library,
    /// Human-readable name (e.g. "metal", "cuda").
    pub name: String,
    /// Priority — higher is more preferred.
    pub priority: u32,
    /// Device type the backend runs on (`AmDeviceType` value).
    pub device_type: i32,
    /// Optional thread-local last-error retrieval, if the backend exports it.
    pub last_error: Option<AmLastErrorFn>,

    // ===== per-kernel vtables =====
    /// Matmul kernel vtable, if the backend exports the full
    /// `am_matmul_*` symbol family.
    pub matmul: Option<MatmulOps>,
}

// SAFETY: The function pointers come from a loaded shared library whose
// handle (`_lib`) is kept alive for the lifetime of this struct. The
// pointers themselves are plain `extern "C"` fn pointers which are Send+Sync.
unsafe impl Send for Backend {}
unsafe impl Sync for Backend {}

impl Backend {
    /// Try to load a backend from the shared library at `path`.
    ///
    /// Returns `None` if the library can't be opened, doesn't export the
    /// mandatory identification symbols, or reports an ABI version that
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
        let device_type_fn: AmBackendDeviceTypeFn = unsafe {
            *lib.get::<AmBackendDeviceTypeFn>(b"am_backend_device_type\0")
                .ok()?
        };

        let name = unsafe {
            let ptr = name_fn();
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        };
        let priority = unsafe { priority_fn() };
        let device_type = unsafe { device_type_fn() };

        // Optional last-error retrieval
        let last_error: Option<AmLastErrorFn> = unsafe {
            lib.get::<AmLastErrorFn>(b"am_last_error_message\0")
                .ok()
                .map(|s| *s)
        };

        // Per-kernel vtables — each loads as an all-or-nothing unit.
        let matmul = MatmulOps::load(&lib);

        Some(Backend {
            _lib: lib,
            name,
            priority,
            device_type,
            last_error,
            matmul,
        })
    }

    /// Read the backend's last-error message on the current thread, if it
    /// exports `am_last_error_message`. Returns an empty string if the
    /// backend doesn't export the symbol or no error has been recorded.
    pub fn last_error_message(&self) -> String {
        let Some(f) = self.last_error else {
            return String::new();
        };
        let ptr = unsafe { f() };
        if ptr.is_null() {
            return String::new();
        }
        unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

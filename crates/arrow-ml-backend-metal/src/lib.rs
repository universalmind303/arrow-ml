//! Metal GPU backend for arrow-ml.
//!
//! Compiled as a cdylib (`libarrow_ml_backend_metal.dylib`).
//! Discovered and loaded at runtime by `arrow-ml-common::registry`.

mod matmul;

use std::ffi::c_char;

/// ABI version this backend was built against. Must match
/// `arrow_ml_common::backend::ARROW_ML_BACKEND_ABI_VERSION` or the loader
/// will reject this backend.
const BACKEND_ABI_VERSION: u32 = 1;

/// ABI result codes (must match arrow-ml-common/src/backend.rs)
const AM_OK: i32 = 0;
const AM_ERR_GPU: i32 = -2;

// ---------------------------------------------------------------------------
// Mandatory ABI exports
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn am_backend_abi_version() -> u32 {
    BACKEND_ABI_VERSION
}

#[no_mangle]
pub extern "C" fn am_backend_name() -> *const c_char {
    c"metal".as_ptr()
}

#[no_mangle]
pub extern "C" fn am_backend_priority() -> u32 {
    100
}

// ---------------------------------------------------------------------------
// Kernel exports
// ---------------------------------------------------------------------------

/// f32 matrix multiplication on Metal GPU.
///
/// # Safety
///
/// - `a` must point to at least `m * k` f32 values.
/// - `b` must point to at least `k * n` f32 values.
/// - `c` must point to at least `m * n` writable f32 values.
#[no_mangle]
pub unsafe extern "C" fn am_matmul_f32(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: u32,
    k: u32,
    n: u32,
) -> i32 {
    let m = m as usize;
    let k = k as usize;
    let n = n as usize;

    let a_slice = std::slice::from_raw_parts(a, m * k);
    let b_slice = std::slice::from_raw_parts(b, k * n);

    match matmul::metal_matmul_f32(a_slice, b_slice, m, k, n) {
        Ok(result) => {
            std::ptr::copy_nonoverlapping(result.as_ptr(), c, m * n);
            AM_OK
        }
        Err(_msg) => AM_ERR_GPU,
    }
}

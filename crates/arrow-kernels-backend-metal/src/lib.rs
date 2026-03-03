//! Metal GPU backend for arrow-kernels.
//!
//! Compiled as a cdylib (`libarrow_kernels_backend_metal.dylib`).
//! Discovered and loaded at runtime by `arrow-kernels-common::registry`.

mod matmul;

use std::ffi::c_char;

/// ABI result codes (must match arrow-kernels-common/src/backend.rs)
const AK_OK: i32 = 0;
const AK_ERR_GPU: i32 = -2;

// ---------------------------------------------------------------------------
// Mandatory ABI exports
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn ak_backend_name() -> *const c_char {
    c"metal".as_ptr()
}

#[no_mangle]
pub extern "C" fn ak_backend_priority() -> u32 {
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
pub unsafe extern "C" fn ak_matmul_f32(
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
            AK_OK
        }
        Err(_msg) => AK_ERR_GPU,
    }
}

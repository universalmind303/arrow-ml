//! Metal GPU backend for arrow-ml.
//!
//! Compiled as a cdylib (`libarrow_ml_backend_metal.dylib`).
//! Discovered and loaded at runtime by `arrow-ml-common::registry`.

mod matmul;

use arrow_ml_common::backend::{
    AM_ERR_DEVICE_MISMATCH, AM_ERR_GPU, AM_ERR_INVALID, AM_ERR_UNSUPPORTED_DTYPE, AM_OK,
    ARROW_ML_BACKEND_ABI_VERSION,
};
use arrow_ml_common::device_tensor::{dtype, AmDeviceType, FFI_TensorArray};
use arrow_ml_common::kernels::matmul::AmMatmulKernel;
use matmul::MatmulKernel;
use std::cell::RefCell;
use std::ffi::{c_char, CString};

// ---------------------------------------------------------------------------
// Mandatory ABI exports
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn am_backend_abi_version() -> u32 {
    ARROW_ML_BACKEND_ABI_VERSION
}

#[no_mangle]
pub extern "C" fn am_backend_name() -> *const c_char {
    c"metal".as_ptr()
}

#[no_mangle]
pub extern "C" fn am_backend_priority() -> u32 {
    100
}

#[no_mangle]
pub extern "C" fn am_backend_device_type() -> i32 {
    AmDeviceType::Metal as i32
}

// ---------------------------------------------------------------------------
// Optional thread-local last-error reporting
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: impl Into<String>) {
    let cstring = CString::new(msg.into())
        .unwrap_or_else(|_| CString::new("invalid error message").unwrap());
    let _ = LAST_ERROR.try_with(|cell| {
        *cell.borrow_mut() = Some(cstring);
    });
}

fn clear_last_error() {
    let _ = LAST_ERROR.try_with(|cell| {
        *cell.borrow_mut() = None;
    });
}

#[no_mangle]
pub extern "C" fn am_last_error_message() -> *const c_char {
    LAST_ERROR
        .try_with(|cell| match cell.borrow().as_ref() {
            Some(s) => s.as_ptr(),
            None => std::ptr::null(),
        })
        .unwrap_or(std::ptr::null())
}

// ---------------------------------------------------------------------------
// Matmul kernel ABI
// ---------------------------------------------------------------------------

/// Returns 1 if the Metal matmul kernel can handle this `(dtype, device)`
/// combination, 0 otherwise. The Metal MSL backend is f32-only and accepts
/// either CPU tensors (host-staged into MTLBuffers internally) or Metal
/// device tensors. f64 is intentionally unsupported because Metal MSL has
/// no `double` type.
#[no_mangle]
pub extern "C" fn am_matmul_supports(dtype_code: i32, device_type: i32) -> i32 {
    let cpu = AmDeviceType::Cpu as i32;
    let metal_dev = AmDeviceType::Metal as i32;
    if dtype_code == dtype::FLOAT32 && (device_type == cpu || device_type == metal_dev) {
        1
    } else {
        0
    }
}

/// # Safety
///
/// `out_handle` must be a valid, writable pointer to a `*mut AmMatmulKernel`
/// slot or NULL. On success the slot receives an opaque kernel handle that
/// must be released via [`am_matmul_close`]; on failure the slot is set to
/// NULL.
#[no_mangle]
pub unsafe extern "C" fn am_matmul_open(
    dtype_code: i32,
    device_type: i32,
    out_handle: *mut *mut AmMatmulKernel,
) -> i32 {
    if out_handle.is_null() {
        set_last_error("am_matmul_open: out_handle is NULL");
        return AM_ERR_INVALID;
    }
    unsafe { *out_handle = std::ptr::null_mut() };

    if dtype_code != dtype::FLOAT32 {
        set_last_error(format!(
            "metal matmul: unsupported dtype {dtype_code} (only FLOAT32)"
        ));
        return AM_ERR_UNSUPPORTED_DTYPE;
    }
    let cpu = AmDeviceType::Cpu as i32;
    let metal_dev = AmDeviceType::Metal as i32;
    if device_type != cpu && device_type != metal_dev {
        set_last_error(format!(
            "metal matmul: unsupported device {device_type} (only CPU or Metal)"
        ));
        return AM_ERR_DEVICE_MISMATCH;
    }

    match MatmulKernel::new(dtype_code) {
        Ok(kernel) => {
            clear_last_error();
            let boxed = Box::new(kernel);
            unsafe { *out_handle = Box::into_raw(boxed) as *mut AmMatmulKernel };
            AM_OK
        }
        Err(msg) => {
            set_last_error(msg);
            AM_ERR_GPU
        }
    }
}

/// # Safety
///
/// `handle` must be a kernel handle previously returned by [`am_matmul_open`]
/// and not yet closed. `a`, `b`, `c` must be valid, non-NULL
/// `FFI_TensorArray` pointers whose tensor metadata accurately describes
/// their backing storage for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn am_matmul_invoke(
    handle: *mut AmMatmulKernel,
    a: *const FFI_TensorArray,
    b: *const FFI_TensorArray,
    c: *mut FFI_TensorArray,
) -> i32 {
    if handle.is_null() {
        set_last_error("am_matmul_invoke: handle is NULL");
        return AM_ERR_INVALID;
    }
    if a.is_null() || b.is_null() || c.is_null() {
        set_last_error("am_matmul_invoke: tensor pointer is NULL");
        return AM_ERR_INVALID;
    }
    let kernel = unsafe { &mut *(handle as *mut MatmulKernel) };
    let a_ref = unsafe { &*a };
    let b_ref = unsafe { &*b };
    let c_ref = unsafe { &mut *c };

    match unsafe { kernel.invoke(a_ref, b_ref, c_ref) } {
        Ok(()) => {
            clear_last_error();
            AM_OK
        }
        Err(msg) => {
            set_last_error(msg);
            AM_ERR_GPU
        }
    }
}

/// # Safety
///
/// `handle` must be a kernel handle previously returned by [`am_matmul_open`]
/// and not yet closed, or NULL (in which case this is a no-op). After this
/// call the handle is invalidated and must not be reused.
#[no_mangle]
pub unsafe extern "C" fn am_matmul_close(handle: *mut AmMatmulKernel) {
    if handle.is_null() {
        return;
    }
    let _ = unsafe { Box::from_raw(handle as *mut MatmulKernel) };
    clear_last_error();
}

//! Metal GPU backend for arrow-ml.
//!
//! Compiled as a cdylib (`libarrow_ml_backend_metal.dylib`).
//! Discovered and loaded at runtime by `arrow-ml-common::registry`.

mod matmul;

use arrow_ml_common::backend::{
    AM_ERR_DEVICE_MISMATCH, AM_ERR_GPU, AM_ERR_INVALID, AM_ERR_UNSUPPORTED_DTYPE, AM_OK,
    ARROW_ML_BACKEND_ABI_VERSION,
};
use arrow_ml_common::device_tensor::{dtype, AmDeviceType, FFI_DeviceTensor};
use arrow_ml_common::kernels::matmul::AmMatmulKernel;
use matmul::MatmulKernel;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CString};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

static COPY_COUNT: AtomicU64 = AtomicU64::new(0);


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
    let cstring =
        CString::new(msg.into()).unwrap_or_else(|_| CString::new("invalid error message").unwrap());
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
// Metal device handle
// ---------------------------------------------------------------------------

fn metal_device() -> Option<&'static metal::Device> {
    static INSTANCE: OnceLock<Option<metal::Device>> = OnceLock::new();
    INSTANCE
        .get_or_init(metal::Device::system_default)
        .as_ref()
}

// ---------------------------------------------------------------------------
// Required device memory ABI
// ---------------------------------------------------------------------------
//
// Allocations use `StorageModeShared`, so `contents()` is a host-visible
// pointer into unified memory. We return that pointer directly as the
// device_ptr, which lets the caller advance it freely for slicing. A
// global registry maps each contents pointer back to its `metal::Buffer`
// so that `am_device_free` can release the GPU allocation.
//
// Because all pointers are in the unified address space, `am_device_copy`
// is a plain `memcpy` regardless of device types.

fn alloc_registry() -> &'static Mutex<HashMap<usize, metal::Buffer>> {
    static INSTANCE: OnceLock<Mutex<HashMap<usize, metal::Buffer>>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[no_mangle]
pub extern "C" fn am_device_alloc(
    device_type: i32,
    _device_id: i64,
    nbytes: u64,
    out_ptr: *mut *mut c_void,
) -> i32 {
    clear_last_error();
    if device_type != AmDeviceType::Metal as i32 {
        set_last_error(format!(
            "metal backend cannot allocate on device_type {device_type}"
        ));
        return AM_ERR_DEVICE_MISMATCH;
    }
    if out_ptr.is_null() {
        set_last_error("am_device_alloc: out_ptr is null");
        return AM_ERR_INVALID;
    }
    let device = match metal_device() {
        Some(d) => d,
        None => {
            set_last_error("no Metal device available");
            return AM_ERR_GPU;
        }
    };
    let buffer = device.new_buffer(nbytes, metal::MTLResourceOptions::StorageModeShared);
    let ptr = buffer.contents();
    if ptr.is_null() {
        set_last_error("metal buffer contents is null");
        return AM_ERR_GPU;
    }
    alloc_registry()
        .lock()
        .unwrap()
        .insert(ptr as usize, buffer);
    unsafe {
        *out_ptr = ptr;
    }
    AM_OK
}

#[no_mangle]
pub extern "C" fn am_device_free(ptr: *mut c_void, _nbytes: u64) {
    if ptr.is_null() {
        return;
    }
    alloc_registry().lock().unwrap().remove(&(ptr as usize));
}

#[no_mangle]
pub extern "C" fn am_device_copy(
    src: *const c_void,
    _src_dev: i32,
    dst: *mut c_void,
    _dst_dev: i32,
    nbytes: u64,
) -> i32 {
    clear_last_error();
    if src.is_null() || dst.is_null() {
        set_last_error("am_device_copy: null pointer");
        return AM_ERR_INVALID;
    }
    COPY_COUNT.fetch_add(1, Ordering::SeqCst);
    unsafe {
        std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, nbytes as usize);
    }
    AM_OK
}

// ---------------------------------------------------------------------------
// Matmul kernel ABI
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn am_matmul_supports(dt: i32, device_type: i32) -> i32 {
    if dt == dtype::FLOAT32 && device_type == AmDeviceType::Metal as i32 {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn am_matmul_open(
    dt: i32,
    _device_type: i32,
    out_handle: *mut *mut AmMatmulKernel,
) -> i32 {
    clear_last_error();
    if dt != dtype::FLOAT32 {
        set_last_error(format!("metal matmul only supports f32, got dtype {dt}"));
        return AM_ERR_UNSUPPORTED_DTYPE;
    }
    match MatmulKernel::new(dt) {
        Ok(k) => {
            let boxed = Box::new(k);
            unsafe {
                *out_handle = Box::into_raw(boxed) as *mut AmMatmulKernel;
            }
            AM_OK
        }
        Err(msg) => {
            set_last_error(msg);
            AM_ERR_GPU
        }
    }
}

#[no_mangle]
pub extern "C" fn am_matmul_invoke(
    handle: *mut AmMatmulKernel,
    a: *const FFI_DeviceTensor,
    b: *const FFI_DeviceTensor,
    c: *mut FFI_DeviceTensor,
) -> i32 {
    clear_last_error();
    if handle.is_null() || a.is_null() || b.is_null() || c.is_null() {
        set_last_error("am_matmul_invoke: null pointer");
        return AM_ERR_INVALID;
    }
    let kernel = unsafe { &mut *(handle as *mut MatmulKernel) };
    let (a_ref, b_ref, c_ref) = unsafe { (&*a, &*b, &mut *c) };
    match unsafe { kernel.invoke(a_ref, b_ref, c_ref) } {
        Ok(()) => AM_OK,
        Err(msg) => {
            set_last_error(msg);
            AM_ERR_GPU
        }
    }
}

#[no_mangle]
pub extern "C" fn am_matmul_close(handle: *mut AmMatmulKernel) {
    if handle.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(handle as *mut MatmulKernel));
    }
}

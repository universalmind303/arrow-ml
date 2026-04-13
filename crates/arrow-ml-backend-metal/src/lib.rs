//! Metal GPU backend for arrow-ml.
//!
//! Compiled as a cdylib (`libarrow_ml_backend_metal.dylib`).
//! Discovered and loaded at runtime by `arrow-ml-common::registry`.

mod gelu;
mod layernorm;
mod matmul;
mod softmax;

use arrow_ml_common::backend::{AmStatus, ARROW_ML_BACKEND_ABI_VERSION};
use arrow_ml_common::device_tensor::{dtype, AmDeviceType, FFI_DeviceTensor};
use arrow_ml_common::kernels::gelu::AmGeluKernel;
use arrow_ml_common::kernels::layernorm::AmLayerNormKernel;
use arrow_ml_common::kernels::matmul::AmMatmulKernel;
use arrow_ml_common::kernels::softmax::AmSoftmaxKernel;
use gelu::GeluKernel;
use layernorm::LayerNormKernel;
use matmul::MatmulKernel;
use softmax::SoftmaxKernel;
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
        return AmStatus::ErrDeviceMismatch as i32;
    }
    if out_ptr.is_null() {
        set_last_error("am_device_alloc: out_ptr is null");
        return AmStatus::ErrInvalid as i32;
    }
    if nbytes == 0 {
        unsafe {
            *out_ptr = std::mem::align_of::<f64>() as *mut c_void;
        }
        return AmStatus::Ok as i32;
    }
    let device = match metal_device() {
        Some(d) => d,
        None => {
            set_last_error("no Metal device available");
            return AmStatus::ErrGpu as i32;
        }
    };
    let buffer = device.new_buffer(nbytes, metal::MTLResourceOptions::StorageModeShared);
    let ptr = buffer.contents();
    if ptr.is_null() {
        set_last_error("metal buffer contents is null");
        return AmStatus::ErrGpu as i32;
    }
    alloc_registry()
        .lock()
        .unwrap()
        .insert(ptr as usize, buffer);
    unsafe {
        *out_ptr = ptr;
    }
    AmStatus::Ok as i32
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
        return AmStatus::ErrInvalid as i32;
    }
    COPY_COUNT.fetch_add(1, Ordering::SeqCst);
    unsafe {
        std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, nbytes as usize);
    }
    AmStatus::Ok as i32
}

// ---------------------------------------------------------------------------
// Per-kernel ABI exports
// ---------------------------------------------------------------------------
//
// The `metal_kernel_exports!` macro generates the identical supports/open/close
// functions. Only invoke is hand-written per kernel since its signature varies.

macro_rules! metal_kernel_exports {
    (
        abi_prefix: $prefix:ident,
        opaque_handle: $Handle:ty,
        impl_kernel: $Kernel:ty,
        kernel_label: $label:expr,
    ) => {
        ::paste::paste! {
            #[no_mangle]
            pub extern "C" fn [<am_ $prefix _supports>](dt: i32, device_type: i32) -> i32 {
                if dt == dtype::FLOAT32 && device_type == AmDeviceType::Metal as i32 {
                    1
                } else {
                    0
                }
            }

            #[no_mangle]
            pub extern "C" fn [<am_ $prefix _open>](
                dt: i32,
                _device_type: i32,
                out_handle: *mut *mut $Handle,
            ) -> i32 {
                clear_last_error();
                if dt != dtype::FLOAT32 {
                    set_last_error(format!(
                        concat!("metal ", $label, " only supports f32, got dtype {}"), dt
                    ));
                    return AmStatus::ErrUnsupportedDtype as i32;
                }
                match <$Kernel>::new(dt) {
                    Ok(k) => {
                        let boxed = Box::new(k);
                        unsafe {
                            *out_handle = Box::into_raw(boxed) as *mut $Handle;
                        }
                        AmStatus::Ok as i32
                    }
                    Err(msg) => {
                        set_last_error(msg);
                        AmStatus::ErrGpu as i32
                    }
                }
            }

            #[no_mangle]
            pub extern "C" fn [<am_ $prefix _close>](handle: *mut $Handle) {
                if handle.is_null() {
                    return;
                }
                unsafe {
                    drop(Box::from_raw(handle as *mut $Kernel));
                }
            }
        }
    };
}

metal_kernel_exports! {
    abi_prefix: matmul,
    opaque_handle: AmMatmulKernel,
    impl_kernel: MatmulKernel,
    kernel_label: "matmul",
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
        return AmStatus::ErrInvalid as i32;
    }
    let kernel = unsafe { &mut *(handle as *mut MatmulKernel) };
    let (a_ref, b_ref, c_ref) = unsafe { (&*a, &*b, &mut *c) };
    match unsafe { kernel.invoke(a_ref, b_ref, c_ref) } {
        Ok(()) => AmStatus::Ok as i32,
        Err(msg) => {
            set_last_error(msg);
            AmStatus::ErrGpu as i32
        }
    }
}

metal_kernel_exports! {
    abi_prefix: softmax,
    opaque_handle: AmSoftmaxKernel,
    impl_kernel: SoftmaxKernel,
    kernel_label: "softmax",
}

#[no_mangle]
pub extern "C" fn am_softmax_invoke(
    handle: *mut AmSoftmaxKernel,
    input: *const FFI_DeviceTensor,
    output: *mut FFI_DeviceTensor,
    axis: i32,
) -> i32 {
    clear_last_error();
    if handle.is_null() || input.is_null() || output.is_null() {
        set_last_error("am_softmax_invoke: null pointer");
        return AmStatus::ErrInvalid as i32;
    }
    let kernel = unsafe { &mut *(handle as *mut SoftmaxKernel) };
    let (in_ref, out_ref) = unsafe { (&*input, &mut *output) };
    match unsafe { kernel.invoke(in_ref, out_ref, axis) } {
        Ok(()) => AmStatus::Ok as i32,
        Err(msg) => {
            set_last_error(msg);
            AmStatus::ErrGpu as i32
        }
    }
}

metal_kernel_exports! {
    abi_prefix: layernorm,
    opaque_handle: AmLayerNormKernel,
    impl_kernel: LayerNormKernel,
    kernel_label: "layernorm",
}

#[no_mangle]
pub extern "C" fn am_layernorm_invoke(
    handle: *mut AmLayerNormKernel,
    input: *const FFI_DeviceTensor,
    gamma: *const FFI_DeviceTensor,
    beta: *const FFI_DeviceTensor,
    output: *mut FFI_DeviceTensor,
    axis: i32,
    epsilon: f32,
) -> i32 {
    clear_last_error();
    if handle.is_null() || input.is_null() || gamma.is_null() || beta.is_null() || output.is_null()
    {
        set_last_error("am_layernorm_invoke: null pointer");
        return AmStatus::ErrInvalid as i32;
    }
    let kernel = unsafe { &mut *(handle as *mut LayerNormKernel) };
    let (in_ref, gamma_ref, beta_ref, out_ref) =
        unsafe { (&*input, &*gamma, &*beta, &mut *output) };
    match unsafe { kernel.invoke(in_ref, gamma_ref, beta_ref, out_ref, axis, epsilon) } {
        Ok(()) => AmStatus::Ok as i32,
        Err(msg) => {
            set_last_error(msg);
            AmStatus::ErrGpu as i32
        }
    }
}

metal_kernel_exports! {
    abi_prefix: gelu,
    opaque_handle: AmGeluKernel,
    impl_kernel: GeluKernel,
    kernel_label: "gelu",
}

#[no_mangle]
pub extern "C" fn am_gelu_invoke(
    handle: *mut AmGeluKernel,
    input: *const FFI_DeviceTensor,
    output: *mut FFI_DeviceTensor,
) -> i32 {
    clear_last_error();
    if handle.is_null() || input.is_null() || output.is_null() {
        set_last_error("am_gelu_invoke: null pointer");
        return AmStatus::ErrInvalid as i32;
    }
    let kernel = unsafe { &mut *(handle as *mut GeluKernel) };
    let (in_ref, out_ref) = unsafe { (&*input, &mut *output) };
    match unsafe { kernel.invoke(in_ref, out_ref) } {
        Ok(()) => AmStatus::Ok as i32,
        Err(msg) => {
            set_last_error(msg);
            AmStatus::ErrGpu as i32
        }
    }
}

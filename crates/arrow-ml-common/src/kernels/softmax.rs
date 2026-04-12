//! Softmax kernel ABI.
//!
//! A backend exports the softmax kernel as a four-symbol family:
//!
//! - `am_softmax_supports(dtype, device_type) -> i32` — pre-dispatch capability probe
//! - `am_softmax_open(dtype, device_type, *out_handle) -> i32` — create a reusable handle
//! - `am_softmax_invoke(handle, *input, *output, axis) -> i32` — compute softmax along axis
//! - `am_softmax_close(handle)` — destroy the handle

use crate::backend::{AmStatus, Backend};
use crate::device_tensor::FFI_DeviceTensor;
use crate::error::{KernelError, Result};
use libloading::Library;
use std::sync::Arc;

#[repr(C)]
pub struct AmSoftmaxKernel {
    _opaque: [u8; 0],
}

pub type AmSoftmaxSupportsFn = unsafe extern "C" fn(dtype: i32, device_type: i32) -> i32;

pub type AmSoftmaxOpenFn =
    unsafe extern "C" fn(dtype: i32, device_type: i32, out_handle: *mut *mut AmSoftmaxKernel)
        -> i32;

/// Compute `output = softmax(input, axis)`.
///
/// `input` and `output` must have the same shape and dtype. `axis` may be
/// negative (e.g. -1 for the last axis). The backend resolves it to a
/// positive index internally.
pub type AmSoftmaxInvokeFn = unsafe extern "C" fn(
    handle: *mut AmSoftmaxKernel,
    input: *const FFI_DeviceTensor,
    output: *mut FFI_DeviceTensor,
    axis: i32,
) -> i32;

pub type AmSoftmaxCloseFn = unsafe extern "C" fn(handle: *mut AmSoftmaxKernel);

#[derive(Copy, Clone)]
pub struct SoftmaxOps {
    pub supports: AmSoftmaxSupportsFn,
    pub open: AmSoftmaxOpenFn,
    pub invoke: AmSoftmaxInvokeFn,
    pub close: AmSoftmaxCloseFn,
}

unsafe impl Send for SoftmaxOps {}
unsafe impl Sync for SoftmaxOps {}

impl SoftmaxOps {
    pub(crate) fn load(lib: &Library) -> Option<Self> {
        unsafe {
            Some(SoftmaxOps {
                supports: *lib.get(b"am_softmax_supports\0").ok()?,
                open: *lib.get(b"am_softmax_open\0").ok()?,
                invoke: *lib.get(b"am_softmax_invoke\0").ok()?,
                close: *lib.get(b"am_softmax_close\0").ok()?,
            })
        }
    }

    pub fn supports_dtype(&self, dtype: i32, device_type: i32) -> bool {
        unsafe { (self.supports)(dtype, device_type) == 1 }
    }
}

// ----- Safe RAII wrapper -----

pub struct SoftmaxKernel {
    ops: SoftmaxOps,
    handle: *mut AmSoftmaxKernel,
    backend: Arc<Backend>,
}

unsafe impl Send for SoftmaxKernel {}

impl SoftmaxKernel {
    pub fn open(backend: Arc<Backend>, dtype: i32, device_type: i32) -> Result<Self> {
        let ops = backend.softmax.ok_or(KernelError::Unsupported)?;
        let mut handle: *mut AmSoftmaxKernel = std::ptr::null_mut();
        let rc = unsafe { (ops.open)(dtype, device_type, &mut handle) };
        if rc != AmStatus::Ok as i32 || handle.is_null() {
            return Err(KernelError::from_code(rc, &backend));
        }
        Ok(SoftmaxKernel {
            ops,
            handle,
            backend,
        })
    }

    /// # Safety
    ///
    /// The caller asserts that the underlying buffers behind `input` and
    /// `output` are valid for the duration of the call and that the FFI
    /// tensor metadata accurately describes them.
    pub unsafe fn invoke(
        &self,
        input: &FFI_DeviceTensor,
        output: &mut FFI_DeviceTensor,
        axis: i32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.ops.invoke)(
                self.handle,
                input as *const FFI_DeviceTensor,
                output as *mut FFI_DeviceTensor,
                axis,
            )
        };
        if rc != AmStatus::Ok as i32 {
            return Err(KernelError::from_code(rc, &self.backend));
        }
        Ok(())
    }
}

impl Drop for SoftmaxKernel {
    fn drop(&mut self) {
        unsafe { (self.ops.close)(self.handle) }
    }
}

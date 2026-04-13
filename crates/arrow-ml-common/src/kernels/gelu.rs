//! GELU kernel ABI.
//!
//! - `am_gelu_supports(dtype, device_type) -> i32`
//! - `am_gelu_open(dtype, device_type, *out_handle) -> i32`
//! - `am_gelu_invoke(handle, *input, *output) -> i32`
//! - `am_gelu_close(handle)`

use crate::backend::{AmStatus, Backend};
use crate::device_tensor::FFI_DeviceTensor;
use crate::error::{KernelError, Result};
use libloading::Library;
use std::sync::Arc;

#[repr(C)]
pub struct AmGeluKernel {
    _opaque: [u8; 0],
}

pub type AmGeluSupportsFn = unsafe extern "C" fn(dtype: i32, device_type: i32) -> i32;

pub type AmGeluOpenFn =
    unsafe extern "C" fn(dtype: i32, device_type: i32, out_handle: *mut *mut AmGeluKernel) -> i32;

pub type AmGeluInvokeFn = unsafe extern "C" fn(
    handle: *mut AmGeluKernel,
    input: *const FFI_DeviceTensor,
    output: *mut FFI_DeviceTensor,
) -> i32;

pub type AmGeluCloseFn = unsafe extern "C" fn(handle: *mut AmGeluKernel);

#[derive(Copy, Clone)]
pub struct GeluOps {
    pub supports: AmGeluSupportsFn,
    pub open: AmGeluOpenFn,
    pub invoke: AmGeluInvokeFn,
    pub close: AmGeluCloseFn,
}

unsafe impl Send for GeluOps {}
unsafe impl Sync for GeluOps {}

impl GeluOps {
    pub(crate) fn load(lib: &Library) -> Option<Self> {
        unsafe {
            Some(GeluOps {
                supports: *lib.get(b"am_gelu_supports\0").ok()?,
                open: *lib.get(b"am_gelu_open\0").ok()?,
                invoke: *lib.get(b"am_gelu_invoke\0").ok()?,
                close: *lib.get(b"am_gelu_close\0").ok()?,
            })
        }
    }

    pub fn supports_dtype(&self, dtype: i32, device_type: i32) -> bool {
        unsafe { (self.supports)(dtype, device_type) == 1 }
    }
}

pub struct GeluKernel {
    ops: GeluOps,
    handle: *mut AmGeluKernel,
    backend: Arc<Backend>,
}

unsafe impl Send for GeluKernel {}

impl GeluKernel {
    pub fn open(backend: Arc<Backend>, dtype: i32, device_type: i32) -> Result<Self> {
        let ops = backend.gelu.ok_or(KernelError::Unsupported)?;
        let mut handle: *mut AmGeluKernel = std::ptr::null_mut();
        let rc = unsafe { (ops.open)(dtype, device_type, &mut handle) };
        if rc != AmStatus::Ok as i32 || handle.is_null() {
            return Err(KernelError::from_code(rc, &backend));
        }
        Ok(GeluKernel {
            ops,
            handle,
            backend,
        })
    }

    /// # Safety
    ///
    /// `input` and `output` must be valid tensors with the same shape.
    pub unsafe fn invoke(
        &self,
        input: &FFI_DeviceTensor,
        output: &mut FFI_DeviceTensor,
    ) -> Result<()> {
        let rc = unsafe {
            (self.ops.invoke)(
                self.handle,
                input as *const FFI_DeviceTensor,
                output as *mut FFI_DeviceTensor,
            )
        };
        if rc != AmStatus::Ok as i32 {
            return Err(KernelError::from_code(rc, &self.backend));
        }
        Ok(())
    }
}

impl Drop for GeluKernel {
    fn drop(&mut self) {
        unsafe { (self.ops.close)(self.handle) }
    }
}

//! Layer norm kernel ABI.
//!
//! A backend exports the layer norm kernel as a four-symbol family:
//!
//! - `am_layernorm_supports(dtype, device_type) -> i32`
//! - `am_layernorm_open(dtype, device_type, *out_handle) -> i32`
//! - `am_layernorm_invoke(handle, *input, *gamma, *beta, *output, axis, epsilon) -> i32`
//! - `am_layernorm_close(handle)`

use crate::backend::{AmStatus, Backend};
use crate::device_tensor::FFI_DeviceTensor;
use crate::error::{KernelError, Result};
use libloading::Library;
use std::sync::Arc;

#[repr(C)]
pub struct AmLayerNormKernel {
    _opaque: [u8; 0],
}

pub type AmLayerNormSupportsFn = unsafe extern "C" fn(dtype: i32, device_type: i32) -> i32;

pub type AmLayerNormOpenFn = unsafe extern "C" fn(
    dtype: i32,
    device_type: i32,
    out_handle: *mut *mut AmLayerNormKernel,
) -> i32;

/// Compute `output = layernorm(input, gamma, beta, axis, epsilon)`.
///
/// `input` and `output` must have the same shape. `gamma` and `beta` are
/// 1D tensors of shape `[dim_size]` where `dim_size` is the size of the
/// normalized axis.
pub type AmLayerNormInvokeFn = unsafe extern "C" fn(
    handle: *mut AmLayerNormKernel,
    input: *const FFI_DeviceTensor,
    gamma: *const FFI_DeviceTensor,
    beta: *const FFI_DeviceTensor,
    output: *mut FFI_DeviceTensor,
    axis: i32,
    epsilon: f32,
) -> i32;

pub type AmLayerNormCloseFn = unsafe extern "C" fn(handle: *mut AmLayerNormKernel);

#[derive(Copy, Clone)]
pub struct LayerNormOps {
    pub supports: AmLayerNormSupportsFn,
    pub open: AmLayerNormOpenFn,
    pub invoke: AmLayerNormInvokeFn,
    pub close: AmLayerNormCloseFn,
}

unsafe impl Send for LayerNormOps {}
unsafe impl Sync for LayerNormOps {}

impl LayerNormOps {
    pub(crate) fn load(lib: &Library) -> Option<Self> {
        unsafe {
            Some(LayerNormOps {
                supports: *lib.get(b"am_layernorm_supports\0").ok()?,
                open: *lib.get(b"am_layernorm_open\0").ok()?,
                invoke: *lib.get(b"am_layernorm_invoke\0").ok()?,
                close: *lib.get(b"am_layernorm_close\0").ok()?,
            })
        }
    }

    pub fn supports_dtype(&self, dtype: i32, device_type: i32) -> bool {
        unsafe { (self.supports)(dtype, device_type) == 1 }
    }
}

pub struct LayerNormKernel {
    ops: LayerNormOps,
    handle: *mut AmLayerNormKernel,
    backend: Arc<Backend>,
}

unsafe impl Send for LayerNormKernel {}

impl LayerNormKernel {
    pub fn open(backend: Arc<Backend>, dtype: i32, device_type: i32) -> Result<Self> {
        let ops = backend.layernorm.ok_or(KernelError::Unsupported)?;
        let mut handle: *mut AmLayerNormKernel = std::ptr::null_mut();
        let rc = unsafe { (ops.open)(dtype, device_type, &mut handle) };
        if rc != AmStatus::Ok as i32 || handle.is_null() {
            return Err(KernelError::from_code(rc, &backend));
        }
        Ok(LayerNormKernel {
            ops,
            handle,
            backend,
        })
    }

    /// # Safety
    ///
    /// All tensor pointers must be valid for the duration of the call.
    pub unsafe fn invoke(
        &self,
        input: &FFI_DeviceTensor,
        gamma: &FFI_DeviceTensor,
        beta: &FFI_DeviceTensor,
        output: &mut FFI_DeviceTensor,
        axis: i32,
        epsilon: f32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.ops.invoke)(
                self.handle,
                input as *const FFI_DeviceTensor,
                gamma as *const FFI_DeviceTensor,
                beta as *const FFI_DeviceTensor,
                output as *mut FFI_DeviceTensor,
                axis,
                epsilon,
            )
        };
        if rc != AmStatus::Ok as i32 {
            return Err(KernelError::from_code(rc, &self.backend));
        }
        Ok(())
    }
}

impl Drop for LayerNormKernel {
    fn drop(&mut self) {
        unsafe { (self.ops.close)(self.handle) }
    }
}

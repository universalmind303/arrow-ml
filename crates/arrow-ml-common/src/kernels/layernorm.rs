use crate::device_tensor::FFI_DeviceTensor;

crate::define_kernel_abi! {
    kernel_name: LayerNorm,
    abi_prefix: layernorm,
    backend_field: layernorm,
    invoke_args: (
        input: *const FFI_DeviceTensor,
        gamma: *const FFI_DeviceTensor,
        beta: *const FFI_DeviceTensor,
        output: *mut FFI_DeviceTensor,
        axis: i32,
        epsilon: f32,
    ),
}

impl LayerNormKernel {
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
    ) -> crate::error::Result<()> {
        let rc = unsafe {
            (self.ops().invoke)(
                self.handle(),
                input as *const FFI_DeviceTensor,
                gamma as *const FFI_DeviceTensor,
                beta as *const FFI_DeviceTensor,
                output as *mut FFI_DeviceTensor,
                axis,
                epsilon,
            )
        };
        if rc != crate::backend::AmStatus::Ok as i32 {
            return Err(crate::error::KernelError::from_code(rc, self.backend()));
        }
        Ok(())
    }
}

use crate::device_tensor::FFI_DeviceTensor;

crate::define_kernel_abi! {
    kernel_name: Gelu,
    abi_prefix: gelu,
    backend_field: gelu,
    invoke_args: (
        input: *const FFI_DeviceTensor,
        output: *mut FFI_DeviceTensor,
    ),
}

impl GeluKernel {
    /// # Safety
    ///
    /// `input` and `output` must be valid tensors with the same shape.
    pub unsafe fn invoke(
        &self,
        input: &FFI_DeviceTensor,
        output: &mut FFI_DeviceTensor,
    ) -> crate::error::Result<()> {
        let rc = unsafe {
            (self.ops().invoke)(
                self.handle(),
                input as *const FFI_DeviceTensor,
                output as *mut FFI_DeviceTensor,
            )
        };
        if rc != crate::backend::AmStatus::Ok as i32 {
            return Err(crate::error::KernelError::from_code(rc, self.backend()));
        }
        Ok(())
    }
}

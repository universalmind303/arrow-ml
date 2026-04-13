use crate::device_tensor::FFI_DeviceTensor;

crate::define_kernel_abi! {
    kernel_name: Matmul,
    abi_prefix: matmul,
    backend_field: matmul,
    invoke_args: (
        a: *const FFI_DeviceTensor,
        b: *const FFI_DeviceTensor,
        c: *mut FFI_DeviceTensor,
    ),
}

impl MatmulKernel {
    /// # Safety
    ///
    /// The caller asserts that the underlying buffers behind `a`, `b`, and
    /// `c` are valid for the duration of the call and that the FFI tensor
    /// metadata accurately describes them.
    pub unsafe fn invoke(
        &self,
        a: &FFI_DeviceTensor,
        b: &FFI_DeviceTensor,
        c: &mut FFI_DeviceTensor,
    ) -> crate::error::Result<()> {
        let rc = unsafe {
            (self.ops().invoke)(
                self.handle(),
                a as *const FFI_DeviceTensor,
                b as *const FFI_DeviceTensor,
                c as *mut FFI_DeviceTensor,
            )
        };
        if rc != crate::backend::AmStatus::Ok as i32 {
            return Err(crate::error::KernelError::from_code(rc, self.backend()));
        }
        Ok(())
    }
}

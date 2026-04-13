//! Per-kernel ABI definitions.
//!
//! Each submodule defines the function pointer types, vtable struct, and
//! safe RAII wrapper for one kernel via [`define_kernel_abi!`]. The macro
//! also generates a marker type and [`KernelDescriptor`] impl so that
//! [`crate::BackendRegistry::get_kernel`] can find and open any kernel
//! generically.

pub mod device;
pub mod gelu;
pub mod layernorm;
pub mod matmul;
pub mod softmax;

/// Trait that connects a kernel's marker type to its vtable and handle,
/// enabling generic dispatch via [`crate::BackendRegistry::get_kernel`].
pub trait KernelDescriptor {
    type Ops: Copy + Send + Sync;
    type Handle;

    fn ops_from_backend(backend: &crate::backend::Backend) -> Option<Self::Ops>;
    fn supports(ops: &Self::Ops, dtype: i32, device_type: i32) -> bool;
    fn open(
        backend: std::sync::Arc<crate::backend::Backend>,
        dtype: i32,
        device_type: i32,
    ) -> crate::error::Result<Self::Handle>;
}

/// Generates the full ABI scaffolding for a kernel: opaque handle, function
/// pointer types, vtable with `load()`, RAII wrapper with `open()`/`Drop`,
/// a marker type, and a [`KernelDescriptor`] impl.
///
/// The `invoke()` method is left for the caller to implement — it's the only
/// part that varies per kernel.
///
/// # Example
///
/// ```ignore
/// define_kernel_abi! {
///     kernel_name: Gelu,
///     abi_prefix: gelu,
///     backend_field: gelu,
///     invoke_args: (
///         input: *const FFI_DeviceTensor,
///         output: *mut FFI_DeviceTensor,
///     ),
/// }
/// ```
///
/// This generates: `Gelu` (marker type), `AmGeluKernel` (opaque handle),
/// `GeluOps` (vtable), `GeluKernel` (RAII wrapper), and all associated
/// function pointer type aliases.
#[macro_export]
macro_rules! define_kernel_abi {
    (
        kernel_name: $Name:ident,
        abi_prefix: $prefix:ident,
        backend_field: $field:ident,
        invoke_args: ( $($arg_name:ident : $arg_ty:ty),* $(,)? ),
    ) => {
        ::paste::paste! {
            // --- Marker type for generic dispatch ---
            pub struct $Name;

            // --- Opaque handle ---
            #[repr(C)]
            pub struct [<Am $Name Kernel>] {
                _opaque: [u8; 0],
            }

            // --- Function pointer types ---
            pub type [<Am $Name SupportsFn>] =
                unsafe extern "C" fn(dtype: i32, device_type: i32) -> i32;

            pub type [<Am $Name OpenFn>] =
                unsafe extern "C" fn(
                    dtype: i32,
                    device_type: i32,
                    out_handle: *mut *mut [<Am $Name Kernel>],
                ) -> i32;

            pub type [<Am $Name InvokeFn>] =
                unsafe extern "C" fn(
                    handle: *mut [<Am $Name Kernel>],
                    $($arg_name : $arg_ty),*
                ) -> i32;

            pub type [<Am $Name CloseFn>] =
                unsafe extern "C" fn(handle: *mut [<Am $Name Kernel>]);

            // --- Vtable ---
            #[derive(Copy, Clone)]
            pub struct [<$Name Ops>] {
                pub supports: [<Am $Name SupportsFn>],
                pub open: [<Am $Name OpenFn>],
                pub invoke: [<Am $Name InvokeFn>],
                pub close: [<Am $Name CloseFn>],
            }

            unsafe impl Send for [<$Name Ops>] {}
            unsafe impl Sync for [<$Name Ops>] {}

            impl [<$Name Ops>] {
                pub(crate) fn load(lib: &::libloading::Library) -> Option<Self> {
                    unsafe {
                        Some([<$Name Ops>] {
                            supports: *lib.get(
                                concat!("am_", stringify!($prefix), "_supports\0").as_bytes()
                            ).ok()?,
                            open: *lib.get(
                                concat!("am_", stringify!($prefix), "_open\0").as_bytes()
                            ).ok()?,
                            invoke: *lib.get(
                                concat!("am_", stringify!($prefix), "_invoke\0").as_bytes()
                            ).ok()?,
                            close: *lib.get(
                                concat!("am_", stringify!($prefix), "_close\0").as_bytes()
                            ).ok()?,
                        })
                    }
                }

                pub fn supports_dtype(&self, dtype: i32, device_type: i32) -> bool {
                    unsafe { (self.supports)(dtype, device_type) == 1 }
                }
            }

            // --- RAII wrapper ---
            pub struct [<$Name Kernel>] {
                ops: [<$Name Ops>],
                handle: *mut [<Am $Name Kernel>],
                backend: ::std::sync::Arc<$crate::backend::Backend>,
            }

            unsafe impl Send for [<$Name Kernel>] {}

            impl [<$Name Kernel>] {
                pub fn open(
                    backend: ::std::sync::Arc<$crate::backend::Backend>,
                    dtype: i32,
                    device_type: i32,
                ) -> $crate::error::Result<Self> {
                    let ops = backend.$field
                        .ok_or($crate::error::KernelError::Unsupported)?;
                    let mut handle: *mut [<Am $Name Kernel>] = ::std::ptr::null_mut();
                    let rc = unsafe { (ops.open)(dtype, device_type, &mut handle) };
                    if rc != $crate::backend::AmStatus::Ok as i32 || handle.is_null() {
                        return Err($crate::error::KernelError::from_code(rc, &backend));
                    }
                    Ok([<$Name Kernel>] { ops, handle, backend })
                }

                #[doc(hidden)]
                pub fn handle(&self) -> *mut [<Am $Name Kernel>] { self.handle }
                #[doc(hidden)]
                pub fn ops(&self) -> &[<$Name Ops>] { &self.ops }
                #[doc(hidden)]
                pub fn backend(&self) -> &$crate::backend::Backend { &self.backend }
            }

            impl Drop for [<$Name Kernel>] {
                fn drop(&mut self) {
                    unsafe { (self.ops.close)(self.handle) }
                }
            }

            // --- KernelDescriptor impl ---
            impl $crate::kernels::KernelDescriptor for $Name {
                type Ops = [<$Name Ops>];
                type Handle = [<$Name Kernel>];

                fn ops_from_backend(backend: &$crate::backend::Backend) -> Option<Self::Ops> {
                    backend.$field
                }

                fn supports(ops: &Self::Ops, dtype: i32, device_type: i32) -> bool {
                    ops.supports_dtype(dtype, device_type)
                }

                fn open(
                    backend: ::std::sync::Arc<$crate::backend::Backend>,
                    dtype: i32,
                    device_type: i32,
                ) -> $crate::error::Result<Self::Handle> {
                    [<$Name Kernel>]::open(backend, dtype, device_type)
                }
            }
        }
    };
}

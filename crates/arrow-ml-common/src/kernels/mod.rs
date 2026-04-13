//! Per-kernel ABI definitions.
//!
//! Each submodule defines the function pointer types, vtable struct, and
//! safe RAII wrapper for one kernel. Adding a new kernel = a new submodule
//! plus a field on [`crate::backend::Backend`] plus a vtable load call in
//! [`crate::backend::Backend::load`].
//!
//! There is intentionally no central kernel-name registry and no generic
//! dispatch indirection — each kernel is its own typed entry point.

pub mod device;
pub mod gelu;
pub mod layernorm;
pub mod matmul;
pub mod softmax;

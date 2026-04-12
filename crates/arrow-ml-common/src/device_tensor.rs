//! FFI types for the backend ABI.
//!
//! These mirror the Rust-side `DeviceBuffer` and `DeviceTensor` from
//! `arrow-ml-core` but are `#[repr(C)]` for crossing the dylib boundary.
//! Lifetime management is the caller's responsibility — the FFI types are
//! non-owning views into memory managed by the Rust side.

use std::ffi::c_void;

/// Arrow C Device Data Interface device type codes.
///
/// Numeric values match the upstream Arrow spec exactly so we can interop
/// with foreign Arrow consumers later if we need to.
#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AmDeviceType {
    Cpu = 1,
    Cuda = 2,
    CudaHost = 3,
    OpenCl = 4,
    Vulkan = 7,
    Metal = 8,
    Vpi = 9,
    Rocm = 10,
    RocmHost = 11,
    ExtDev = 12,
    CudaManaged = 13,
    OneApi = 14,
    WebGpu = 15,
    Hexagon = 16,
}

/// Arrow primitive type codes used for the tensor `dtype` field.
///
/// Mirrors the IPC type IDs so we don't invent yet another enum.
pub mod dtype {
    pub const BOOL: i32 = 1;
    pub const INT8: i32 = 2;
    pub const INT16: i32 = 3;
    pub const INT32: i32 = 4;
    pub const INT64: i32 = 5;
    pub const UINT8: i32 = 6;
    pub const UINT16: i32 = 7;
    pub const UINT32: i32 = 8;
    pub const UINT64: i32 = 9;
    pub const FLOAT16: i32 = 10;
    pub const FLOAT32: i32 = 11;
    pub const FLOAT64: i32 = 12;
    pub const BFLOAT16: i32 = 13;
}

/// FFI representation of a device-aware byte buffer. Mirrors
/// `arrow_ml_core::buffer::DeviceBuffer`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FFI_DeviceBuffer {
    pub data: *mut c_void,
    pub byte_length: u64,
    pub device_type: i32,
    pub _pad: i32,
    pub device_id: i64,
}

/// FFI representation of a device-aware N-D tensor. Mirrors
/// `arrow_ml_core::tensor::DeviceTensor`.
///
/// `shape` and `strides` are caller-owned pointers to arrays of length
/// `ndim`. They must remain valid for the lifetime of this struct.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FFI_DeviceTensor {
    pub buffer: FFI_DeviceBuffer,
    pub dtype: i32,
    pub ndim: i32,
    pub shape: *const i64,
    pub strides: *const i64,
}

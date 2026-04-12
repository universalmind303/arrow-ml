//! Device memory management ABI.
//!
//! Three function pointer types let backends allocate, free, and copy
//! device-resident memory through a uniform C ABI. These are infrastructure
//! — not a kernel family — and are **required** of every backend: a plugin
//! that can't allocate device memory can't be a backend.
//!
//! The pointers live directly on [`crate::backend::Backend`] as
//! non-`Option` fields and are looked up at load time.

use std::ffi::c_void;

/// Allocate `nbytes` of device-resident memory on `(device_type, device_id)`.
///
/// On success, writes the resulting pointer to `*out_ptr` and returns
/// [`crate::backend::AM_OK`]. On failure (OOM, invalid device, etc.)
/// writes NULL and returns a negative error code.
pub type AmDeviceAllocFn = unsafe extern "C" fn(
    device_type: i32,
    device_id: i64,
    nbytes: u64,
    out_ptr: *mut *mut c_void,
) -> i32;

/// Free a device allocation previously returned by [`AmDeviceAllocFn`].
///
/// `nbytes` is the size that was passed to the matching `alloc` call.
pub type AmDeviceFreeFn = unsafe extern "C" fn(ptr: *mut c_void, nbytes: u64);

/// Copy `nbytes` of memory between two device locations.
///
/// `src` lives on `src_dev`, `dst` lives on `dst_dev`. Either side may be
/// the host (`AmDeviceType::Cpu as i32`). The backend dispatches to the
/// correct host/device, device/host, or device/device path internally.
pub type AmDeviceCopyFn = unsafe extern "C" fn(
    src: *const c_void,
    src_dev: i32,
    dst: *mut c_void,
    dst_dev: i32,
    nbytes: u64,
) -> i32;

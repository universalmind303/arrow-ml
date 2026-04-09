//! FFI tensor type for the v2 backend ABI.
//!
//! `FFI_TensorArray` is a dense N-dimensional tensor laid out for crossing
//! the C ABI boundary. It composes over `arrow::ffi::FFI_ArrowArray` so that
//! the underlying buffer carries an Arrow release callback for memory
//! management, while the surrounding fields add the dtype/shape/strides/device
//! metadata that ML kernels need.
//!
//! The data buffer lives in `array.buffers[1]` per the standard Arrow
//! primitive layout (`buffers[0]` is the validity bitmap and is always NULL
//! for dense tensors, which have no nulls).

use arrow::ffi::FFI_ArrowArray;
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

/// FFI representation of an N-dimensional Arrow tensor on a specific device.
///
/// Composed over `FFI_ArrowArray` for data buffer lifetime management: the
/// embedded array's release callback is responsible for the underlying
/// Arrow buffer(s). The `shape` and `strides` pointers are owned separately
/// by the producer-side wrapper and must remain valid for the lifetime of
/// that wrapper (for example, [`crate::host_tensor::OwnedFFITensor`] owns
/// them as boxed slices).
#[repr(C)]
pub struct FFI_TensorArray {
    /// Standard Arrow C Data Interface array. Offset 0, downcastable to
    /// `*FFI_ArrowArray` for code that doesn't need tensor metadata.
    /// Layout: `n_buffers = 2`, `buffers[0] = NULL`, `buffers[1] = data ptr`.
    pub array: FFI_ArrowArray,

    // ----- tensor metadata -----
    /// Arrow primitive dtype code (see [`dtype`]).
    pub dtype: i32,
    /// Number of dimensions.
    pub ndim: i32,
    /// Length-`ndim` array of dimension sizes.
    pub shape: *const i64,
    /// Length-`ndim` array of byte strides per dimension. Matches the Arrow
    /// IPC Tensor convention (strides in bytes, not elements).
    pub strides: *const i64,

    // ----- device metadata -----
    /// Which device this tensor lives on. See [`AmDeviceType`].
    pub device_type: i32,
    /// Padding to keep `device_id` 8-byte aligned.
    pub _pad: i32,
    /// Device ordinal (CUDA device index, etc.). -1 for CPU.
    pub device_id: i64,
    /// Producer-side ready signal. **In v2 always NULL.** Reserved for
    /// future async support.
    pub sync_event: *mut c_void,
    /// Reserved for future spec evolution. Must be zeroed.
    pub reserved: [i64; 3],
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn ffi_tensor_array_layout() {
        // FFI_ArrowArray is the upstream Arrow C Data Interface struct.
        // Whatever its size is, our tensor type embeds it at offset 0.
        assert_eq!(offset_of!(FFI_TensorArray, array), 0);

        // The metadata block follows the embedded array. We assert
        // relative ordering rather than absolute offsets so the test
        // doesn't break if upstream FFI_ArrowArray grows.
        let off_dtype = offset_of!(FFI_TensorArray, dtype);
        let off_ndim = offset_of!(FFI_TensorArray, ndim);
        let off_shape = offset_of!(FFI_TensorArray, shape);
        let off_strides = offset_of!(FFI_TensorArray, strides);
        let off_device_type = offset_of!(FFI_TensorArray, device_type);
        let off_pad = offset_of!(FFI_TensorArray, _pad);
        let off_device_id = offset_of!(FFI_TensorArray, device_id);
        let off_sync_event = offset_of!(FFI_TensorArray, sync_event);
        let off_reserved = offset_of!(FFI_TensorArray, reserved);

        assert!(off_dtype >= size_of::<FFI_ArrowArray>());
        assert_eq!(off_ndim, off_dtype + size_of::<i32>());
        assert_eq!(off_shape % align_of::<*const i64>(), 0);
        assert!(off_shape >= off_ndim + size_of::<i32>());
        assert_eq!(off_strides, off_shape + size_of::<*const i64>());
        assert_eq!(off_device_type, off_strides + size_of::<*const i64>());
        assert_eq!(off_pad, off_device_type + size_of::<i32>());
        assert_eq!(off_device_id % 8, 0);
        assert_eq!(off_device_id, off_pad + size_of::<i32>());
        assert_eq!(off_sync_event, off_device_id + size_of::<i64>());
        assert_eq!(off_reserved, off_sync_event + size_of::<*mut c_void>());
        assert_eq!(
            size_of::<FFI_TensorArray>(),
            off_reserved + 3 * size_of::<i64>()
        );

        assert_eq!(align_of::<FFI_TensorArray>() % 8, 0);
    }

    #[test]
    fn am_device_type_values() {
        assert_eq!(AmDeviceType::Cpu as i32, 1);
        assert_eq!(AmDeviceType::Cuda as i32, 2);
        assert_eq!(AmDeviceType::Metal as i32, 8);
    }
}

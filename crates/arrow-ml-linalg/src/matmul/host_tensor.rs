//! Host-staging adapter: wraps a Rust slice (or fresh allocation) into an
//! [`FFI_TensorArray`] that linalg can hand to a backend's matmul kernel.
//!
//! ## Design note vs. the v2 plan
//!
//! The plan's "implementation gotcha" section described two approaches for
//! managing the shape/strides storage relative to the embedded
//! `FFI_ArrowArray`'s release callback:
//!
//! - (a) Stash the shape/strides inside `FFI_ArrowArray.private_data` and
//!   replace the release callback.
//! - (b) Build the `FFI_ArrowArray` by hand with our own release callback
//!   that owns everything.
//!
//! Both options require touching `FFI_ArrowArray`'s private fields, which
//! arrow-rs does not expose. Instead we use a third option that lives
//! entirely outside `FFI_ArrowArray`: an [`OwnedHostTensor`] wrapper struct
//! that owns both the FFI tensor and the shape/strides boxes side-by-side.
//! Drop order (declaration order in Rust) frees the FFI tensor first
//! (firing the arrow-generated release callback for the buffer) and then
//! the shape/strides boxes.
//!
//! This is sound because the FFI tensor never escapes the wrapper —
//! [`OwnedHostTensor::as_ffi`] hands out borrowed references whose lifetime
//! is tied to `&self`, and the backend is required by the v2 ABI to treat
//! the tensor as a borrow that does not outlive the call.

use arrow::array::ArrayData;
use arrow::buffer::Buffer;
use arrow::datatypes::DataType;
use arrow::ffi::FFI_ArrowArray;
use arrow_ml_common::device_tensor::{dtype, AmDeviceType, FFI_TensorArray};

/// A host-resident tensor wrapper that owns its data buffer, its
/// shape/strides boxes, and the [`FFI_TensorArray`] that points into them.
///
/// Always lives on the CPU device. Always rank-2 in v2 (matmul is the only
/// caller); generalize when a second kernel needs higher rank.
pub(crate) struct OwnedHostTensor {
    /// The FFI tensor handed to backends. Declared first so it drops
    /// first — firing the arrow release callback for the data buffer
    /// before we free `_shape` / `_strides`.
    ffi: FFI_TensorArray,
    /// Backing storage for `ffi.shape`. Lives as long as `self`.
    _shape: Box<[i64]>,
    /// Backing storage for `ffi.strides`. Lives as long as `self`.
    _strides: Box<[i64]>,
}

impl OwnedHostTensor {
    /// Build a row-major host tensor from an `f32` slice.
    ///
    /// The slice is copied into a fresh `Vec<f32>` (the host-staging copy
    /// the backend later uploads to the GPU). `shape = [rows, cols]`,
    /// `strides = [cols * 4, 4]` in bytes.
    pub(crate) fn from_f32_slice(data: &[f32], rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        let owned: Vec<f32> = data.to_vec();
        Self::from_f32_vec(owned, rows, cols)
    }

    /// Build a row-major host tensor that owns the given `Vec<f32>`.
    ///
    /// Used for the output tensor where linalg pre-allocates the result
    /// buffer and hands it to the backend to write into.
    pub(crate) fn from_f32_vec(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        let buf = Buffer::from_vec(data);
        let array_data = ArrayData::builder(DataType::Float32)
            .len(rows * cols)
            .add_buffer(buf)
            .build()
            .expect("ArrayData::build for Float32 host tensor");
        let array = FFI_ArrowArray::new(&array_data);

        let shape: Box<[i64]> = Box::new([rows as i64, cols as i64]);
        let strides: Box<[i64]> = Box::new([(cols * std::mem::size_of::<f32>()) as i64, std::mem::size_of::<f32>() as i64]);

        let ffi = FFI_TensorArray {
            array,
            dtype: dtype::FLOAT32,
            ndim: 2,
            shape: shape.as_ptr(),
            strides: strides.as_ptr(),
            device_type: AmDeviceType::Cpu as i32,
            _pad: 0,
            device_id: -1,
            sync_event: std::ptr::null_mut(),
            reserved: [0; 3],
        };

        OwnedHostTensor {
            ffi,
            _shape: shape,
            _strides: strides,
        }
    }

    /// Borrow the FFI tensor for a backend call.
    pub(crate) fn as_ffi(&self) -> &FFI_TensorArray {
        &self.ffi
    }

    /// Mutably borrow the FFI tensor (for output buffers).
    pub(crate) fn as_ffi_mut(&mut self) -> &mut FFI_TensorArray {
        &mut self.ffi
    }

    /// Read the data buffer back as an `f32` slice.
    ///
    /// Reads through `array.buffer(1)` (the values buffer; index 0 is the
    /// always-NULL validity bitmap inserted by `FFI_ArrowArray::new` for
    /// types that can contain a null mask).
    pub(crate) fn as_f32_slice(&self) -> &[f32] {
        let n = (self._shape[0] * self._shape[1]) as usize;
        let ptr = self.ffi.array.buffer(1) as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n) }
    }
}

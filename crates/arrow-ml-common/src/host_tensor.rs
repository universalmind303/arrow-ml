//! `OwnedFFITensor`: ownership wrapper that bridges an `arrow::tensor::Tensor`
//! into an [`FFI_TensorArray`] for the v2 backend ABI.
//!
//! ## Why this wrapper exists
//!
//! [`FFI_TensorArray::shape`] and [`FFI_TensorArray::strides`] are
//! `*const i64` raw pointers. `arrow::tensor::Tensor` stores its shape and
//! strides as `Vec<usize>` instead. The two layouts can't share storage,
//! and arrow-rs's [`FFI_ArrowArray`] doesn't expose its `private_data`
//! field, so there is no way to attach the converted i64 boxes to the
//! embedded array. Something has to allocate the boxes and keep them alive
//! for the duration of the kernel call — that's this wrapper's only job.
//!
//! Drop order matters: declaring `ffi` first means it's released first
//! (firing the arrow C Data Interface release callback for the data
//! buffer) before `_shape` / `_strides` are freed.
//!
//! ## Zero-copy semantics
//!
//! The data buffer is *not* copied. `arrow::buffer::Buffer` is `Arc`-backed,
//! so cloning it during the conversion just bumps a reference count. The
//! wrapper and the original `Tensor` share the same backing memory; a
//! kernel writing into the wrapper's FFI tensor mutates memory that's
//! visible through the original `Tensor` once the kernel returns.

use crate::device_tensor::{dtype, AmDeviceType, FFI_TensorArray};
use arrow::array::ArrayData;
use arrow::datatypes::{ArrowPrimitiveType, DataType};
use arrow::ffi::FFI_ArrowArray;
use arrow::tensor::Tensor;

/// Owns the backing storage (arrow buffer reference + shape/strides boxes)
/// for an [`FFI_TensorArray`] that can be handed to a backend's kernel
/// invoke function.
pub struct OwnedFFITensor {
    ffi: FFI_TensorArray,
    _shape: Box<[i64]>,
    _strides: Box<[i64]>,
}

impl OwnedFFITensor {
    /// Borrow the FFI tensor to pass to a backend kernel.
    pub fn as_ffi(&self) -> &FFI_TensorArray {
        &self.ffi
    }

    /// Mutably borrow the FFI tensor (for output buffers a kernel will
    /// write into).
    pub fn as_ffi_mut(&mut self) -> &mut FFI_TensorArray {
        &mut self.ffi
    }
}

impl<T: ArrowPrimitiveType> From<&Tensor<'_, T>> for OwnedFFITensor {
    fn from(tensor: &Tensor<'_, T>) -> Self {
        let shape_usize = tensor
            .shape()
            .cloned()
            .expect("OwnedFFITensor: tensor has no shape (0-d scalar tensors not supported)");
        let strides_usize = tensor
            .strides()
            .cloned()
            .expect("OwnedFFITensor: tensor has no strides");

        let shape: Box<[i64]> = shape_usize.iter().map(|&d| d as i64).collect();
        let strides: Box<[i64]> = strides_usize.iter().map(|&s| s as i64).collect();
        let ndim = shape.len() as i32;
        let dtype_code = am_dtype_code(&T::DATA_TYPE);
        let total_len: usize = shape_usize.iter().product();

        let array_data = ArrayData::builder(T::DATA_TYPE)
            .len(total_len)
            .add_buffer(tensor.data().clone())
            .build()
            .expect("ArrayData::build for primitive tensor");
        let array = FFI_ArrowArray::new(&array_data);

        let ffi = FFI_TensorArray {
            array,
            dtype: dtype_code,
            ndim,
            shape: shape.as_ptr(),
            strides: strides.as_ptr(),
            device_type: AmDeviceType::Cpu as i32,
            _pad: 0,
            device_id: -1,
            sync_event: std::ptr::null_mut(),
            reserved: [0; 3],
        };

        OwnedFFITensor {
            ffi,
            _shape: shape,
            _strides: strides,
        }
    }
}

/// Map an arrow `DataType` to its `dtype::*` integer code on the v2 ABI.
///
/// Panics on unsupported data types. The supported set covers every
/// numeric primitive arrow exposes; non-numeric types (Date/Time/Interval/
/// Decimal/Boolean) are intentionally not mapped because ML kernels do
/// not consume them and the panic message will surface the gap clearly
/// if a future caller passes one.
fn am_dtype_code(data_type: &DataType) -> i32 {
    match data_type {
        DataType::Int8 => dtype::INT8,
        DataType::Int16 => dtype::INT16,
        DataType::Int32 => dtype::INT32,
        DataType::Int64 => dtype::INT64,
        DataType::UInt8 => dtype::UINT8,
        DataType::UInt16 => dtype::UINT16,
        DataType::UInt32 => dtype::UINT32,
        DataType::UInt64 => dtype::UINT64,
        DataType::Float16 => dtype::FLOAT16,
        DataType::Float32 => dtype::FLOAT32,
        DataType::Float64 => dtype::FLOAT64,
        other => panic!("OwnedFFITensor: unsupported arrow DataType {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::{Float32Type, Int8Type};

    #[test]
    fn from_f32_tensor_round_trip() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let buf = ScalarBuffer::<f32>::from(data.clone()).into_inner();
        let tensor =
            Tensor::<Float32Type>::new_row_major(buf, Some(vec![3, 4]), None).unwrap();

        let owned = OwnedFFITensor::from(&tensor);
        let ffi = owned.as_ffi();

        assert_eq!(ffi.dtype, dtype::FLOAT32);
        assert_eq!(ffi.ndim, 2);
        assert_eq!(ffi.device_type, AmDeviceType::Cpu as i32);

        let shape = unsafe { std::slice::from_raw_parts(ffi.shape, 2) };
        assert_eq!(shape, &[3i64, 4]);

        let strides = unsafe { std::slice::from_raw_parts(ffi.strides, 2) };
        // row-major byte strides: [4 * 4, 4]
        assert_eq!(strides, &[16i64, 4]);

        let read_back =
            unsafe { std::slice::from_raw_parts(ffi.array.buffer(1) as *const f32, 12) };
        assert_eq!(read_back, data.as_slice());
    }

    #[test]
    fn from_i8_tensor_dtype_code() {
        let data: Vec<i8> = (0..6).collect();
        let buf = ScalarBuffer::<i8>::from(data).into_inner();
        let tensor =
            Tensor::<Int8Type>::new_row_major(buf, Some(vec![2, 3]), None).unwrap();

        let owned = OwnedFFITensor::from(&tensor);
        assert_eq!(owned.as_ffi().dtype, dtype::INT8);
    }
}

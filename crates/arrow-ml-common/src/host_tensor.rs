//! `OwnedFFITensor`: ownership wrapper that bridges an arrow buffer into an
//! [`FFI_TensorArray`] for the v2 backend ABI.
//!
//! ## Why this wrapper exists
//!
//! [`FFI_TensorArray::shape`] and [`FFI_TensorArray::strides`] are
//! `*const i64` raw pointers. Arrow's `Tensor` stores its shape and strides
//! as `Vec<usize>` instead. The two layouts can't share storage, and
//! arrow-rs's [`FFI_ArrowArray`] doesn't expose its `private_data` field, so
//! there is no way to attach the converted i64 boxes to the embedded array.
//! Something has to allocate the boxes and keep them alive for the duration
//! of the kernel call — that's this wrapper's only job.
//!
//! Drop order matters: declaring `ffi` first means it's released first
//! (firing the arrow C Data Interface release callback for the data
//! buffer) before `_shape` / `_strides` are freed.
//!
//! ## Zero-copy semantics
//!
//! The data buffer is *not* copied. `arrow::buffer::Buffer` is `Arc`-backed,
//! so cloning it during the conversion just bumps a reference count. The
//! wrapper and the producer share the same backing memory; a kernel writing
//! into the wrapper's FFI tensor mutates memory that's visible through the
//! producer-side `Buffer` once the kernel returns.

use crate::device_tensor::{dtype, AmDeviceType, FFI_TensorArray};
use crate::error::{KernelError, Result};
use arrow::array::ArrayData;
use arrow::buffer::Buffer;
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

    /// Build an `OwnedFFITensor` from an owned arrow [`Buffer`] plus explicit
    /// tensor metadata.
    ///
    /// The buffer is moved into the embedded [`FFI_ArrowArray`] (which clones
    /// its underlying `Arc` internally). Callers who want to recover the
    /// original allocation via [`Buffer::into_vec`] after the kernel writes
    /// should hold their own clone of the buffer and drop this wrapper
    /// before attempting the recovery.
    pub fn from_buffer(
        buf: Buffer,
        data_type: &DataType,
        shape: &[usize],
        strides: &[usize],
    ) -> Result<Self> {
        if shape.is_empty() {
            return Err(KernelError::InvalidArgument(
                "OwnedFFITensor: 0-d scalar tensors are not supported".to_string(),
            ));
        }
        if strides.len() != shape.len() {
            return Err(KernelError::InvalidArgument(format!(
                "OwnedFFITensor: strides len {} != shape len {}",
                strides.len(),
                shape.len()
            )));
        }

        let dtype_code = am_dtype_code(data_type)?;
        let shape_box: Box<[i64]> = shape.iter().map(|&d| d as i64).collect();
        let strides_box: Box<[i64]> = strides.iter().map(|&s| s as i64).collect();
        let ndim = shape_box.len() as i32;
        let total_len: usize = shape.iter().product();

        let array_data = ArrayData::builder(data_type.clone())
            .len(total_len)
            .add_buffer(buf)
            .build()
            .map_err(KernelError::from)?;
        let array = FFI_ArrowArray::new(&array_data);

        let ffi = FFI_TensorArray {
            array,
            dtype: dtype_code,
            ndim,
            shape: shape_box.as_ptr(),
            strides: strides_box.as_ptr(),
            device_type: AmDeviceType::Cpu as i32,
            _pad: 0,
            device_id: -1,
            sync_event: std::ptr::null_mut(),
            reserved: [0; 3],
        };

        Ok(OwnedFFITensor {
            ffi,
            _shape: shape_box,
            _strides: strides_box,
        })
    }
}

impl<T: ArrowPrimitiveType> TryFrom<&Tensor<'_, T>> for OwnedFFITensor {
    type Error = KernelError;

    fn try_from(tensor: &Tensor<'_, T>) -> Result<Self> {
        let shape = tensor.shape().ok_or_else(|| {
            KernelError::InvalidArgument(
                "OwnedFFITensor: tensor has no shape (0-d scalar tensors not supported)"
                    .to_string(),
            )
        })?;

        // Arrow Tensor strides are optional even when shape is set. Fall back
        // to computing row-major byte strides from the shape so callers with
        // a sanely-constructed row-major tensor don't have to care.
        let fallback_strides;
        let strides: &[usize] = match tensor.strides() {
            Some(s) => s.as_slice(),
            None => {
                let elem_size = T::DATA_TYPE.primitive_width().ok_or_else(|| {
                    KernelError::InvalidArgument(format!(
                        "OwnedFFITensor: dtype {:?} has no fixed element width",
                        T::DATA_TYPE
                    ))
                })?;
                fallback_strides = row_major_byte_strides(shape.as_slice(), elem_size);
                &fallback_strides
            }
        };

        OwnedFFITensor::from_buffer(
            tensor.data().clone(),
            &T::DATA_TYPE,
            shape.as_slice(),
            strides,
        )
    }
}

/// Compute row-major byte strides for a dense tensor with the given shape
/// and element size.
fn row_major_byte_strides(shape: &[usize], elem_size: usize) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    strides[shape.len() - 1] = elem_size;
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Map an arrow `DataType` to its `dtype::*` integer code on the v2 ABI.
///
/// Returns [`KernelError::InvalidArgument`] for types that aren't one of the
/// numeric primitives ML kernels consume (Date/Time/Interval/Decimal/Boolean
/// and composite types). Falls through to an error rather than panicking so
/// callers can fall back to a CPU implementation or surface a clean failure.
fn am_dtype_code(data_type: &DataType) -> Result<i32> {
    Ok(match data_type {
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
        other => {
            return Err(KernelError::InvalidArgument(format!(
                "OwnedFFITensor: unsupported arrow DataType {other:?}"
            )))
        }
    })
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
        let tensor = Tensor::<Float32Type>::new_row_major(buf, Some(vec![3, 4]), None).unwrap();

        let owned = OwnedFFITensor::try_from(&tensor).unwrap();
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
        let tensor = Tensor::<Int8Type>::new_row_major(buf, Some(vec![2, 3]), None).unwrap();

        let owned = OwnedFFITensor::try_from(&tensor).unwrap();
        assert_eq!(owned.as_ffi().dtype, dtype::INT8);
    }

    #[test]
    fn from_buffer_direct() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buf = Buffer::from_vec(data.clone());

        let owned = OwnedFFITensor::from_buffer(
            buf,
            &DataType::Float32,
            &[2, 3],
            &row_major_byte_strides(&[2, 3], 4),
        )
        .unwrap();

        let ffi = owned.as_ffi();
        assert_eq!(ffi.ndim, 2);
        assert_eq!(ffi.dtype, dtype::FLOAT32);

        let shape = unsafe { std::slice::from_raw_parts(ffi.shape, 2) };
        assert_eq!(shape, &[2i64, 3]);

        let read_back = unsafe { std::slice::from_raw_parts(ffi.array.buffer(1) as *const f32, 6) };
        assert_eq!(read_back, data.as_slice());
    }

    #[test]
    fn unsupported_dtype_returns_error() {
        let err = am_dtype_code(&DataType::Boolean).unwrap_err();
        assert!(matches!(err, KernelError::InvalidArgument(_)));
    }
}

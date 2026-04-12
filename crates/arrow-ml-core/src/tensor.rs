use crate::buffer::DeviceBuffer;
use crate::device::Device;
use crate::error::{DeviceError, Result};
use arrow::datatypes::{ArrowPrimitiveType, DataType};
use arrow::tensor::Tensor as ArrowTensor;
use arrow_ml_common::device_tensor::{dtype, FFI_DeviceBuffer, FFI_DeviceTensor};
use std::ffi::c_void;

/// N-D dense tensor on some device. Mirrors [`arrow::tensor::Tensor`] —
/// wraps a [`DeviceBuffer`] directly with shape and strides, no nulls.
#[derive(Debug, Clone)]
pub struct Tensor {
    data_type: DataType,
    buffer: DeviceBuffer,
    shape: Option<Vec<usize>>,
    strides: Option<Vec<usize>>,
}

impl Tensor {
    pub fn new(
        data_type: DataType,
        buffer: DeviceBuffer,
        shape: Option<Vec<usize>>,
        strides: Option<Vec<usize>>,
    ) -> Self {
        if let (Some(s), Some(st)) = (&shape, &strides) {
            assert_eq!(
                s.len(),
                st.len(),
                "shape and strides must have the same number of dimensions"
            );
        }
        Self {
            data_type,
            buffer,
            shape,
            strides,
        }
    }

    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }

    pub fn buffer(&self) -> &DeviceBuffer {
        &self.buffer
    }

    pub fn into_buffer(self) -> DeviceBuffer {
        self.buffer
    }

    pub fn shape(&self) -> Option<&[usize]> {
        self.shape.as_deref()
    }

    pub fn strides(&self) -> Option<&[usize]> {
        self.strides.as_deref()
    }

    pub fn ndim(&self) -> usize {
        self.shape.as_ref().map_or(0, |s| s.len())
    }

    pub fn device(&self) -> Device {
        self.buffer.device()
    }

    pub fn size(&self) -> usize {
        self.shape
            .as_ref()
            .map_or(0, |s| s.iter().copied().product())
    }

    pub fn to(&self, device: Device) -> Self {
        Self {
            data_type: self.data_type.clone(),
            buffer: self.buffer.to(device),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}

impl Tensor {
    pub fn as_ffi(&self) -> FFI_DeviceTensor {
        let shape = self.shape.as_deref().unwrap_or(&[]);
        let strides = self.strides.as_deref().unwrap_or(&[]);

        FFI_DeviceTensor {
            buffer: FFI_DeviceBuffer {
                data: self.buffer.as_ptr() as *mut c_void,
                byte_length: self.buffer.len() as u64,
                device_type: self.buffer.device().to_am() as i32,
                _pad: 0,
                device_id: self.buffer.device().id(),
            },
            dtype: am_dtype_code(&self.data_type),
            ndim: shape.len() as i32,
            shape: shape.as_ptr() as *const i64,
            strides: strides.as_ptr() as *const i64,
        }
    }
}

fn am_dtype_code(data_type: &DataType) -> i32 {
    match data_type {
        DataType::Boolean => dtype::BOOL,
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
        _ => -1,
    }
}

impl<T: ArrowPrimitiveType> From<ArrowTensor<'_, T>> for Tensor {
    fn from(tensor: ArrowTensor<'_, T>) -> Self {
        Self {
            data_type: tensor.data_type().clone(),
            buffer: DeviceBuffer::from(tensor.data().clone()),
            shape: tensor.shape().cloned(),
            strides: tensor.strides().cloned(),
        }
    }
}

impl<T: ArrowPrimitiveType> TryFrom<Tensor> for ArrowTensor<'static, T> {
    type Error = DeviceError;

    fn try_from(tensor: Tensor) -> Result<Self> {
        let buffer: arrow::buffer::Buffer = tensor.buffer.try_into()?;
        ArrowTensor::try_new(buffer, tensor.shape, tensor.strides, None)
            .map_err(|_| DeviceError::CannotConvert)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Float32Type;
    use arrow_ml_common::device_tensor::AmDeviceType;

    fn f32_tensor(data: &[f32], shape: Vec<usize>) -> Tensor {
        let buf: DeviceBuffer = arrow::buffer::Buffer::from_slice_ref(data).into();
        Tensor::new(DataType::Float32, buf, Some(shape), None)
    }

    #[test]
    fn new_and_accessors() {
        let t = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(t.data_type(), &DataType::Float32);
        assert_eq!(t.shape(), Some([2, 3].as_slice()));
        assert!(t.strides().is_none());
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.size(), 6);
        assert_eq!(t.device(), Device::Cpu);
    }

    #[test]
    fn no_shape() {
        let buf: DeviceBuffer = arrow::buffer::Buffer::from_slice_ref(&[1.0f32]).into();
        let t = Tensor::new(DataType::Float32, buf, None, None);
        assert!(t.shape().is_none());
        assert!(t.strides().is_none());
        assert_eq!(t.ndim(), 0);
        assert_eq!(t.size(), 0);
    }

    #[test]
    fn with_strides() {
        let buf: DeviceBuffer =
            arrow::buffer::Buffer::from_slice_ref(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).into();
        let t = Tensor::new(DataType::Float32, buf, Some(vec![2, 3]), Some(vec![12, 4]));
        assert_eq!(t.strides(), Some([12, 4].as_slice()));
        assert_eq!(t.ndim(), 2);
    }

    #[test]
    #[should_panic(expected = "shape and strides must have the same number of dimensions")]
    fn shape_strides_dimension_mismatch() {
        let buf: DeviceBuffer = arrow::buffer::Buffer::from_slice_ref(&[1.0f32; 6]).into();
        Tensor::new(DataType::Float32, buf, Some(vec![2, 3]), Some(vec![12]));
    }

    #[test]
    fn into_buffer() {
        let buf: DeviceBuffer = arrow::buffer::Buffer::from_slice_ref(&[1.0f32, 2.0]).into();
        let buf_clone = buf.clone();
        let t = Tensor::new(DataType::Float32, buf, Some(vec![2]), None);
        let extracted = t.into_buffer();
        assert!(buf_clone.ptr_eq(&extracted));
    }

    #[test]
    fn clone_shares_buffer() {
        let t = f32_tensor(&[1.0, 2.0], vec![2]);
        let cloned = t.clone();
        assert!(t.buffer().ptr_eq(cloned.buffer()));
    }

    #[test]
    fn to_same_device_is_cheap() {
        let t = f32_tensor(&[1.0, 2.0, 3.0], vec![3]);
        let moved = t.to(Device::Cpu);
        assert!(t.buffer().ptr_eq(moved.buffer()));
        assert_eq!(moved.shape(), Some([3].as_slice()));
    }

    #[test]
    fn from_arrow_tensor_no_strides() {
        let data = arrow::buffer::Buffer::from_slice_ref(&[1.0f32, 2.0, 3.0, 4.0]);
        let arrow_t =
            ArrowTensor::<Float32Type>::try_new(data, Some(vec![2, 2]), None, None).unwrap();
        let dt = Tensor::from(arrow_t);

        assert_eq!(dt.data_type(), &DataType::Float32);
        assert_eq!(dt.shape(), Some([2, 2].as_slice()));
        assert_eq!(dt.device(), Device::Cpu);
        assert_eq!(dt.size(), 4);
    }

    #[test]
    fn from_arrow_tensor_with_strides() {
        let data = arrow::buffer::Buffer::from_slice_ref(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let arrow_t =
            ArrowTensor::<Float32Type>::try_new(data, Some(vec![2, 3]), Some(vec![12, 4]), None)
                .unwrap();
        let dt = Tensor::from(arrow_t);

        assert_eq!(dt.shape(), Some([2, 3].as_slice()));
        assert_eq!(dt.strides(), Some([12, 4].as_slice()));
    }

    #[test]
    fn try_into_arrow_tensor_roundtrip() {
        let data = arrow::buffer::Buffer::from_slice_ref(&[10.0f32, 20.0, 30.0, 40.0]);
        let arrow_t =
            ArrowTensor::<Float32Type>::try_new(data, Some(vec![2, 2]), None, None).unwrap();
        let dt = Tensor::from(arrow_t);
        let back: ArrowTensor<'static, Float32Type> = dt.try_into().unwrap();

        assert_eq!(back.shape().map(|v| v.as_slice()), Some([2, 2].as_slice()));
        let values = back.data().typed_data::<f32>();
        assert_eq!(values, &[10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn scalar_tensor() {
        let t = f32_tensor(&[42.0], vec![]);
        assert_eq!(t.ndim(), 0);
        assert_eq!(t.size(), 1);
        assert_eq!(t.shape(), Some([].as_slice()));
    }

    #[test]
    fn high_dimensional() {
        let data = vec![0.0f32; 2 * 3 * 4 * 5];
        let t = f32_tensor(&data, vec![2, 3, 4, 5]);
        assert_eq!(t.ndim(), 4);
        assert_eq!(t.size(), 120);
    }

    #[test]
    fn as_ffi_basic() {
        let t = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let ffi = t.as_ffi();

        assert_eq!(ffi.dtype, dtype::FLOAT32);
        assert_eq!(ffi.ndim, 2);
        assert_eq!(ffi.buffer.byte_length, 24);
        assert_eq!(ffi.buffer.device_type, AmDeviceType::Cpu as i32);
        assert_eq!(ffi.buffer.device_id, -1);
        assert!(!ffi.buffer.data.is_null());
        assert_eq!(ffi.buffer._pad, 0);
    }

    #[test]
    fn as_ffi_shape_pointer_valid() {
        let t = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let ffi = t.as_ffi();

        unsafe {
            assert_eq!(*ffi.shape, 2);
            assert_eq!(*ffi.shape.add(1), 3);
        }
    }

    #[test]
    fn as_ffi_strides_pointer_valid() {
        let buf: DeviceBuffer = arrow::buffer::Buffer::from_slice_ref(&[1.0f32; 6]).into();
        let t = Tensor::new(DataType::Float32, buf, Some(vec![2, 3]), Some(vec![12, 4]));
        let ffi = t.as_ffi();

        unsafe {
            assert_eq!(*ffi.strides, 12);
            assert_eq!(*ffi.strides.add(1), 4);
        }
    }

    #[test]
    fn as_ffi_dtype_codes() {
        let cases: Vec<(DataType, i32)> = vec![
            (DataType::Boolean, dtype::BOOL),
            (DataType::Int8, dtype::INT8),
            (DataType::Int16, dtype::INT16),
            (DataType::Int32, dtype::INT32),
            (DataType::Int64, dtype::INT64),
            (DataType::UInt8, dtype::UINT8),
            (DataType::UInt16, dtype::UINT16),
            (DataType::UInt32, dtype::UINT32),
            (DataType::UInt64, dtype::UINT64),
            (DataType::Float16, dtype::FLOAT16),
            (DataType::Float32, dtype::FLOAT32),
            (DataType::Float64, dtype::FLOAT64),
        ];
        for (dt, expected_code) in cases {
            let buf: DeviceBuffer = arrow::buffer::Buffer::from_vec(vec![0u8; 8]).into();
            let t = Tensor::new(dt.clone(), buf, Some(vec![1]), None);
            assert_eq!(t.as_ffi().dtype, expected_code, "dtype mismatch for {dt:?}");
        }
    }

    #[test]
    fn as_ffi_unsupported_dtype_returns_negative() {
        let buf: DeviceBuffer = arrow::buffer::Buffer::from_vec(vec![0u8; 8]).into();
        let t = Tensor::new(DataType::Utf8, buf, Some(vec![1]), None);
        assert_eq!(t.as_ffi().dtype, -1);
    }

    #[test]
    fn as_ffi_no_shape() {
        let buf: DeviceBuffer = arrow::buffer::Buffer::from_slice_ref(&[1.0f32]).into();
        let t = Tensor::new(DataType::Float32, buf, None, None);
        let ffi = t.as_ffi();
        assert_eq!(ffi.ndim, 0);
    }

    #[test]
    fn as_ffi_data_pointer_matches_buffer() {
        let t = f32_tensor(&[1.0, 2.0], vec![2]);
        let ffi = t.as_ffi();
        assert_eq!(ffi.buffer.data as *const u8, t.buffer().as_ptr());
    }

    // ---------------------------------------------------------------
    // Device-path tests (require a loaded GPU backend)
    // ---------------------------------------------------------------

    fn has_metal() -> bool {
        arrow_ml_common::BackendRegistry::global()
            .loaded_backends()
            .contains(&"metal")
    }

    #[test]
    fn to_device_changes_pointer() {
        if !has_metal() {
            return;
        }
        let host = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let device = host.to(Device::metal(0));

        assert_ne!(host.buffer().as_ptr(), device.buffer().as_ptr());
        assert!(!host.buffer().ptr_eq(device.buffer()));
    }

    #[test]
    fn to_device_preserves_metadata() {
        if !has_metal() {
            return;
        }
        let host = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let device = host.to(Device::metal(0));

        assert_eq!(device.device(), Device::Metal(0));
        assert_eq!(device.data_type(), &DataType::Float32);
        assert_eq!(device.shape(), Some([2, 3].as_slice()));
        assert_eq!(device.ndim(), 2);
        assert_eq!(device.size(), 6);
    }

    #[test]
    fn to_device_buffer_rejects_host_access() {
        if !has_metal() {
            return;
        }
        let device = f32_tensor(&[1.0], vec![1]).to(Device::metal(0));
        assert!(device.buffer().typed_data::<f32>().is_err());
        assert!(device.buffer().as_slice().is_err());
    }

    #[test]
    fn device_round_trip_preserves_data() {
        if !has_metal() {
            return;
        }
        let data: Vec<f32> = (0..120).map(|i| i as f32 * 0.1).collect();
        let host = f32_tensor(&data, vec![2, 3, 4, 5]);
        let back = host.to(Device::metal(0)).to(Device::cpu());

        assert_eq!(back.device(), Device::Cpu);
        assert_eq!(back.shape(), Some([2, 3, 4, 5].as_slice()));
        assert_eq!(back.buffer().typed_data::<f32>().unwrap(), &data[..]);
    }

    #[test]
    fn device_round_trip_is_fresh_allocation() {
        if !has_metal() {
            return;
        }
        let host = f32_tensor(&[1.0, 2.0, 3.0], vec![3]);
        let back = host.to(Device::metal(0)).to(Device::cpu());

        assert!(!host.buffer().ptr_eq(back.buffer()));
    }

    #[test]
    fn try_into_arrow_from_device_errors() {
        if !has_metal() {
            return;
        }
        let device = f32_tensor(&[1.0], vec![1]).to(Device::metal(0));
        let result: std::result::Result<ArrowTensor<'static, Float32Type>, _> = device.try_into();
        assert!(result.is_err());
    }

    #[test]
    fn as_ffi_on_device_tensor() {
        if !has_metal() {
            return;
        }
        let device = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).to(Device::metal(0));
        let ffi = device.as_ffi();

        assert_eq!(ffi.dtype, dtype::FLOAT32);
        assert_eq!(ffi.ndim, 2);
        assert_eq!(ffi.buffer.byte_length, 24);
        assert_eq!(ffi.buffer.device_type, AmDeviceType::Metal as i32,);
        assert_eq!(ffi.buffer.device_id, 0);
        assert!(!ffi.buffer.data.is_null());
    }
}

use crate::buffer::DeviceBuffer;
use crate::device::Device;
use crate::error::{DeviceError, Result};
use arrow::array::PrimitiveArray;
use arrow::buffer::{NullBuffer, ScalarBuffer};
use arrow::datatypes::{ArrowPrimitiveType, DataType};

/// 1-D device-aware primitive array with optional nulls. Mirrors
/// [`arrow::array::PrimitiveArray`] — wraps a [`DeviceBuffer`] for values
/// and an optional [`NullBuffer`] for validity (kept on host since bitmap
/// ops don't need device placement).
#[derive(Debug, Clone)]
pub struct Array {
    data_type: DataType,
    values: DeviceBuffer,
    nulls: Option<NullBuffer>,
}

impl Array {
    pub fn new(data_type: DataType, values: DeviceBuffer, nulls: Option<NullBuffer>) -> Self {
        Self {
            data_type,
            values,
            nulls,
        }
    }

    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }

    pub fn values(&self) -> &DeviceBuffer {
        &self.values
    }

    pub fn into_values(self) -> DeviceBuffer {
        self.values
    }

    pub fn nulls(&self) -> Option<&NullBuffer> {
        self.nulls.as_ref()
    }

    pub fn device(&self) -> Device {
        self.values.device()
    }

    pub fn len(&self) -> usize {
        self.nulls
            .as_ref()
            .map_or_else(|| self.values.len(), |n| n.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn null_count(&self) -> usize {
        self.nulls.as_ref().map_or(0, |n| n.null_count())
    }

    pub fn to(&self, device: Device) -> Self {
        Self {
            data_type: self.data_type.clone(),
            values: self.values.to(device),
            nulls: self.nulls.clone(),
        }
    }
}

impl<T: ArrowPrimitiveType> From<PrimitiveArray<T>> for Array {
    fn from(arr: PrimitiveArray<T>) -> Self {
        let (data_type, values, nulls) = arr.into_parts();
        Self {
            data_type,
            values: DeviceBuffer::from(values.into_inner()),
            nulls,
        }
    }
}

impl<T: ArrowPrimitiveType> TryFrom<Array> for PrimitiveArray<T> {
    type Error = DeviceError;

    fn try_from(arr: Array) -> Result<Self> {
        let len = arr.len();
        let Array { values, nulls, .. } = arr;
        let buffer: arrow::buffer::Buffer = values.try_into()?;
        let scalar = ScalarBuffer::new(buffer, 0, len);
        Ok(PrimitiveArray::new(scalar, nulls))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array as ArrowArray, Float32Array, Int32Array};
    use arrow::buffer::BooleanBuffer;
    use arrow::datatypes::{Float32Type, Int32Type};

    fn host_buf(data: &[f32]) -> DeviceBuffer {
        arrow::buffer::Buffer::from_slice_ref(data).into()
    }

    #[test]
    fn new_and_accessors() {
        let data = [1.0f32, 2.0, 3.0];
        let buf = host_buf(&data);
        let arr = Array::new(DataType::Float32, buf, None);

        assert_eq!(arr.data_type(), &DataType::Float32);
        assert_eq!(arr.device(), Device::Cpu);
        assert!(arr.nulls().is_none());
        assert_eq!(arr.null_count(), 0);
        assert!(!arr.is_empty());
    }

    #[test]
    fn len_without_nulls() {
        let buf = host_buf(&[1.0, 2.0, 3.0]);
        let arr = Array::new(DataType::Float32, buf, None);
        // Without nulls, len comes from buffer byte length
        assert_eq!(arr.len(), 3 * std::mem::size_of::<f32>());
    }

    #[test]
    fn len_with_nulls() {
        let buf = host_buf(&[1.0, 2.0, 3.0]);
        let bools = BooleanBuffer::from(vec![true, false, true]);
        let nulls = NullBuffer::new(bools);
        let arr = Array::new(DataType::Float32, buf, Some(nulls));
        // With nulls, len comes from NullBuffer::len()
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 1);
    }

    #[test]
    fn empty_array() {
        let buf: DeviceBuffer = arrow::buffer::Buffer::from_vec(Vec::<f32>::new()).into();
        let arr = Array::new(DataType::Float32, buf, None);
        assert!(arr.is_empty());
        assert_eq!(arr.len(), 0);
        assert_eq!(arr.null_count(), 0);
    }

    #[test]
    fn from_primitive_array_no_nulls() {
        let arrow_arr = Int32Array::from(vec![10, 20, 30]);
        let dev_arr = Array::from(arrow_arr);

        assert_eq!(dev_arr.data_type(), &DataType::Int32);
        assert_eq!(dev_arr.device(), Device::Cpu);
        assert!(dev_arr.nulls().is_none());
        assert_eq!(dev_arr.null_count(), 0);
    }

    #[test]
    fn from_primitive_array_with_nulls() {
        let arrow_arr = Int32Array::from(vec![Some(1), None, Some(3)]);
        let dev_arr = Array::from(arrow_arr);

        assert_eq!(dev_arr.data_type(), &DataType::Int32);
        assert!(dev_arr.nulls().is_some());
        assert_eq!(dev_arr.null_count(), 1);
        assert_eq!(dev_arr.len(), 3);
    }

    #[test]
    fn try_into_primitive_array_with_nulls_roundtrip() {
        let original = Int32Array::from(vec![Some(10), None, Some(30)]);
        let dev_arr = Array::from(original.clone());
        let back: PrimitiveArray<Int32Type> = dev_arr.try_into().unwrap();

        assert_eq!(back.len(), 3);
        assert!(back.is_valid(0));
        assert!(!back.is_valid(1));
        assert!(back.is_valid(2));
        assert_eq!(back.value(0), 10);
        assert_eq!(back.value(2), 30);
    }

    #[test]
    #[should_panic(expected = "the offset of the new Buffer cannot exceed the existing length")]
    fn try_into_without_nulls_panics_len_is_bytes_not_elements() {
        // BUG: DeviceArray::len() returns byte count when nulls are absent,
        // but TryFrom uses it as element count in ScalarBuffer::new.
        // This roundtrip panics for any type wider than 1 byte.
        let original = Float32Array::from(vec![1.5, 2.5, 3.5]);
        let dev_arr = Array::from(original);
        let _: PrimitiveArray<Float32Type> = dev_arr.try_into().unwrap();
    }

    #[test]
    fn into_values() {
        let buf = host_buf(&[1.0, 2.0]);
        let arr = Array::new(DataType::Float32, buf.clone(), None);
        let extracted = arr.into_values();
        assert!(buf.ptr_eq(&extracted));
    }

    #[test]
    fn clone_shares_buffer() {
        let arr = Array::from(Float32Array::from(vec![1.0, 2.0, 3.0]));
        let cloned = arr.clone();
        assert!(arr.values().ptr_eq(cloned.values()));
    }

    #[test]
    fn to_same_device_is_cheap() {
        let arr = Array::from(Float32Array::from(vec![1.0, 2.0]));
        let moved = arr.to(Device::Cpu);
        assert!(arr.values().ptr_eq(moved.values()));
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
    fn device_to_changes_value_pointer() {
        if !has_metal() {
            return;
        }
        let arr = Array::from(Float32Array::from(vec![1.0, 2.0, 3.0]));
        let device = arr.to(Device::metal(0));

        assert_ne!(arr.values().as_ptr(), device.values().as_ptr());
        assert!(!arr.values().ptr_eq(device.values()));
    }

    #[test]
    fn device_reports_correct_device() {
        if !has_metal() {
            return;
        }
        let device = Array::from(Float32Array::from(vec![1.0])).to(Device::metal(0));
        assert_eq!(device.device(), Device::Metal(0));
    }

    #[test]
    fn device_preserves_metadata() {
        if !has_metal() {
            return;
        }
        let arr = Array::from(Float32Array::from(vec![1.0, 2.0, 3.0]));
        let device = arr.to(Device::metal(0));

        assert_eq!(device.data_type(), &DataType::Float32);
        assert_eq!(device.values().len(), arr.values().len());
    }

    #[test]
    fn device_values_reject_host_access() {
        if !has_metal() {
            return;
        }
        let device = Array::from(Float32Array::from(vec![1.0])).to(Device::metal(0));
        assert!(device.values().typed_data::<f32>().is_err());
        assert!(device.values().as_slice().is_err());
    }

    #[test]
    fn device_nulls_stay_on_host() {
        if !has_metal() {
            return;
        }
        let arr = Array::from(Float32Array::from(vec![Some(1.0), None, Some(3.0)]));
        let device = arr.to(Device::metal(0));

        assert_eq!(device.device(), Device::Metal(0));
        assert_eq!(device.null_count(), 1);
        assert_eq!(device.len(), 3);

        let nulls = device
            .nulls()
            .expect("nulls should survive device transfer");
        assert!(!nulls.is_null(0));
        assert!(nulls.is_null(1));
        assert!(!nulls.is_null(2));
    }

    #[test]
    fn device_round_trip_with_nulls() {
        if !has_metal() {
            return;
        }
        let original = Float32Array::from(vec![Some(10.0), None, Some(30.0), None, Some(50.0)]);
        let arr = Array::from(original);
        let back = arr.to(Device::metal(0)).to(Device::cpu());

        assert_eq!(back.device(), Device::Cpu);
        assert_eq!(back.null_count(), 2);
        assert_eq!(back.len(), 5);

        let recovered: PrimitiveArray<Float32Type> = back.try_into().unwrap();
        assert!(recovered.is_valid(0));
        assert!(!recovered.is_valid(1));
        assert!(recovered.is_valid(2));
        assert!(!recovered.is_valid(3));
        assert!(recovered.is_valid(4));
        assert!((recovered.value(0) - 10.0).abs() < f32::EPSILON);
        assert!((recovered.value(2) - 30.0).abs() < f32::EPSILON);
        assert!((recovered.value(4) - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn device_round_trip_no_nulls() {
        if !has_metal() {
            return;
        }
        let original = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        let arr = Array::from(original);
        let back_arr = arr.to(Device::metal(0)).to(Device::cpu());

        assert!(back_arr.nulls().is_none());
        assert_eq!(back_arr.null_count(), 0);
    }

    #[test]
    fn device_round_trip_is_fresh_allocation() {
        if !has_metal() {
            return;
        }
        let arr = Array::from(Float32Array::from(vec![1.0, 2.0]));
        let back = arr.to(Device::metal(0)).to(Device::cpu());

        assert!(!arr.values().ptr_eq(back.values()));
    }

    #[test]
    fn device_to_same_device_is_cheap() {
        if !has_metal() {
            return;
        }
        let device = Array::from(Float32Array::from(vec![1.0])).to(Device::metal(0));
        let same = device.to(Device::metal(0));

        assert!(device.values().ptr_eq(same.values()));
        assert_eq!(device.values().strong_count(), 2);
    }
}

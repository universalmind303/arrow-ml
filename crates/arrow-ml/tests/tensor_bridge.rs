use arrow::buffer::ScalarBuffer;
use arrow::datatypes::{DataType, Float32Type, Float64Type};
use arrow::tensor::Tensor as ArrowTensor;
use arrow_ml_core::buffer::DeviceBuffer;
use arrow_ml_core::device::Device;
use arrow_ml_core::tensor::Tensor;

fn arrow_f32(data: Vec<f32>, shape: Vec<usize>) -> ArrowTensor<'static, Float32Type> {
    let buffer = ScalarBuffer::<f32>::from(data).into_inner();
    ArrowTensor::new_row_major(buffer, Some(shape), None).unwrap()
}

fn arrow_f64(data: Vec<f64>, shape: Vec<usize>) -> ArrowTensor<'static, Float64Type> {
    let buffer = ScalarBuffer::<f64>::from(data).into_inner();
    ArrowTensor::new_row_major(buffer, Some(shape), None).unwrap()
}

#[test]
fn f32_round_trip_zero_copy() {
    let original = arrow_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let original_ptr = original.data().as_ptr();

    let dt = Tensor::from(original);
    assert_eq!(dt.data_type(), &DataType::Float32);
    assert_eq!(dt.shape(), Some([2, 3].as_slice()));
    assert_eq!(dt.device(), Device::Cpu);
    assert_eq!(dt.buffer().as_ptr(), original_ptr);

    let back: ArrowTensor<'static, Float32Type> = dt.try_into().unwrap();
    assert_eq!(back.data().as_ptr(), original_ptr);
    assert_eq!(
        back.data().typed_data::<f32>(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(back.shape().unwrap(), &[2, 3]);
}

#[test]
fn f64_round_trip_zero_copy() {
    let original = arrow_f64(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
    let original_ptr = original.data().as_ptr();

    let dt = Tensor::from(original);
    assert_eq!(dt.data_type(), &DataType::Float64);
    assert_eq!(dt.buffer().as_ptr(), original_ptr);

    let back: ArrowTensor<'static, Float64Type> = dt.try_into().unwrap();
    assert_eq!(back.data().as_ptr(), original_ptr);
    assert_eq!(back.data().typed_data::<f64>(), &[10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn round_trip_preserves_strides() {
    let data = arrow::buffer::Buffer::from_slice_ref(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let original =
        ArrowTensor::<Float32Type>::try_new(data, Some(vec![2, 3]), Some(vec![12, 4]), None)
            .unwrap();

    let dt = Tensor::from(original);
    assert_eq!(dt.strides(), Some([12, 4].as_slice()));

    let back: ArrowTensor<'static, Float32Type> = dt.try_into().unwrap();
    assert_eq!(back.strides().unwrap(), &[12, 4]);
}

#[test]
fn buffer_refcount_preserved() {
    let original = arrow_f32(vec![1.0, 2.0, 3.0], vec![3]);
    let dt = Tensor::from(original);

    assert_eq!(dt.buffer().strong_count(), 1);

    let dt2 = dt.clone();
    assert_eq!(dt.buffer().strong_count(), 2);
    assert!(dt.buffer().ptr_eq(dt2.buffer()));
}

#[test]
fn to_cpu_is_noop() {
    let original = arrow_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let dt = Tensor::from(original);
    let moved = dt.to(Device::Cpu);

    assert!(dt.buffer().ptr_eq(moved.buffer()));
    assert_eq!(dt.buffer().strong_count(), 2);
}

#[test]
fn scalar_tensor_round_trip() {
    let original = arrow_f32(vec![42.0], vec![1]);
    let dt = Tensor::from(original);
    assert_eq!(dt.size(), 1);

    let back: ArrowTensor<'static, Float32Type> = dt.try_into().unwrap();
    assert_eq!(back.data().typed_data::<f32>(), &[42.0]);
}

#[test]
fn high_dimensional_round_trip() {
    let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
    let original = arrow_f32(data.clone(), vec![2, 3, 4, 5]);
    let dt = Tensor::from(original);

    assert_eq!(dt.ndim(), 4);
    assert_eq!(dt.size(), 120);

    let back: ArrowTensor<'static, Float32Type> = dt.try_into().unwrap();
    assert_eq!(back.data().typed_data::<f32>(), &data[..]);
    assert_eq!(back.shape().unwrap(), &[2, 3, 4, 5]);
}

#[test]
fn from_raw_buffer_and_back() {
    let buf = DeviceBuffer::from(arrow::buffer::Buffer::from_slice_ref(&[5.0f32, 10.0, 15.0]));
    let dt = Tensor::new(DataType::Float32, buf, Some(vec![3]), None);
    assert_eq!(dt.buffer().typed_data::<f32>().unwrap(), &[5.0, 10.0, 15.0]);

    let back: ArrowTensor<'static, Float32Type> = dt.try_into().unwrap();
    assert_eq!(back.data().typed_data::<f32>(), &[5.0, 10.0, 15.0]);
}

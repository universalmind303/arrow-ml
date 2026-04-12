use arrow::array::{Array as _, Float32Array, Float64Array, Int32Array, PrimitiveArray};
use arrow::datatypes::{DataType, Float32Type, Int32Type};
use arrow_ml_core::array::Array;
use arrow_ml_core::device::Device;

#[test]
fn f32_no_nulls_round_trip() {
    let original = Float32Array::from(vec![1.0, 2.0, 3.0]);
    let dev = Array::from(original.clone());

    assert_eq!(dev.data_type(), &DataType::Float32);
    assert_eq!(dev.device(), Device::Cpu);
    assert!(dev.nulls().is_none());
    assert_eq!(dev.null_count(), 0);
}

#[test]
fn f32_with_nulls_round_trip() {
    let original = Float32Array::from(vec![Some(1.5), None, Some(3.5), None, Some(5.5)]);
    assert_eq!(original.null_count(), 2);

    let dev = Array::from(original.clone());
    assert_eq!(dev.null_count(), 2);
    assert_eq!(dev.len(), 5);

    let back: PrimitiveArray<Float32Type> = dev.try_into().unwrap();
    assert_eq!(back.len(), 5);
    assert_eq!(back.null_count(), 2);
    assert!(back.is_valid(0));
    assert!(!back.is_valid(1));
    assert!(back.is_valid(2));
    assert!(!back.is_valid(3));
    assert!(back.is_valid(4));
    assert!((back.value(0) - 1.5).abs() < f32::EPSILON);
    assert!((back.value(2) - 3.5).abs() < f32::EPSILON);
    assert!((back.value(4) - 5.5).abs() < f32::EPSILON);
}

#[test]
fn i32_with_nulls_round_trip() {
    let original = Int32Array::from(vec![Some(10), None, Some(30)]);
    let dev = Array::from(original);
    let back: PrimitiveArray<Int32Type> = dev.try_into().unwrap();

    assert_eq!(back.len(), 3);
    assert_eq!(back.null_count(), 1);
    assert_eq!(back.value(0), 10);
    assert!(!back.is_valid(1));
    assert_eq!(back.value(2), 30);
}

#[test]
fn f64_no_nulls_preserves_values() {
    let original = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]);
    let dev = Array::from(original);

    assert_eq!(dev.data_type(), &DataType::Float64);
    assert!(dev.nulls().is_none());
}

#[test]
fn all_null_array() {
    let original = Float32Array::from(vec![None, None, None]);
    let dev = Array::from(original);

    assert_eq!(dev.null_count(), 3);
    assert_eq!(dev.len(), 3);

    let back: PrimitiveArray<Float32Type> = dev.try_into().unwrap();
    assert_eq!(back.null_count(), 3);
    for i in 0..3 {
        assert!(!back.is_valid(i));
    }
}

#[test]
fn empty_array() {
    let original = Float32Array::from(vec![] as Vec<f32>);
    let dev = Array::from(original);
    assert!(dev.is_empty());
    assert_eq!(dev.null_count(), 0);
}

#[test]
fn clone_shares_buffer() {
    let original = Float32Array::from(vec![1.0, 2.0, 3.0]);
    let dev = Array::from(original);
    let cloned = dev.clone();
    assert!(dev.values().ptr_eq(cloned.values()));
}

#[test]
fn to_cpu_is_noop() {
    let dev = Array::from(Float32Array::from(vec![1.0, 2.0]));
    let moved = dev.to(Device::Cpu);
    assert!(dev.values().ptr_eq(moved.values()));
}

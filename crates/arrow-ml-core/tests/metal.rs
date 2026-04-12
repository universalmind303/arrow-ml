#![cfg(target_os = "macos")]

use arrow::array::{Float32Array, Int32Array};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::{DataType, Float32Type};
use arrow::tensor::Tensor as ArrowTensor;
use arrow_ml_common::device_tensor::AmDeviceType;
use arrow_ml_common::BackendRegistry;
use arrow_ml_core::array::Array;
use arrow_ml_core::buffer::DeviceBuffer;
use arrow_ml_core::device::Device;
use arrow_ml_core::tensor::Tensor;

fn require_metal() {
    let reg = BackendRegistry::global();
    if reg.best_for_device(AmDeviceType::Metal, 0).is_none() {
        panic!(
            "Metal backend not loaded. The cdylib should be built \
             automatically via the dev-dependency on arrow-ml-backend-metal."
        );
    }
}

// ---------------------------------------------------------------------------
// Backend discovery
// ---------------------------------------------------------------------------

#[test]
fn metal_backend_loads() {
    require_metal();
    let names = BackendRegistry::global().loaded_backends();
    assert!(names.contains(&"metal"), "expected 'metal' in {names:?}");
}

#[test]
fn metal_backend_reports_matmul_support() {
    require_metal();
    let reg = BackendRegistry::global();
    let backend = reg.best_matmul().expect("no matmul backend");
    assert_eq!(backend.device_type, AmDeviceType::Metal as i32);
}

// ---------------------------------------------------------------------------
// DeviceBuffer — host→metal→host roundtrip
// ---------------------------------------------------------------------------

#[test]
fn buffer_roundtrip_f32() {
    require_metal();
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let host_buf = DeviceBuffer::from(Buffer::from_slice_ref(&data));

    assert_eq!(host_buf.device(), Device::Cpu);

    let metal_buf = host_buf.to(Device::metal(0));
    assert_eq!(metal_buf.device(), Device::metal(0));
    assert_eq!(metal_buf.len(), host_buf.len());

    let back = metal_buf.to(Device::cpu());
    assert_eq!(back.device(), Device::Cpu);
    let result = back.typed_data::<f32>().unwrap();
    assert_eq!(result, &data);
}

#[test]
fn buffer_roundtrip_u8() {
    require_metal();
    let data: Vec<u8> = (0..=255).collect();
    let host_buf = DeviceBuffer::from(Buffer::from_slice_ref(&data));

    let metal_buf = host_buf.to(Device::metal(0));
    let back = metal_buf.to(Device::cpu());
    assert_eq!(back.as_slice().unwrap(), &data[..]);
}

#[test]
fn buffer_roundtrip_f64() {
    require_metal();
    let data: Vec<f64> = vec![std::f64::consts::PI, std::f64::consts::E, 0.0, -1.0];
    let host_buf = DeviceBuffer::from(Buffer::from_slice_ref(&data));

    let back = host_buf.to(Device::metal(0)).to(Device::cpu());
    assert_eq!(back.typed_data::<f64>().unwrap(), &data);
}

#[test]
fn buffer_roundtrip_single_byte() {
    require_metal();
    let host_buf = DeviceBuffer::from(Buffer::from_vec(vec![0xFFu8]));

    let metal_buf = host_buf.to(Device::metal(0));
    assert_eq!(metal_buf.len(), 1);

    let back = metal_buf.to(Device::cpu());
    assert_eq!(back.as_slice().unwrap(), &[0xFF]);
}

// ---------------------------------------------------------------------------
// DeviceBuffer — device operations
// ---------------------------------------------------------------------------

#[test]
fn buffer_alloc_on_metal() {
    require_metal();
    let buf = DeviceBuffer::new(1024, Device::metal(0));
    assert_eq!(buf.device(), Device::metal(0));
    assert_eq!(buf.len(), 1024);
    assert!(!buf.as_ptr().is_null());
}

#[test]
fn buffer_metal_to_same_device_is_cheap() {
    require_metal();
    let host_buf = DeviceBuffer::from(Buffer::from_slice_ref(&[1.0f32, 2.0, 3.0]));
    let metal_buf = host_buf.to(Device::metal(0));

    let clone = metal_buf.to(Device::metal(0));
    assert!(metal_buf.ptr_eq(&clone));
    assert_eq!(metal_buf.strong_count(), 2);
}

#[test]
fn buffer_metal_slice() {
    require_metal();
    let data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let host_buf = DeviceBuffer::from(Buffer::from_slice_ref(&data));
    let metal_buf = host_buf.to(Device::metal(0));

    // Slice off the first 2 floats (8 bytes)
    let sliced = metal_buf.slice_with_length(8, 12);
    assert_eq!(sliced.device(), Device::metal(0));
    assert_eq!(sliced.len(), 12);

    let back = sliced.to(Device::cpu());
    assert_eq!(back.typed_data::<f32>().unwrap(), &[30.0, 40.0, 50.0]);
}

#[test]
fn buffer_metal_slice_shares_alloc() {
    require_metal();
    let host_buf = DeviceBuffer::from(Buffer::from_slice_ref(&[1.0f32; 100]));
    let metal_buf = host_buf.to(Device::metal(0));

    let sliced = metal_buf.slice(4);
    assert_eq!(metal_buf.strong_count(), 2);
    assert_eq!(sliced.strong_count(), 2);
}

#[test]
fn buffer_metal_ptr_offset() {
    require_metal();
    let host_buf = DeviceBuffer::from(Buffer::from_slice_ref(&[0u8; 64]));
    let metal_buf = host_buf.to(Device::metal(0));
    assert_eq!(metal_buf.ptr_offset(), 0);

    let sliced = metal_buf.slice(16);
    assert_eq!(sliced.ptr_offset(), 16);
}

#[test]
fn buffer_metal_as_slice_returns_err() {
    require_metal();
    let metal_buf = DeviceBuffer::new(64, Device::metal(0));
    assert!(metal_buf.as_slice().is_err());
}

#[test]
fn buffer_metal_typed_data_returns_err() {
    require_metal();
    let metal_buf = DeviceBuffer::new(64, Device::metal(0));
    assert!(metal_buf.typed_data::<f32>().is_err());
}

#[test]
fn buffer_metal_into_mutable_returns_err() {
    require_metal();
    let metal_buf = DeviceBuffer::new(64, Device::metal(0));
    assert!(metal_buf.into_mutable().is_err());
}

#[test]
fn buffer_metal_into_vec_returns_err() {
    require_metal();
    let metal_buf = DeviceBuffer::new(64, Device::metal(0));
    assert!(metal_buf.into_vec::<f32>().is_err());
}

#[test]
fn buffer_metal_try_into_arrow_buffer_returns_err() {
    require_metal();
    let metal_buf = DeviceBuffer::new(64, Device::metal(0));
    let result: Result<Buffer, _> = metal_buf.try_into();
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// DeviceBuffer — concurrency
// ---------------------------------------------------------------------------

#[test]
fn buffer_metal_send_across_threads() {
    require_metal();
    let host_buf = DeviceBuffer::from(Buffer::from_slice_ref(&[42.0f32; 10]));
    let metal_buf = host_buf.to(Device::metal(0));

    let handle = std::thread::spawn(move || metal_buf.to(Device::cpu()));
    let back = handle.join().unwrap();
    assert_eq!(back.typed_data::<f32>().unwrap(), &[42.0f32; 10]);
}

#[test]
fn buffer_metal_clone_across_threads() {
    require_metal();
    let host_buf = DeviceBuffer::from(Buffer::from_slice_ref(&[1.0f32, 2.0, 3.0]));
    let metal_buf = host_buf.to(Device::metal(0));
    let clone = metal_buf.clone();

    let h1 = std::thread::spawn(move || metal_buf.to(Device::cpu()));
    let h2 = std::thread::spawn(move || clone.to(Device::cpu()));

    let r1 = h1.join().unwrap().typed_data::<f32>().unwrap().to_vec();
    let r2 = h2.join().unwrap().typed_data::<f32>().unwrap().to_vec();
    assert_eq!(r1, r2);
    assert_eq!(r1, &[1.0, 2.0, 3.0]);
}

// ---------------------------------------------------------------------------
// DeviceArray — host→metal→host
// ---------------------------------------------------------------------------

#[test]
fn array_roundtrip_no_nulls() {
    require_metal();
    let arr = Array::from(Float32Array::from(vec![1.0, 2.0, 3.0, 4.0]));
    assert_eq!(arr.device(), Device::Cpu);

    let metal_arr = arr.to(Device::metal(0));
    assert_eq!(metal_arr.device(), Device::metal(0));
    assert!(metal_arr.nulls().is_none());

    let back = metal_arr.to(Device::cpu());
    assert_eq!(back.device(), Device::Cpu);
    assert_eq!(
        back.values().as_slice().unwrap(),
        arr.values().as_slice().unwrap()
    );
}

#[test]
fn array_roundtrip_with_nulls() {
    require_metal();
    let arr = Array::from(Int32Array::from(vec![
        Some(10),
        None,
        Some(30),
        None,
        Some(50),
    ]));
    assert_eq!(arr.null_count(), 2);

    let metal_arr = arr.to(Device::metal(0));
    assert_eq!(metal_arr.device(), Device::metal(0));
    assert_eq!(metal_arr.null_count(), 2);
    assert_eq!(metal_arr.len(), 5);

    let back = metal_arr.to(Device::cpu());
    assert_eq!(back.null_count(), 2);
    assert_eq!(
        back.values().as_slice().unwrap(),
        arr.values().as_slice().unwrap()
    );
}

#[test]
fn array_metal_preserves_data_type() {
    require_metal();
    let arr = Array::from(Float32Array::from(vec![1.0]));
    let metal_arr = arr.to(Device::metal(0));
    assert_eq!(metal_arr.data_type(), &DataType::Float32);
}

// ---------------------------------------------------------------------------
// DeviceTensor — host→metal→host
// ---------------------------------------------------------------------------

fn make_f32_tensor(data: Vec<f32>, rows: usize, cols: usize) -> Tensor {
    let buffer = ScalarBuffer::<f32>::from(data).into_inner();
    let tensor =
        ArrowTensor::<Float32Type>::new_row_major(buffer, Some(vec![rows, cols]), None).unwrap();
    Tensor::from(tensor)
}

#[test]
fn tensor_roundtrip_2d() {
    require_metal();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = make_f32_tensor(data.clone(), 2, 3);

    let metal_t = t.to(Device::metal(0));
    assert_eq!(metal_t.device(), Device::metal(0));
    assert_eq!(metal_t.shape(), Some([2, 3].as_slice()));
    assert_eq!(metal_t.ndim(), 2);
    assert_eq!(metal_t.size(), 6);

    let back = metal_t.to(Device::cpu());
    assert_eq!(back.device(), Device::Cpu);
    assert_eq!(back.buffer().typed_data::<f32>().unwrap(), &data);
}

#[test]
fn tensor_roundtrip_preserves_strides() {
    require_metal();
    let buf = DeviceBuffer::from(Buffer::from_slice_ref(&[1.0f32; 6]));
    let t = Tensor::new(DataType::Float32, buf, Some(vec![2, 3]), Some(vec![12, 4]));

    let metal_t = t.to(Device::metal(0));
    assert_eq!(metal_t.strides(), Some([12, 4].as_slice()));

    let back = metal_t.to(Device::cpu());
    assert_eq!(back.strides(), Some([12, 4].as_slice()));
}

#[test]
fn tensor_metal_ffi_reports_metal_device() {
    require_metal();
    let t = make_f32_tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let metal_t = t.to(Device::metal(0));
    let ffi = metal_t.as_ffi();

    assert_eq!(ffi.buffer.device_type, AmDeviceType::Metal as i32);
    assert_eq!(ffi.buffer.device_id, 0);
    assert!(!ffi.buffer.data.is_null());
}

#[test]
fn tensor_roundtrip_1d() {
    require_metal();
    let data = vec![10.0f32, 20.0, 30.0];
    let buf = DeviceBuffer::from(Buffer::from_slice_ref(&data));
    let t = Tensor::new(DataType::Float32, buf, Some(vec![3]), None);

    let back = t.to(Device::metal(0)).to(Device::cpu());
    assert_eq!(back.buffer().typed_data::<f32>().unwrap(), &data);
    assert_eq!(back.shape(), Some([3].as_slice()));
}

#[test]
fn tensor_roundtrip_large() {
    require_metal();
    let n = 1024 * 1024;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let buf = DeviceBuffer::from(Buffer::from_slice_ref(&data));
    let t = Tensor::new(DataType::Float32, buf, Some(vec![1024, 1024]), None);

    let back = t.to(Device::metal(0)).to(Device::cpu());
    assert_eq!(back.buffer().typed_data::<f32>().unwrap(), &data);
}

// ---------------------------------------------------------------------------
// Matmul on Metal
// ---------------------------------------------------------------------------

#[test]
fn matmul_on_metal_small() {
    require_metal();
    let a = make_f32_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = make_f32_tensor(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);

    let a_metal = a.to(Device::metal(0));
    let b_metal = b.to(Device::metal(0));

    let c_metal = arrow_ml_linalg::matmul::matmul(&a_metal, &b_metal).unwrap();
    assert_eq!(c_metal.device(), Device::metal(0));
    assert_eq!(c_metal.shape(), Some([2, 2].as_slice()));

    let c_host = c_metal.to(Device::cpu());
    let data = c_host.buffer().typed_data::<f32>().unwrap();
    assert!((data[0] - 58.0).abs() < 1e-3, "got {}", data[0]);
    assert!((data[1] - 64.0).abs() < 1e-3, "got {}", data[1]);
    assert!((data[2] - 139.0).abs() < 1e-3, "got {}", data[2]);
    assert!((data[3] - 154.0).abs() < 1e-3, "got {}", data[3]);
}

#[test]
fn matmul_on_metal_identity() {
    require_metal();
    let eye = make_f32_tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
    let a = make_f32_tensor(vec![5.0, 6.0, 7.0, 8.0], 2, 2);

    let c = arrow_ml_linalg::matmul::matmul(&eye.to(Device::metal(0)), &a.to(Device::metal(0)))
        .unwrap();

    let data = c
        .to(Device::cpu())
        .buffer()
        .typed_data::<f32>()
        .unwrap()
        .to_vec();
    assert!((data[0] - 5.0).abs() < 1e-3);
    assert!((data[1] - 6.0).abs() < 1e-3);
    assert!((data[2] - 7.0).abs() < 1e-3);
    assert!((data[3] - 8.0).abs() < 1e-3);
}

#[test]
fn matmul_metal_vs_cpu_agree() {
    require_metal();
    let a = make_f32_tensor(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        3,
        4,
    );
    let b = make_f32_tensor(vec![1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0], 4, 2);

    let c_cpu = arrow_ml_linalg::matmul::matmul(&a, &b).unwrap();
    let cpu_data = c_cpu.buffer().typed_data::<f32>().unwrap();

    let c_metal =
        arrow_ml_linalg::matmul::matmul(&a.to(Device::metal(0)), &b.to(Device::metal(0))).unwrap();
    let metal_data = c_metal
        .to(Device::cpu())
        .buffer()
        .typed_data::<f32>()
        .unwrap()
        .to_vec();

    assert_eq!(cpu_data.len(), metal_data.len());
    for (i, (c, m)) in cpu_data.iter().zip(metal_data.iter()).enumerate() {
        assert!(
            (c - m).abs() < 1e-3,
            "mismatch at index {i}: cpu={c}, metal={m}"
        );
    }
}

#[test]
fn matmul_device_mismatch_errors() {
    require_metal();
    let a = make_f32_tensor(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b_metal = make_f32_tensor(vec![1.0, 0.0, 0.0, 1.0], 2, 2).to(Device::metal(0));

    let result = arrow_ml_linalg::matmul::matmul(&a, &b_metal);
    assert!(result.is_err());
}

#[test]
fn matmul_on_metal_1x1() {
    require_metal();
    let a = make_f32_tensor(vec![3.0], 1, 1).to(Device::metal(0));
    let b = make_f32_tensor(vec![7.0], 1, 1).to(Device::metal(0));

    let c = arrow_ml_linalg::matmul::matmul(&a, &b).unwrap();
    let data = c
        .to(Device::cpu())
        .buffer()
        .typed_data::<f32>()
        .unwrap()
        .to_vec();
    assert!((data[0] - 21.0).abs() < 1e-3);
}

#[test]
fn matmul_on_metal_non_square() {
    require_metal();
    // 1x4 * 4x1 = 1x1
    let a = make_f32_tensor(vec![1.0, 2.0, 3.0, 4.0], 1, 4).to(Device::metal(0));
    let b = make_f32_tensor(vec![1.0, 1.0, 1.0, 1.0], 4, 1).to(Device::metal(0));

    let c = arrow_ml_linalg::matmul::matmul(&a, &b).unwrap();
    assert_eq!(c.shape(), Some([1, 1].as_slice()));
    let data = c
        .to(Device::cpu())
        .buffer()
        .typed_data::<f32>()
        .unwrap()
        .to_vec();
    assert!((data[0] - 10.0).abs() < 1e-3);
}

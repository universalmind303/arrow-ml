use crate::error::{LinalgError, Result};
use arrow::buffer::Buffer;
use arrow::datatypes::DataType;
use arrow_ml_common::device_tensor::dtype;
use arrow_ml_common::kernels::matmul::MatmulKernel;
use arrow_ml_common::BackendRegistry;
use arrow_ml_core::buffer::DeviceBuffer;
use arrow_ml_core::device::Device;
use arrow_ml_core::tensor::Tensor;
use std::mem;

fn shape_2d(tensor: &Tensor) -> Result<(usize, usize)> {
    let shape = tensor
        .shape()
        .ok_or_else(|| LinalgError::InvalidArgument("matmul: tensor has no shape".into()))?;
    if shape.len() != 2 {
        return Err(LinalgError::InvalidArgument(format!(
            "matmul: expected 2D tensor, got {}D",
            shape.len()
        )));
    }
    Ok((shape[0], shape[1]))
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (m, k1) = shape_2d(a)?;
    let (k2, n) = shape_2d(b)?;

    if k1 != k2 {
        return Err(LinalgError::ShapeMismatch {
            operation: "matmul",
            expected: format!("A columns ({k1}) == B rows"),
            actual: format!("B rows = {k2}"),
        });
    }

    if a.device() != b.device() {
        return Err(LinalgError::DeviceMismatch);
    }

    if a.data_type() != b.data_type() {
        return Err(LinalgError::InvalidArgument(format!(
            "matmul: dtype mismatch: {:?} vs {:?}",
            a.data_type(),
            b.data_type()
        )));
    }

    match a.device() {
        Device::Cpu => cpu_matmul(a, b, m, k1, n),
        device @ (Device::Metal(_) | Device::Cuda(_)) => device_matmul(a, b, m, n, device),
    }
}

fn cpu_matmul(a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
    let dt = a.data_type().clone();
    match &dt {
        DataType::Float32 => {
            let a_data = a.buffer().typed_data::<f32>()?;
            let b_data = b.buffer().typed_data::<f32>()?;
            let c = naive_gemm(a_data, b_data, m, k, n);
            Ok(make_tensor_2d(dt, c, m, n))
        }
        DataType::Float64 => {
            let a_data = a.buffer().typed_data::<f64>()?;
            let b_data = b.buffer().typed_data::<f64>()?;
            let c = naive_gemm(a_data, b_data, m, k, n);
            Ok(make_tensor_2d(dt, c, m, n))
        }
        other => Err(LinalgError::UnsupportedDtype(format!("{other:?}"))),
    }
}

fn device_matmul(a: &Tensor, b: &Tensor, m: usize, n: usize, device: Device) -> Result<Tensor> {
    let dt = a.data_type();
    let dtype_code = match dt {
        DataType::Float32 => dtype::FLOAT32,
        other => return Err(LinalgError::UnsupportedDtype(format!("{other:?}"))),
    };

    let am_dev = device.to_am();
    let backend = BackendRegistry::global()
        .best_matmul_for(dtype_code, am_dev as i32)
        .ok_or_else(|| LinalgError::InvalidArgument("no matmul backend for this device".into()))?;

    let kernel = MatmulKernel::open(backend, dtype_code, am_dev as i32)
        .map_err(|e| LinalgError::InvalidArgument(format!("{e}")))?;

    let elem = mem::size_of::<f32>();
    let c_bytes = m * n * elem;

    let c_tensor = Tensor::new(
        dt.clone(),
        DeviceBuffer::new(c_bytes, device),
        Some(vec![m, n]),
        Some(vec![n * elem, elem]),
    );

    let a_ffi = a.as_ffi();
    let b_ffi = b.as_ffi();
    let mut c_ffi = c_tensor.as_ffi();

    unsafe {
        kernel
            .invoke(&a_ffi, &b_ffi, &mut c_ffi)
            .map_err(|e| LinalgError::InvalidArgument(format!("{e}")))?;
    }

    Ok(c_tensor)
}

fn naive_gemm<T>(a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Vec<T>
where
    T: Copy + Default + std::ops::Mul<Output = T> + std::ops::AddAssign,
{
    let mut c = vec![T::default(); m * n];
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
    c
}

fn make_tensor_2d<T: arrow::datatypes::ArrowNativeType>(
    data_type: DataType,
    data: Vec<T>,
    m: usize,
    n: usize,
) -> Tensor {
    let elem_size = std::mem::size_of::<T>();
    Tensor::new(
        data_type,
        DeviceBuffer::from(Buffer::from_vec(data)),
        Some(vec![m, n]),
        Some(vec![n * elem_size, elem_size]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Float32Type, Float64Type};
    use arrow::tensor::Tensor as ArrowTensor;

    fn make_f32(data: Vec<f32>, rows: usize, cols: usize) -> Tensor {
        let arrow_t =
            ArrowTensor::<Float32Type>::new_row_major(data.into(), Some(vec![rows, cols]), None)
                .unwrap();
        Tensor::from(arrow_t)
    }

    fn make_f64(data: Vec<f64>, rows: usize, cols: usize) -> Tensor {
        let arrow_t =
            ArrowTensor::<Float64Type>::new_row_major(data.into(), Some(vec![rows, cols]), None)
                .unwrap();
        Tensor::from(arrow_t)
    }

    #[test]
    fn test_matmul_2x3_times_3x2() {
        let a = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = make_f32(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
        let c = matmul(&a, &b).unwrap();

        assert_eq!(c.shape().unwrap(), &[2, 2]);
        let data = c.buffer().typed_data::<f32>().unwrap();
        assert!((data[0] - 58.0).abs() < 1e-5);
        assert!((data[1] - 64.0).abs() < 1e-5);
        assert!((data[2] - 139.0).abs() < 1e-5);
        assert!((data[3] - 154.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_identity() {
        let eye = make_f32(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let a = make_f32(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = matmul(&eye, &a).unwrap();
        let data = c.buffer().typed_data::<f32>().unwrap();
        assert_eq!(data, &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_matmul_f64() {
        let a = make_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = make_f64(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
        let c = matmul(&a, &b).unwrap();

        let data = c.buffer().typed_data::<f64>().unwrap();
        assert!((data[0] - 58.0).abs() < 1e-10);
        assert!((data[1] - 64.0).abs() < 1e-10);
        assert!((data[2] - 139.0).abs() < 1e-10);
        assert!((data[3] - 154.0).abs() < 1e-10);
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let a = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = make_f32(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn test_matmul_1x1() {
        let a = make_f32(vec![3.0], 1, 1);
        let b = make_f32(vec![7.0], 1, 1);
        let c = matmul(&a, &b).unwrap();
        let data = c.buffer().typed_data::<f32>().unwrap();
        assert!((data[0] - 21.0).abs() < 1e-5);
    }
}

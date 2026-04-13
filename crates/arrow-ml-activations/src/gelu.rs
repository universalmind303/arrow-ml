use arrow::buffer::Buffer;
use arrow::datatypes::DataType;
use arrow_ml_common::device_tensor::dtype;
use arrow_ml_common::error::{KernelError, Result};
use arrow_ml_common::kernels::gelu::GeluKernel;
use arrow_ml_common::BackendRegistry;
use arrow_ml_core::buffer::DeviceBuffer;
use arrow_ml_core::device::Device;
use arrow_ml_core::tensor::Tensor;
use std::mem;

const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
const GELU_COEFF: f64 = 0.044715;

pub fn gelu(input: &Tensor) -> Result<Tensor> {
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("gelu: tensor has no shape".into()))?;

    match input.device() {
        Device::Cpu => cpu_gelu(input, shape),
        device @ (Device::Metal(_) | Device::Cuda(_)) => device_gelu(input, shape, device),
    }
}

fn cpu_gelu(input: &Tensor, shape: &[usize]) -> Result<Tensor> {
    match input.data_type() {
        DataType::Float32 => cpu_gelu_typed::<f32>(input, shape),
        DataType::Float64 => cpu_gelu_typed::<f64>(input, shape),
        other => Err(KernelError::InvalidArgument(format!(
            "gelu: unsupported dtype {other:?}"
        ))),
    }
}

fn cpu_gelu_typed<T>(input: &Tensor, shape: &[usize]) -> Result<Tensor>
where
    T: arrow::datatypes::ArrowNativeType + num_traits::Float,
{
    let data: &[T] = input
        .buffer()
        .typed_data()
        .map_err(|e| KernelError::InvalidArgument(format!("gelu: {e}")))?;

    let half = T::from(0.5).unwrap();
    let one = T::from(1.0).unwrap();
    let coeff = T::from(GELU_COEFF).unwrap();
    let sqrt_2_pi = T::from(SQRT_2_OVER_PI).unwrap();

    let out: Vec<T> = data
        .iter()
        .map(|&x| {
            let inner = sqrt_2_pi * (x + coeff * x * x * x);
            half * x * (one + inner.tanh())
        })
        .collect();

    let ndim = shape.len();
    let elem = mem::size_of::<T>();
    let mut strides = vec![0usize; ndim];
    if ndim > 0 {
        strides[ndim - 1] = elem;
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    Ok(Tensor::new(
        input.data_type().clone(),
        DeviceBuffer::from(Buffer::from_vec(out)),
        Some(shape.to_vec()),
        Some(strides),
    ))
}

fn device_gelu(input: &Tensor, shape: &[usize], device: Device) -> Result<Tensor> {
    let dtype_code = match input.data_type() {
        DataType::Float32 => dtype::FLOAT32,
        other => {
            return Err(KernelError::InvalidArgument(format!(
                "gelu: unsupported dtype {other:?} for device"
            )))
        }
    };

    let am_dev = device.to_am();
    let backend = BackendRegistry::global()
        .best_gelu_for(dtype_code, am_dev as i32)
        .ok_or_else(|| KernelError::InvalidArgument("no gelu backend for this device".into()))?;

    let kernel = GeluKernel::open(backend, dtype_code, am_dev as i32)
        .map_err(|e| KernelError::InvalidArgument(format!("{e}")))?;

    let total: usize = shape.iter().product();
    let elem = mem::size_of::<f32>();
    let ndim = shape.len();
    let mut strides = vec![0usize; ndim];
    if ndim > 0 {
        strides[ndim - 1] = elem;
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    let output = Tensor::new(
        input.data_type().clone(),
        DeviceBuffer::new(total * elem, device),
        Some(shape.to_vec()),
        Some(strides),
    );

    let in_ffi = input.as_ffi();
    let mut out_ffi = output.as_ffi();

    unsafe {
        kernel
            .invoke(&in_ffi, &mut out_ffi)
            .map_err(|e| KernelError::InvalidArgument(format!("{e}")))?;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::Float32Type;
    use arrow::tensor::Tensor as ArrowTensor;

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        let arrow_t = ArrowTensor::<Float32Type>::new_row_major(
            ScalarBuffer::<f32>::from(data).into_inner(),
            Some(shape),
            None,
        )
        .unwrap();
        Tensor::from(arrow_t)
    }

    fn ref_gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + (SQRT_2_OVER_PI as f32 * (x + GELU_COEFF as f32 * x * x * x)).tanh())
    }

    #[test]
    fn test_gelu_zero() {
        let input = make_f32(vec![0.0], vec![1]);
        let output = gelu(&input).unwrap();
        let data = output.buffer().typed_data::<f32>().unwrap();
        assert!((data[0]).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_positive() {
        let input = make_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let output = gelu(&input).unwrap();
        let data = output.buffer().typed_data::<f32>().unwrap();
        for (i, &v) in data.iter().enumerate() {
            let expected = ref_gelu([1.0, 2.0, 3.0][i]);
            assert!(
                (v - expected).abs() < 1e-5,
                "at {i}: got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_gelu_negative() {
        let input = make_f32(vec![-1.0, -2.0, -3.0], vec![3]);
        let output = gelu(&input).unwrap();
        let data = output.buffer().typed_data::<f32>().unwrap();
        // GELU of negative values should be small but not zero
        for &v in data {
            assert!(v < 0.0);
            assert!(v > -0.2);
        }
    }

    #[test]
    fn test_gelu_preserves_shape() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let output = gelu(&input).unwrap();
        assert_eq!(output.shape().unwrap(), &[2, 3]);
    }

    #[test]
    fn test_gelu_large_positive() {
        // For large x, gelu(x) ≈ x
        let input = make_f32(vec![10.0], vec![1]);
        let output = gelu(&input).unwrap();
        let data = output.buffer().typed_data::<f32>().unwrap();
        assert!((data[0] - 10.0).abs() < 1e-3);
    }
}

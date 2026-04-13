use arrow::buffer::Buffer;
use arrow::datatypes::DataType;
use arrow_ml_common::device_tensor::dtype;
use arrow_ml_common::error::{KernelError, Result};
use arrow_ml_common::kernels::softmax;
use arrow_ml_common::BackendRegistry;
use arrow_ml_core::buffer::DeviceBuffer;
use arrow_ml_core::device::Device;
use arrow_ml_core::tensor::Tensor;
use std::mem;

pub fn softmax(input: &Tensor, axis: i64) -> Result<Tensor> {
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("softmax: tensor has no shape".into()))?;
    let ndim = shape.len();
    if ndim == 0 {
        return Err(KernelError::InvalidArgument(
            "softmax: tensor must be at least 1D".into(),
        ));
    }

    let resolved = if axis < 0 { ndim as i64 + axis } else { axis };
    if resolved < 0 || resolved >= ndim as i64 {
        return Err(KernelError::InvalidArgument(format!(
            "softmax: axis {} out of range for {}D tensor",
            axis, ndim
        )));
    }

    match input.device() {
        Device::Cpu => cpu_softmax(input, shape, resolved as usize),
        device @ (Device::Metal(_) | Device::Cuda(_)) => {
            device_softmax(input, shape, axis as i32, device)
        }
    }
}

fn cpu_softmax(input: &Tensor, shape: &[usize], axis: usize) -> Result<Tensor> {
    match input.data_type() {
        DataType::Float32 => cpu_softmax_typed::<f32>(input, shape, axis),
        DataType::Float64 => cpu_softmax_typed::<f64>(input, shape, axis),
        other => Err(KernelError::InvalidArgument(format!(
            "softmax: unsupported dtype {other:?}"
        ))),
    }
}

fn cpu_softmax_typed<T>(input: &Tensor, shape: &[usize], axis: usize) -> Result<Tensor>
where
    T: arrow::datatypes::ArrowNativeType + num_traits::Float + std::ops::AddAssign,
{
    let data: &[T] = input
        .buffer()
        .typed_data()
        .map_err(|e| KernelError::InvalidArgument(format!("softmax: {e}")))?;
    let ndim = shape.len();

    let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
    let dim_size = shape[axis];
    let inner_size: usize = shape[axis + 1..].iter().product::<usize>().max(1);

    let mut out = data.to_vec();

    for o in 0..outer_size {
        for i in 0..inner_size {
            let mut max_val = T::neg_infinity();
            for d in 0..dim_size {
                let idx = o * dim_size * inner_size + d * inner_size + i;
                if data[idx] > max_val {
                    max_val = data[idx];
                }
            }
            let mut sum = T::zero();
            for d in 0..dim_size {
                let idx = o * dim_size * inner_size + d * inner_size + i;
                let e = (data[idx] - max_val).exp();
                out[idx] = e;
                sum += e;
            }
            for d in 0..dim_size {
                let idx = o * dim_size * inner_size + d * inner_size + i;
                out[idx] = out[idx] / sum;
            }
        }
    }

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

fn device_softmax(input: &Tensor, shape: &[usize], axis: i32, device: Device) -> Result<Tensor> {
    let dtype_code = match input.data_type() {
        DataType::Float32 => dtype::FLOAT32,
        other => {
            return Err(KernelError::InvalidArgument(format!(
                "softmax: unsupported dtype {other:?} for device"
            )))
        }
    };

    let am_dev = device.to_am();
    let kernel =
        BackendRegistry::global().get_kernel::<softmax::Softmax>(dtype_code, am_dev as i32)?;

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
            .invoke(&in_ffi, &mut out_ffi, axis)
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

    #[test]
    fn test_softmax_uniform() {
        let input = make_f32(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
        let output = softmax(&input, 0).unwrap();
        let data = output.buffer().typed_data::<f32>().unwrap();
        for v in data {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let output = softmax(&input, 0).unwrap();
        let sum: f32 = output.buffer().typed_data::<f32>().unwrap().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_ordering() {
        let input = make_f32(vec![1.0, 3.0, 2.0], vec![3]);
        let output = softmax(&input, 0).unwrap();
        let data = output.buffer().typed_data::<f32>().unwrap();
        assert!(data[1] > data[2]);
        assert!(data[2] > data[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let input = make_f32(vec![1000.0, 1001.0, 1002.0], vec![3]);
        let output = softmax(&input, 0).unwrap();
        let data = output.buffer().typed_data::<f32>().unwrap();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(data[2] > data[1]);
        assert!(data[1] > data[0]);
    }

    #[test]
    fn test_softmax_2d_axis1() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3]);
        let out = softmax(&input, 1).unwrap();
        let data = out.buffer().typed_data::<f32>().unwrap();
        let row0_sum: f32 = data[0..3].iter().sum();
        let row1_sum: f32 = data[3..6].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-6);
        assert!((row1_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_2d_axis0() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = softmax(&input, 0).unwrap();
        let data = out.buffer().typed_data::<f32>().unwrap();
        for j in 0..3 {
            let col_sum = data[j] + data[3 + j];
            assert!((col_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_3d_attention() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
        let out = softmax(&input, -1).unwrap();
        let data = out.buffer().typed_data::<f32>().unwrap();
        let sum0: f32 = data[0..3].iter().sum();
        let sum1: f32 = data[3..6].iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-6);
        assert!((sum1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_negative_axis() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = softmax(&input, -1).unwrap();
        let data = out.buffer().typed_data::<f32>().unwrap();
        let row0_sum: f32 = data[0..3].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-6);
    }
}

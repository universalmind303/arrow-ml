use arrow::buffer::Buffer;
use arrow::datatypes::DataType;
use arrow_ml_common::device_tensor::dtype;
use arrow_ml_common::error::{KernelError, Result};
use arrow_ml_common::kernels::layernorm::LayerNormKernel;
use arrow_ml_common::BackendRegistry;
use arrow_ml_core::buffer::DeviceBuffer;
use arrow_ml_core::device::Device;
use arrow_ml_core::tensor::Tensor;
use std::mem;

pub fn layer_norm(
    input: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    axis: i64,
    epsilon: f32,
) -> Result<Tensor> {
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument("layer_norm: tensor has no shape".into()))?;
    let ndim = shape.len();
    if ndim == 0 {
        return Err(KernelError::InvalidArgument(
            "layer_norm: tensor must be at least 1D".into(),
        ));
    }

    let resolved = if axis < 0 { ndim as i64 + axis } else { axis };
    if resolved < 0 || resolved >= ndim as i64 {
        return Err(KernelError::InvalidArgument(format!(
            "layer_norm: axis {} out of range for {}D tensor",
            axis, ndim
        )));
    }

    match input.device() {
        Device::Cpu => cpu_layer_norm(input, gamma, beta, shape, resolved as usize, epsilon),
        device @ (Device::Metal(_) | Device::Cuda(_)) => {
            device_layer_norm(input, gamma, beta, shape, axis as i32, epsilon, device)
        }
    }
}

fn cpu_layer_norm(
    input: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    shape: &[usize],
    axis: usize,
    epsilon: f32,
) -> Result<Tensor> {
    match input.data_type() {
        DataType::Float32 => cpu_layer_norm_typed::<f32>(input, gamma, beta, shape, axis, epsilon),
        DataType::Float64 => cpu_layer_norm_typed::<f64>(input, gamma, beta, shape, axis, epsilon),
        other => Err(KernelError::InvalidArgument(format!(
            "layer_norm: unsupported dtype {other:?}"
        ))),
    }
}

fn cpu_layer_norm_typed<T>(
    input: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    shape: &[usize],
    axis: usize,
    epsilon: f32,
) -> Result<Tensor>
where
    T: arrow::datatypes::ArrowNativeType + num_traits::Float + std::ops::AddAssign,
{
    let data: &[T] = input
        .buffer()
        .typed_data()
        .map_err(|e| KernelError::InvalidArgument(format!("layer_norm: {e}")))?;
    let gamma_data: &[T] = gamma
        .buffer()
        .typed_data()
        .map_err(|e| KernelError::InvalidArgument(format!("layer_norm gamma: {e}")))?;
    let beta_data: &[T] = beta
        .buffer()
        .typed_data()
        .map_err(|e| KernelError::InvalidArgument(format!("layer_norm beta: {e}")))?;

    let ndim = shape.len();
    let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
    let dim_size: usize = shape[axis..].iter().product();
    let eps = T::from(epsilon).unwrap_or_else(|| T::from(1e-5).unwrap());

    if gamma_data.len() != dim_size || beta_data.len() != dim_size {
        return Err(KernelError::InvalidArgument(format!(
            "layer_norm: gamma/beta length {}/{} doesn't match dim_size {}",
            gamma_data.len(),
            beta_data.len(),
            dim_size
        )));
    }

    let mut out = data.to_vec();

    for o in 0..outer_size {
        let base = o * dim_size;
        let row = &data[base..base + dim_size];

        let mut sum = T::zero();
        for &v in row {
            sum += v;
        }
        let mean = sum / T::from(dim_size).unwrap();

        let mut var_sum = T::zero();
        for &v in row {
            let diff = v - mean;
            var_sum += diff * diff;
        }
        let inv_std = (var_sum / T::from(dim_size).unwrap() + eps).sqrt().recip();

        for d in 0..dim_size {
            let normalized = (data[base + d] - mean) * inv_std;
            out[base + d] = normalized * gamma_data[d] + beta_data[d];
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

fn device_layer_norm(
    input: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    shape: &[usize],
    axis: i32,
    epsilon: f32,
    device: Device,
) -> Result<Tensor> {
    let dtype_code = match input.data_type() {
        DataType::Float32 => dtype::FLOAT32,
        other => {
            return Err(KernelError::InvalidArgument(format!(
                "layer_norm: unsupported dtype {other:?} for device"
            )))
        }
    };

    let am_dev = device.to_am();
    let backend = BackendRegistry::global()
        .best_layernorm_for(dtype_code, am_dev as i32)
        .ok_or_else(|| {
            KernelError::InvalidArgument("no layernorm backend for this device".into())
        })?;

    let kernel = LayerNormKernel::open(backend, dtype_code, am_dev as i32)
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
    let gamma_ffi = gamma.as_ffi();
    let beta_ffi = beta.as_ffi();
    let mut out_ffi = output.as_ffi();

    unsafe {
        kernel
            .invoke(&in_ffi, &gamma_ffi, &beta_ffi, &mut out_ffi, axis, epsilon)
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

    fn ones(n: usize) -> Tensor {
        make_f32(vec![1.0; n], vec![n])
    }

    fn zeros(n: usize) -> Tensor {
        make_f32(vec![0.0; n], vec![n])
    }

    #[test]
    fn test_layernorm_identity_params() {
        // gamma=1, beta=0 should just normalize
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]);
        let output = layer_norm(&input, &ones(5), &zeros(5), -1, 1e-5).unwrap();
        let data = output.buffer().typed_data::<f32>().unwrap();

        // Mean = 3, Var = 2, so normalized = [-1.414, -0.707, 0, 0.707, 1.414]
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
    }

    #[test]
    fn test_layernorm_scale_shift() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let gamma = make_f32(vec![2.0, 2.0, 2.0, 2.0], vec![4]);
        let beta = make_f32(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
        let output = layer_norm(&input, &gamma, &beta, -1, 1e-5).unwrap();
        let data = output.buffer().typed_data::<f32>().unwrap();

        // After norm (mean=0), scale by 2 and shift by 1 -> mean should be 1
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(
            (mean - 1.0).abs() < 1e-5,
            "mean should be ~1.0 after shift, got {mean}"
        );
    }

    #[test]
    fn test_layernorm_2d_rows() {
        // 2x3 tensor, normalize over last axis
        let input = make_f32(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], vec![2, 3]);
        let output = layer_norm(&input, &ones(3), &zeros(3), -1, 1e-5).unwrap();
        let data = output.buffer().typed_data::<f32>().unwrap();

        // Each row should have mean ~0
        let row0_mean: f32 = data[0..3].iter().sum::<f32>() / 3.0;
        let row1_mean: f32 = data[3..6].iter().sum::<f32>() / 3.0;
        assert!(row0_mean.abs() < 1e-5, "row 0 mean: {row0_mean}");
        assert!(row1_mean.abs() < 1e-5, "row 1 mean: {row1_mean}");
    }

    #[test]
    fn test_layernorm_constant_input() {
        // All same values -> normalized should be all zeros (with gamma=1, beta=0)
        let input = make_f32(vec![5.0, 5.0, 5.0, 5.0], vec![1, 4]);
        let output = layer_norm(&input, &ones(4), &zeros(4), -1, 1e-5).unwrap();
        let data = output.buffer().typed_data::<f32>().unwrap();

        for v in data {
            assert!(
                v.abs() < 1e-3,
                "constant input should normalize to ~0, got {v}"
            );
        }
    }

    #[test]
    fn test_layernorm_negative_axis() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out_neg = layer_norm(&input, &ones(3), &zeros(3), -1, 1e-5).unwrap();
        let out_pos = layer_norm(&input, &ones(3), &zeros(3), 1, 1e-5).unwrap();
        let d1 = out_neg.buffer().typed_data::<f32>().unwrap();
        let d2 = out_pos.buffer().typed_data::<f32>().unwrap();

        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}

use arrow::array::{ArrowPrimitiveType, PrimitiveArray};
use arrow_kernels_common::Result;
use num_traits::Float;
use std::ops::AddAssign;

use arrow_kernels_activations::{gelu, relu, sigmoid, silu, softmax, tanh};

/// Extension trait for method-style activation functions on `PrimitiveArray`.
///
/// Import this trait to call activations as methods:
/// ```ignore
/// use arrow_kernels::array_ops::ArrayOps;
/// let output = input.relu();
/// let probs = logits.softmax()?;
/// ```
pub trait ArrayOps<T: ArrowPrimitiveType> {
    /// ReLU: max(0, x). Null-propagating.
    fn relu(&self) -> PrimitiveArray<T>
    where
        T::Native: Float;

    /// Leaky ReLU: x if x > 0, else alpha * x. Null-propagating.
    fn leaky_relu(&self, alpha: T::Native) -> PrimitiveArray<T>
    where
        T::Native: Float;

    /// Sigmoid: 1 / (1 + exp(-x)). Null-propagating.
    fn sigmoid(&self) -> PrimitiveArray<T>
    where
        T::Native: Float;

    /// Tanh activation. Null-propagating.
    fn tanh_act(&self) -> PrimitiveArray<T>
    where
        T::Native: Float;

    /// GELU (tanh approximation). Null-propagating.
    fn gelu(&self) -> PrimitiveArray<T>
    where
        T::Native: Float;

    /// GELU (exact, erf-based). Null-propagating.
    fn gelu_exact(&self) -> PrimitiveArray<T>
    where
        T::Native: Float;

    /// SiLU (swish): x * sigmoid(x). Null-propagating.
    fn silu(&self) -> PrimitiveArray<T>
    where
        T::Native: Float;

    /// Softmax over all elements. Returns error on nulls or empty.
    fn softmax(&self) -> Result<PrimitiveArray<T>>
    where
        T::Native: Float + AddAssign;
}

impl<T: ArrowPrimitiveType> ArrayOps<T> for PrimitiveArray<T> {
    fn relu(&self) -> PrimitiveArray<T>
    where
        T::Native: Float,
    {
        relu::relu(self)
    }

    fn leaky_relu(&self, alpha: T::Native) -> PrimitiveArray<T>
    where
        T::Native: Float,
    {
        relu::leaky_relu(self, alpha)
    }

    fn sigmoid(&self) -> PrimitiveArray<T>
    where
        T::Native: Float,
    {
        sigmoid::sigmoid(self)
    }

    fn tanh_act(&self) -> PrimitiveArray<T>
    where
        T::Native: Float,
    {
        tanh::tanh(self)
    }

    fn gelu(&self) -> PrimitiveArray<T>
    where
        T::Native: Float,
    {
        gelu::gelu(self)
    }

    fn gelu_exact(&self) -> PrimitiveArray<T>
    where
        T::Native: Float,
    {
        gelu::gelu_exact(self)
    }

    fn silu(&self) -> PrimitiveArray<T>
    where
        T::Native: Float,
    {
        silu::silu(self)
    }

    fn softmax(&self) -> Result<PrimitiveArray<T>>
    where
        T::Native: Float + AddAssign,
    {
        softmax::softmax(self)
    }
}

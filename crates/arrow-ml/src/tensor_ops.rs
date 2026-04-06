use arrow::array::ArrowPrimitiveType;
use arrow::tensor::Tensor;
use arrow_ml_common::Result;
use num_traits::Float;
use std::ops::{Add, AddAssign, Mul};

use arrow_ml_linalg::{
    binary_arithmetic, clip, cumsum, elementwise_math, expand, reduce, reshape, rounding,
    transpose,
};

/// Extension trait for method-style chaining on `Tensor`.
///
/// Import this trait to call operations as methods instead of free functions:
/// ```ignore
/// use arrow_ml::tensor_ops::TensorOps;
/// let result = input.dot(&weights)?.t()?.sum(&[-1], false)?;
/// ```
pub trait TensorOps<T: ArrowPrimitiveType> {
    // --- Matrix / linear algebra ---

    /// Matrix multiply: `self @ other`. Delegates to `matmul::matmul`.
    fn dot(&self, other: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
    where
        T::Native: Float
            + num_traits::Zero
            + num_traits::One
            + Copy
            + Mul<Output = T::Native>
            + Add<Output = T::Native>
            + AddAssign;

    /// Matrix-vector multiply. Delegates to `matmul::matvec`.
    fn matvec(&self, x: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
    where
        T::Native: Float
            + num_traits::Zero
            + num_traits::One
            + Copy
            + Mul<Output = T::Native>
            + Add<Output = T::Native>
            + AddAssign;

    /// Transpose a 2D tensor. Delegates to `transpose::transpose`.
    fn t(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    // --- Reshaping ---

    /// Reshape to a new shape (supports -1 for inference).
    fn reshape(&self, shape: &[i64]) -> Result<Tensor<'static, T>>
    where
        T::Native: Copy;

    /// Flatten from `axis` onward into a single dimension.
    fn flatten(&self, axis: i64) -> Result<Tensor<'static, T>>
    where
        T::Native: Copy;

    /// Remove size-1 dimensions.
    fn squeeze(&self, axes: Option<&[i64]>) -> Result<Tensor<'static, T>>
    where
        T::Native: Copy;

    /// Insert size-1 dimensions.
    fn unsqueeze(&self, axes: &[i64]) -> Result<Tensor<'static, T>>
    where
        T::Native: Copy;

    /// Broadcast to a target shape.
    fn expand(&self, shape: &[usize]) -> Result<Tensor<'static, T>>
    where
        T::Native: Copy;

    // --- Reductions ---

    /// Sum along axes.
    fn sum(&self, axes: &[i64], keepdims: bool) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Mean along axes.
    fn mean(&self, axes: &[i64], keepdims: bool) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Max along axes.
    fn reduce_max(&self, axes: &[i64], keepdims: bool) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Min along axes.
    fn reduce_min(&self, axes: &[i64], keepdims: bool) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Product along axes.
    fn prod(&self, axes: &[i64], keepdims: bool) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Cumulative sum along axis.
    fn cumsum(&self, axis: i64) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    // --- Elementwise math ---

    /// Raise to scalar power.
    fn pow(&self, exponent: T::Native) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Error function (Abramowitz & Stegun approximation).
    fn erf(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Element-wise 1/x.
    fn reciprocal(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Element-wise cosine.
    fn cos(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Element-wise sine.
    fn sin(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Element-wise floor.
    fn floor(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Element-wise ceiling.
    fn ceil(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Element-wise round to nearest.
    fn round(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Clamp values to [min, max].
    fn clip(&self, min: Option<T::Native>, max: Option<T::Native>) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Softmax along an axis.
    fn softmax(&self, axis: i64) -> Result<Tensor<'static, T>>
    where
        T::Native: Float + AddAssign;

    // --- Binary arithmetic with broadcasting ---

    /// Element-wise addition with broadcasting.
    fn add(&self, other: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Element-wise subtraction with broadcasting.
    fn sub(&self, other: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Element-wise multiplication with broadcasting.
    fn mul(&self, other: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Element-wise division with broadcasting.
    fn div(&self, other: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Element-wise square root.
    fn sqrt(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Element-wise exponential (e^x).
    fn exp(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Element-wise natural logarithm.
    fn log(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float;

    /// Transpose with arbitrary axis permutation.
    fn transpose_axes(&self, perm: &[usize]) -> Result<Tensor<'static, T>>
    where
        T::Native: Copy;
}

impl<T: ArrowPrimitiveType> TensorOps<T> for Tensor<'_, T> {
    // --- Matrix / linear algebra ---

    fn dot(&self, other: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
    where
        T::Native: Float
            + num_traits::Zero
            + num_traits::One
            + Copy
            + Mul<Output = T::Native>
            + Add<Output = T::Native>
            + AddAssign,
    {
        arrow_ml_linalg::matmul::matmul(self, other)
    }

    fn matvec(&self, x: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
    where
        T::Native: Float
            + num_traits::Zero
            + num_traits::One
            + Copy
            + Mul<Output = T::Native>
            + Add<Output = T::Native>
            + AddAssign,
    {
        arrow_ml_linalg::matmul::matvec(self, x)
    }

    fn t(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        transpose::transpose(self)
    }

    // --- Reshaping ---

    fn reshape(&self, shape: &[i64]) -> Result<Tensor<'static, T>>
    where
        T::Native: Copy,
    {
        reshape::reshape(self, shape)
    }

    fn flatten(&self, axis: i64) -> Result<Tensor<'static, T>>
    where
        T::Native: Copy,
    {
        reshape::flatten(self, axis)
    }

    fn squeeze(&self, axes: Option<&[i64]>) -> Result<Tensor<'static, T>>
    where
        T::Native: Copy,
    {
        reshape::squeeze(self, axes)
    }

    fn unsqueeze(&self, axes: &[i64]) -> Result<Tensor<'static, T>>
    where
        T::Native: Copy,
    {
        reshape::unsqueeze(self, axes)
    }

    fn expand(&self, shape: &[usize]) -> Result<Tensor<'static, T>>
    where
        T::Native: Copy,
    {
        expand::expand(self, shape)
    }

    // --- Reductions ---

    fn sum(&self, axes: &[i64], keepdims: bool) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        reduce::reduce_sum(self, axes, keepdims)
    }

    fn mean(&self, axes: &[i64], keepdims: bool) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        reduce::reduce_mean(self, axes, keepdims)
    }

    fn reduce_max(&self, axes: &[i64], keepdims: bool) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        reduce::reduce_max(self, axes, keepdims)
    }

    fn reduce_min(&self, axes: &[i64], keepdims: bool) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        reduce::reduce_min(self, axes, keepdims)
    }

    fn prod(&self, axes: &[i64], keepdims: bool) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        reduce::reduce_prod(self, axes, keepdims)
    }

    fn cumsum(&self, axis: i64) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        cumsum::cumsum(self, axis)
    }

    // --- Elementwise math ---

    fn pow(&self, exponent: T::Native) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        elementwise_math::pow(self, exponent)
    }

    fn erf(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        elementwise_math::erf(self)
    }

    fn reciprocal(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        elementwise_math::reciprocal(self)
    }

    fn cos(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        elementwise_math::cos_op(self)
    }

    fn sin(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        elementwise_math::sin_op(self)
    }

    fn floor(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        rounding::floor(self)
    }

    fn ceil(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        rounding::ceil(self)
    }

    fn round(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        rounding::round(self)
    }

    fn clip(&self, min: Option<T::Native>, max: Option<T::Native>) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        clip::clip(self, min, max)
    }

    fn softmax(&self, axis: i64) -> Result<Tensor<'static, T>>
    where
        T::Native: Float + AddAssign,
    {
        arrow_ml_activations::softmax::softmax_tensor(self, axis)
    }

    fn add(&self, other: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        binary_arithmetic::add(self, other)
    }

    fn sub(&self, other: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        binary_arithmetic::sub(self, other)
    }

    fn mul(&self, other: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        binary_arithmetic::mul(self, other)
    }

    fn div(&self, other: &Tensor<'_, T>) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        binary_arithmetic::div(self, other)
    }

    fn sqrt(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        elementwise_math::sqrt(self)
    }

    fn exp(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        elementwise_math::exp(self)
    }

    fn log(&self) -> Result<Tensor<'static, T>>
    where
        T::Native: Float,
    {
        elementwise_math::log(self)
    }

    fn transpose_axes(&self, perm: &[usize]) -> Result<Tensor<'static, T>>
    where
        T::Native: Copy,
    {
        transpose::transpose_axes(self, perm)
    }
}

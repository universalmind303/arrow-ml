use std::simd::prelude::*;

use arrow::array::ArrowPrimitiveType;
use arrow::buffer::Buffer;
use arrow::datatypes::DataType;
use arrow::tensor::Tensor;
use arrow_ml_common::{KernelError, Result};
use num_traits::{Float, One, Zero};

/// Minimum reduction-axis length to switch from the scalar path to the SIMD path.
/// Below this threshold the SIMD setup cost is not worth the savings.
const SIMD_THRESHOLD: usize = 32;

// ---------------------------------------------------------------------------
// Operation tag – drives typed (SIMD-aware) dispatch.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
}

// ---------------------------------------------------------------------------
// SIMD helpers – f32 (8-lane)
// ---------------------------------------------------------------------------

/// Horizontal sum of a contiguous f32 slice using 8-lane SIMD.
fn simd_sum_f32(vals: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut acc = Simd::<f32, LANES>::splat(0.0_f32);
    let mut chunks = vals.chunks_exact(LANES);
    for chunk in chunks.by_ref() {
        acc += Simd::<f32, LANES>::from_slice(chunk);
    }
    let mut s = acc.reduce_sum();
    for &v in chunks.remainder() {
        s += v;
    }
    s
}

/// Horizontal maximum of a contiguous f32 slice using 8-lane SIMD.
fn simd_max_f32(vals: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut acc = Simd::<f32, LANES>::splat(f32::NEG_INFINITY);
    let mut chunks = vals.chunks_exact(LANES);
    for chunk in chunks.by_ref() {
        acc = acc.simd_max(Simd::<f32, LANES>::from_slice(chunk));
    }
    let mut m = acc.reduce_max();
    for &v in chunks.remainder() {
        if v > m {
            m = v;
        }
    }
    m
}

/// Horizontal minimum of a contiguous f32 slice using 8-lane SIMD.
fn simd_min_f32(vals: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut acc = Simd::<f32, LANES>::splat(f32::INFINITY);
    let mut chunks = vals.chunks_exact(LANES);
    for chunk in chunks.by_ref() {
        acc = acc.simd_min(Simd::<f32, LANES>::from_slice(chunk));
    }
    let mut m = acc.reduce_min();
    for &v in chunks.remainder() {
        if v < m {
            m = v;
        }
    }
    m
}

/// Horizontal product of a contiguous f32 slice using 8-lane SIMD.
fn simd_prod_f32(vals: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut acc = Simd::<f32, LANES>::splat(1.0_f32);
    let mut chunks = vals.chunks_exact(LANES);
    for chunk in chunks.by_ref() {
        acc *= Simd::<f32, LANES>::from_slice(chunk);
    }
    let mut p = acc.reduce_product();
    for &v in chunks.remainder() {
        p *= v;
    }
    p
}

// ---------------------------------------------------------------------------
// SIMD helpers – f64 (4-lane)
// ---------------------------------------------------------------------------

/// Horizontal sum of a contiguous f64 slice using 4-lane SIMD.
fn simd_sum_f64(vals: &[f64]) -> f64 {
    const LANES: usize = 4;
    let mut acc = Simd::<f64, LANES>::splat(0.0_f64);
    let mut chunks = vals.chunks_exact(LANES);
    for chunk in chunks.by_ref() {
        acc += Simd::<f64, LANES>::from_slice(chunk);
    }
    let mut s = acc.reduce_sum();
    for &v in chunks.remainder() {
        s += v;
    }
    s
}

/// Horizontal maximum of a contiguous f64 slice using 4-lane SIMD.
fn simd_max_f64(vals: &[f64]) -> f64 {
    const LANES: usize = 4;
    let mut acc = Simd::<f64, LANES>::splat(f64::NEG_INFINITY);
    let mut chunks = vals.chunks_exact(LANES);
    for chunk in chunks.by_ref() {
        acc = acc.simd_max(Simd::<f64, LANES>::from_slice(chunk));
    }
    let mut m = acc.reduce_max();
    for &v in chunks.remainder() {
        if v > m {
            m = v;
        }
    }
    m
}

/// Horizontal minimum of a contiguous f64 slice using 4-lane SIMD.
fn simd_min_f64(vals: &[f64]) -> f64 {
    const LANES: usize = 4;
    let mut acc = Simd::<f64, LANES>::splat(f64::INFINITY);
    let mut chunks = vals.chunks_exact(LANES);
    for chunk in chunks.by_ref() {
        acc = acc.simd_min(Simd::<f64, LANES>::from_slice(chunk));
    }
    let mut m = acc.reduce_min();
    for &v in chunks.remainder() {
        if v < m {
            m = v;
        }
    }
    m
}

/// Horizontal product of a contiguous f64 slice using 4-lane SIMD.
fn simd_prod_f64(vals: &[f64]) -> f64 {
    const LANES: usize = 4;
    let mut acc = Simd::<f64, LANES>::splat(1.0_f64);
    let mut chunks = vals.chunks_exact(LANES);
    for chunk in chunks.by_ref() {
        acc *= Simd::<f64, LANES>::from_slice(chunk);
    }
    let mut p = acc.reduce_product();
    for &v in chunks.remainder() {
        p *= v;
    }
    p
}

// ---------------------------------------------------------------------------
// Dispatch helpers: apply one reduction operation to a slice, picking SIMD or
// scalar based on `use_simd`.
// ---------------------------------------------------------------------------

/// Applies one reduce operation to a `f32` slice, choosing the SIMD or scalar
/// path based on `use_simd`.
///
/// `use_simd` should be `true` when `vals.len() >= SIMD_THRESHOLD`.
#[inline]
fn apply_f32(vals: &[f32], op: ReduceOp, use_simd: bool) -> f32 {
    if use_simd {
        match op {
            ReduceOp::Sum => simd_sum_f32(vals),
            ReduceOp::Mean => simd_sum_f32(vals) / vals.len() as f32,
            ReduceOp::Max => simd_max_f32(vals),
            ReduceOp::Min => simd_min_f32(vals),
            ReduceOp::Prod => simd_prod_f32(vals),
        }
    } else {
        match op {
            ReduceOp::Sum => vals.iter().sum(),
            ReduceOp::Mean => vals.iter().sum::<f32>() / vals.len() as f32,
            ReduceOp::Max => vals
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, |a, b| if b > a { b } else { a }),
            ReduceOp::Min => vals
                .iter()
                .copied()
                .fold(f32::INFINITY, |a, b| if b < a { b } else { a }),
            ReduceOp::Prod => vals.iter().product(),
        }
    }
}

/// Applies one reduce operation to a `f64` slice, choosing the SIMD or scalar
/// path based on `use_simd`.
///
/// `use_simd` should be `true` when `vals.len() >= SIMD_THRESHOLD`.
#[inline]
fn apply_f64(vals: &[f64], op: ReduceOp, use_simd: bool) -> f64 {
    if use_simd {
        match op {
            ReduceOp::Sum => simd_sum_f64(vals),
            ReduceOp::Mean => simd_sum_f64(vals) / vals.len() as f64,
            ReduceOp::Max => simd_max_f64(vals),
            ReduceOp::Min => simd_min_f64(vals),
            ReduceOp::Prod => simd_prod_f64(vals),
        }
    } else {
        match op {
            ReduceOp::Sum => vals.iter().sum(),
            ReduceOp::Mean => vals.iter().sum::<f64>() / vals.len() as f64,
            ReduceOp::Max => vals
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a }),
            ReduceOp::Min => vals
                .iter()
                .copied()
                .fold(f64::INFINITY, |a, b| if b < a { b } else { a }),
            ReduceOp::Prod => vals.iter().product(),
        }
    }
}

// ---------------------------------------------------------------------------
// Typed single-axis reducers (f32 / f64)
//
// For the contiguous case (inner_size == 1) we pass the data slice directly
// to the reduction function, avoiding any intermediate copy.  For the
// non-contiguous case we still gather into a temporary buffer.
// ---------------------------------------------------------------------------

/// Reduce `data` (shaped by `shape`) along `axis` using a typed f32 kernel.
///
/// When `inner_size == 1` the data for each output element is contiguous so
/// the reduction function receives a direct slice with no intermediate copy.
/// Otherwise elements are gathered into a reusable temporary buffer.
///
/// Returns `(output_data, new_shape)`.
fn reduce_single_axis_f32(
    data: &[f32],
    shape: &[usize],
    axis: usize,
    op: ReduceOp,
) -> (Vec<f32>, Vec<usize>) {
    let outer_size: usize = {
        let n: usize = shape[..axis].iter().product();
        if n == 0 { 1 } else { n }
    };
    let reduce_size = shape[axis];
    let inner_size: usize = {
        let n: usize = shape[axis + 1..].iter().product();
        if n == 0 { 1 } else { n }
    };
    let use_simd = reduce_size >= SIMD_THRESHOLD;
    let mut out = Vec::with_capacity(outer_size * inner_size);

    if inner_size == 1 {
        // Contiguous: slice the data directly, no copy needed.
        for o in 0..outer_size {
            let slice = &data[o * reduce_size..(o + 1) * reduce_size];
            out.push(apply_f32(slice, op, use_simd));
        }
    } else {
        // Non-contiguous: gather elements into a reusable tmp buffer.
        let mut tmp = Vec::with_capacity(reduce_size);
        for o in 0..outer_size {
            for i in 0..inner_size {
                tmp.clear();
                for r in 0..reduce_size {
                    tmp.push(data[o * reduce_size * inner_size + r * inner_size + i]);
                }
                out.push(apply_f32(&tmp, op, use_simd));
            }
        }
    }

    let mut new_shape: Vec<usize> = (0..shape.len())
        .filter(|&d| d != axis)
        .map(|d| shape[d])
        .collect();
    if new_shape.is_empty() {
        new_shape.push(1);
    }
    (out, new_shape)
}

/// Reduce `data` (shaped by `shape`) along `axis` using a typed f64 kernel.
///
/// When `inner_size == 1` the data for each output element is contiguous so
/// the reduction function receives a direct slice with no intermediate copy.
/// Otherwise elements are gathered into a reusable temporary buffer.
///
/// Returns `(output_data, new_shape)`.
fn reduce_single_axis_f64(
    data: &[f64],
    shape: &[usize],
    axis: usize,
    op: ReduceOp,
) -> (Vec<f64>, Vec<usize>) {
    let outer_size: usize = {
        let n: usize = shape[..axis].iter().product();
        if n == 0 { 1 } else { n }
    };
    let reduce_size = shape[axis];
    let inner_size: usize = {
        let n: usize = shape[axis + 1..].iter().product();
        if n == 0 { 1 } else { n }
    };
    let use_simd = reduce_size >= SIMD_THRESHOLD;
    let mut out = Vec::with_capacity(outer_size * inner_size);

    if inner_size == 1 {
        for o in 0..outer_size {
            let slice = &data[o * reduce_size..(o + 1) * reduce_size];
            out.push(apply_f64(slice, op, use_simd));
        }
    } else {
        let mut tmp = Vec::with_capacity(reduce_size);
        for o in 0..outer_size {
            for i in 0..inner_size {
                tmp.clear();
                for r in 0..reduce_size {
                    tmp.push(data[o * reduce_size * inner_size + r * inner_size + i]);
                }
                out.push(apply_f64(&tmp, op, use_simd));
            }
        }
    }

    let mut new_shape: Vec<usize> = (0..shape.len())
        .filter(|&d| d != axis)
        .map(|d| shape[d])
        .collect();
    if new_shape.is_empty() {
        new_shape.push(1);
    }
    (out, new_shape)
}

// ---------------------------------------------------------------------------
// Shared shape-management logic for the typed (f32 / f64) path.
//
// `single_axis` is called once per axis being reduced.  Returns the final
// (data, shape) pair ready to be wrapped into a Buffer/Tensor.
// ---------------------------------------------------------------------------

/// Shared shape-management pipeline for the typed (f32 / f64) SIMD path.
///
/// Handles axis normalisation, multi-axis reduction (one axis at a time, high
/// to low so indices stay valid), and the `keepdims` shape rebuild.
///
/// `single_axis` is called once per axis being reduced; it receives the
/// current `(data, shape, axis)` and must return the updated `(data, shape)`.
///
/// Returns `(final_data, final_shape)` ready to be wrapped into a `Buffer`.
fn reduce_typed<N, F>(
    data: &[N],
    shape: &[usize],
    axes: &[i64],
    keepdims: bool,
    op: &str,
    single_axis: F,
) -> Result<(Vec<N>, Vec<usize>)>
where
    N: Clone,
    F: Fn(&[N], &[usize], usize) -> (Vec<N>, Vec<usize>),
{
    let ndim = shape.len();
    let mut resolved = normalize_axes(axes, ndim, op)?;
    resolved.sort_unstable();
    resolved.dedup();

    let mut current_data: Vec<N> = data.to_vec();
    let mut current_shape = shape.to_vec();

    for &ax in resolved.iter().rev() {
        let (new_data, new_shape) = single_axis(&current_data, &current_shape, ax);
        current_data = new_data;
        current_shape = new_shape;
    }

    let final_shape = if keepdims {
        let mut s = shape.to_vec();
        for &ax in &resolved {
            s[ax] = 1;
        }
        s
    } else {
        current_shape
    };
    let final_shape = if final_shape.is_empty() {
        vec![1]
    } else {
        final_shape
    };

    Ok((current_data, final_shape))
}

// ---------------------------------------------------------------------------
// Public API
//
// Each function dispatches to the typed SIMD path for f32/f64 and falls back
// to the generic scalar path for all other element types.
// ---------------------------------------------------------------------------

/// Reduce by sum along specified axes.
pub fn reduce_sum<T>(
    input: &Tensor<'_, T>,
    axes: &[i64],
    keepdims: bool,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    match T::DATA_TYPE {
        DataType::Float32 => {
            let shape = input.shape().ok_or_else(|| {
                KernelError::InvalidArgument("reduce_sum: tensor has no shape".into())
            })?;
            let data: &[f32] = input.data().typed_data();
            let (out, final_shape) = reduce_typed(data, shape, axes, keepdims, "reduce_sum", |d, s, ax| {
                reduce_single_axis_f32(d, s, ax, ReduceOp::Sum)
            })?;
            let buf = Buffer::from_vec(out);
            Tensor::new_row_major(buf, Some(final_shape), None).map_err(KernelError::from)
        }
        DataType::Float64 => {
            let shape = input.shape().ok_or_else(|| {
                KernelError::InvalidArgument("reduce_sum: tensor has no shape".into())
            })?;
            let data: &[f64] = input.data().typed_data();
            let (out, final_shape) = reduce_typed(data, shape, axes, keepdims, "reduce_sum", |d, s, ax| {
                reduce_single_axis_f64(d, s, ax, ReduceOp::Sum)
            })?;
            let buf = Buffer::from_vec(out);
            Tensor::new_row_major(buf, Some(final_shape), None).map_err(KernelError::from)
        }
        _ => reduce_impl(input, axes, keepdims, "reduce_sum", |vals| {
            vals.iter()
                .copied()
                .fold(<T::Native as Zero>::zero(), |a, b| a + b)
        }),
    }
}

/// Reduce by mean along specified axes.
pub fn reduce_mean<T>(
    input: &Tensor<'_, T>,
    axes: &[i64],
    keepdims: bool,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    match T::DATA_TYPE {
        DataType::Float32 => {
            let shape = input.shape().ok_or_else(|| {
                KernelError::InvalidArgument("reduce_mean: tensor has no shape".into())
            })?;
            let data: &[f32] = input.data().typed_data();
            let (out, final_shape) = reduce_typed(data, shape, axes, keepdims, "reduce_mean", |d, s, ax| {
                reduce_single_axis_f32(d, s, ax, ReduceOp::Mean)
            })?;
            let buf = Buffer::from_vec(out);
            Tensor::new_row_major(buf, Some(final_shape), None).map_err(KernelError::from)
        }
        DataType::Float64 => {
            let shape = input.shape().ok_or_else(|| {
                KernelError::InvalidArgument("reduce_mean: tensor has no shape".into())
            })?;
            let data: &[f64] = input.data().typed_data();
            let (out, final_shape) = reduce_typed(data, shape, axes, keepdims, "reduce_mean", |d, s, ax| {
                reduce_single_axis_f64(d, s, ax, ReduceOp::Mean)
            })?;
            let buf = Buffer::from_vec(out);
            Tensor::new_row_major(buf, Some(final_shape), None).map_err(KernelError::from)
        }
        _ => reduce_impl(input, axes, keepdims, "reduce_mean", |vals| {
            let sum = vals
                .iter()
                .copied()
                .fold(<T::Native as Zero>::zero(), |a, b| a + b);
            let n = <T::Native as num_traits::NumCast>::from(vals.len()).unwrap();
            sum / n
        }),
    }
}

/// Reduce by max along specified axes.
pub fn reduce_max<T>(
    input: &Tensor<'_, T>,
    axes: &[i64],
    keepdims: bool,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    match T::DATA_TYPE {
        DataType::Float32 => {
            let shape = input.shape().ok_or_else(|| {
                KernelError::InvalidArgument("reduce_max: tensor has no shape".into())
            })?;
            let data: &[f32] = input.data().typed_data();
            let (out, final_shape) = reduce_typed(data, shape, axes, keepdims, "reduce_max", |d, s, ax| {
                reduce_single_axis_f32(d, s, ax, ReduceOp::Max)
            })?;
            let buf = Buffer::from_vec(out);
            Tensor::new_row_major(buf, Some(final_shape), None).map_err(KernelError::from)
        }
        DataType::Float64 => {
            let shape = input.shape().ok_or_else(|| {
                KernelError::InvalidArgument("reduce_max: tensor has no shape".into())
            })?;
            let data: &[f64] = input.data().typed_data();
            let (out, final_shape) = reduce_typed(data, shape, axes, keepdims, "reduce_max", |d, s, ax| {
                reduce_single_axis_f64(d, s, ax, ReduceOp::Max)
            })?;
            let buf = Buffer::from_vec(out);
            Tensor::new_row_major(buf, Some(final_shape), None).map_err(KernelError::from)
        }
        _ => reduce_impl(input, axes, keepdims, "reduce_max", |vals| {
            vals.iter()
                .copied()
                .fold(T::Native::neg_infinity(), |a, b| if b > a { b } else { a })
        }),
    }
}

/// Reduce by min along specified axes.
pub fn reduce_min<T>(
    input: &Tensor<'_, T>,
    axes: &[i64],
    keepdims: bool,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    match T::DATA_TYPE {
        DataType::Float32 => {
            let shape = input.shape().ok_or_else(|| {
                KernelError::InvalidArgument("reduce_min: tensor has no shape".into())
            })?;
            let data: &[f32] = input.data().typed_data();
            let (out, final_shape) = reduce_typed(data, shape, axes, keepdims, "reduce_min", |d, s, ax| {
                reduce_single_axis_f32(d, s, ax, ReduceOp::Min)
            })?;
            let buf = Buffer::from_vec(out);
            Tensor::new_row_major(buf, Some(final_shape), None).map_err(KernelError::from)
        }
        DataType::Float64 => {
            let shape = input.shape().ok_or_else(|| {
                KernelError::InvalidArgument("reduce_min: tensor has no shape".into())
            })?;
            let data: &[f64] = input.data().typed_data();
            let (out, final_shape) = reduce_typed(data, shape, axes, keepdims, "reduce_min", |d, s, ax| {
                reduce_single_axis_f64(d, s, ax, ReduceOp::Min)
            })?;
            let buf = Buffer::from_vec(out);
            Tensor::new_row_major(buf, Some(final_shape), None).map_err(KernelError::from)
        }
        _ => reduce_impl(input, axes, keepdims, "reduce_min", |vals| {
            vals.iter()
                .copied()
                .fold(T::Native::infinity(), |a, b| if b < a { b } else { a })
        }),
    }
}

/// Reduce by product along specified axes.
pub fn reduce_prod<T>(
    input: &Tensor<'_, T>,
    axes: &[i64],
    keepdims: bool,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    match T::DATA_TYPE {
        DataType::Float32 => {
            let shape = input.shape().ok_or_else(|| {
                KernelError::InvalidArgument("reduce_prod: tensor has no shape".into())
            })?;
            let data: &[f32] = input.data().typed_data();
            let (out, final_shape) = reduce_typed(data, shape, axes, keepdims, "reduce_prod", |d, s, ax| {
                reduce_single_axis_f32(d, s, ax, ReduceOp::Prod)
            })?;
            let buf = Buffer::from_vec(out);
            Tensor::new_row_major(buf, Some(final_shape), None).map_err(KernelError::from)
        }
        DataType::Float64 => {
            let shape = input.shape().ok_or_else(|| {
                KernelError::InvalidArgument("reduce_prod: tensor has no shape".into())
            })?;
            let data: &[f64] = input.data().typed_data();
            let (out, final_shape) = reduce_typed(data, shape, axes, keepdims, "reduce_prod", |d, s, ax| {
                reduce_single_axis_f64(d, s, ax, ReduceOp::Prod)
            })?;
            let buf = Buffer::from_vec(out);
            Tensor::new_row_major(buf, Some(final_shape), None).map_err(KernelError::from)
        }
        _ => reduce_impl(input, axes, keepdims, "reduce_prod", |vals| {
            vals.iter()
                .copied()
                .fold(<T::Native as One>::one(), |a, b| a * b)
        }),
    }
}

// ---------------------------------------------------------------------------
// Generic scalar fallback (used for non-f32/f64 element types)
// ---------------------------------------------------------------------------

fn reduce_impl<T, F>(
    input: &Tensor<'_, T>,
    axes: &[i64],
    keepdims: bool,
    op: &str,
    reducer: F,
) -> Result<Tensor<'static, T>>
where
    T: ArrowPrimitiveType,
    T::Native: Float,
    F: Fn(&[T::Native]) -> T::Native,
{
    let shape = input
        .shape()
        .ok_or_else(|| KernelError::InvalidArgument(format!("{op}: tensor has no shape")))?;
    let ndim = shape.len();

    let mut sorted = normalize_axes(axes, ndim, op)?;
    sorted.sort_unstable();
    sorted.dedup();

    let mut current_data: Vec<T::Native> = input.data().typed_data::<T::Native>().to_vec();
    let mut current_shape = shape.to_vec();

    for &ax in sorted.iter().rev() {
        let (new_data, new_shape) =
            reduce_single_axis::<T, F>(&current_data, &current_shape, ax, &reducer);
        current_data = new_data;
        current_shape = new_shape;
    }

    let final_shape = if keepdims {
        let mut s = shape.to_vec();
        for &ax in &sorted {
            s[ax] = 1;
        }
        s
    } else {
        current_shape
    };
    let final_shape = if final_shape.is_empty() {
        vec![1]
    } else {
        final_shape
    };

    let buf = Buffer::from_vec(current_data);
    Tensor::new_row_major(buf, Some(final_shape), None).map_err(KernelError::from)
}

fn normalize_axes(axes: &[i64], ndim: usize, op: &str) -> Result<Vec<usize>> {
    axes.iter()
        .map(|&a| {
            let r = if a < 0 { ndim as i64 + a } else { a };
            if r < 0 || r >= ndim as i64 {
                Err(KernelError::InvalidArgument(format!(
                    "{op}: axis {a} out of range for {ndim}D tensor"
                )))
            } else {
                Ok(r as usize)
            }
        })
        .collect()
}

fn reduce_single_axis<T, F>(
    data: &[T::Native],
    shape: &[usize],
    axis: usize,
    reducer: &F,
) -> (Vec<T::Native>, Vec<usize>)
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
    F: Fn(&[T::Native]) -> T::Native,
{
    let ndim = shape.len();
    let outer_size: usize = shape[..axis].iter().product();
    let reduce_size = shape[axis];
    let inner_size: usize = shape[axis + 1..].iter().product();

    let mut out = Vec::with_capacity(outer_size * inner_size);
    let mut tmp = Vec::with_capacity(reduce_size);

    for o in 0..outer_size {
        for i in 0..inner_size {
            tmp.clear();
            for r in 0..reduce_size {
                tmp.push(data[o * reduce_size * inner_size + r * inner_size + i]);
            }
            out.push(reducer(&tmp));
        }
    }

    let mut new_shape: Vec<usize> = (0..ndim).filter(|&d| d != axis).map(|d| shape[d]).collect();
    if new_shape.is_empty() {
        new_shape.push(1);
    }

    (out, new_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::Float32Type;

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<'static, Float32Type> {
        let buffer = Buffer::from(ScalarBuffer::<f32>::from(data).into_inner());
        Tensor::new_row_major(buffer, Some(shape), None).unwrap()
    }

    #[test]
    fn test_reduce_sum_axis0() {
        // [[1,2,3],[4,5,6]] -> sum axis 0 -> [5,7,9]
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_sum(&input, &[0], false).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![3]);
        assert_eq!(out.data().typed_data::<f32>(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_reduce_sum_axis1() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_sum(&input, &[1], false).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2]);
        assert_eq!(out.data().typed_data::<f32>(), &[6.0, 15.0]);
    }

    #[test]
    fn test_reduce_sum_keepdims() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_sum(&input, &[0], true).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 3]);
        assert_eq!(out.data().typed_data::<f32>(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_reduce_sum_all_axes() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_sum(&input, &[0, 1], false).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[21.0]);
    }

    #[test]
    fn test_reduce_mean_axis0() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_mean(&input, &[0], false).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_reduce_max_axis1() {
        let input = make_f32(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]);
        let out = reduce_max(&input, &[1], false).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[5.0, 6.0]);
    }

    #[test]
    fn test_reduce_min_axis0() {
        let input = make_f32(vec![3.0, 1.0, 5.0, 2.0, 4.0, 0.0], vec![2, 3]);
        let out = reduce_min(&input, &[0], false).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[2.0, 1.0, 0.0]);
    }

    #[test]
    fn test_reduce_negative_axis() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_sum(&input, &[-1], false).unwrap();
        assert_eq!(out.data().typed_data::<f32>(), &[6.0, 15.0]);
    }

    #[test]
    fn test_reduce_3d() {
        // 2x2x3 tensor, reduce axis 1
        let input = make_f32(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 2, 3],
        );
        let out = reduce_sum(&input, &[1], false).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2, 3]);
        assert_eq!(
            out.data().typed_data::<f32>(),
            &[5.0, 7.0, 9.0, 17.0, 19.0, 21.0]
        );
    }

    #[test]
    fn test_reduce_keepdims_all() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_sum(&input, &[0, 1], true).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![1, 1]);
        assert_eq!(out.data().typed_data::<f32>(), &[21.0]);
    }

    #[test]
    fn test_reduce_prod_axis1() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_prod(&input, &[1], false).unwrap();
        assert_eq!(out.shape().unwrap(), &vec![2]);
        let data = out.data().typed_data::<f32>();
        assert!((data[0] - 6.0).abs() < 1e-6); // 1*2*3
        assert!((data[1] - 120.0).abs() < 1e-4); // 4*5*6
    }

    #[test]
    fn test_reduce_prod_axis0() {
        let input = make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_prod(&input, &[0], false).unwrap();
        let data = out.data().typed_data::<f32>();
        assert!((data[0] - 4.0).abs() < 1e-6); // 1*4
        assert!((data[1] - 10.0).abs() < 1e-5); // 2*5
        assert!((data[2] - 18.0).abs() < 1e-5); // 3*6
    }
}

use std::simd::prelude::*;
use std::simd::StdFloat;

const SIMD_WIDTH: usize = 4;

/// Fast exp(x) approximation for f64 SIMD.
#[inline(always)]
fn fast_exp(x: Simd<f64, SIMD_WIDTH>) -> Simd<f64, SIMD_WIDTH> {
    let x = x.simd_clamp(Simd::splat(-709.0), Simd::splat(709.0));

    let log2e = Simd::splat(1.4426950408889634f64);
    let ln2 = Simd::splat(0.6931471805599453f64);

    let z = x * log2e;
    let floor_z = z.floor();
    let frac = x - floor_z * ln2;

    let c1 = Simd::splat(1.0f64);
    let c2 = Simd::splat(0.5f64);
    let c3 = Simd::splat(1.0 / 6.0f64);
    let c4 = Simd::splat(1.0 / 24.0f64);
    let c5 = Simd::splat(1.0 / 120.0f64);
    let c6 = Simd::splat(1.0 / 720.0f64);
    let poly = c1 + frac * (c1 + frac * (c2 + frac * (c3 + frac * (c4 + frac * (c5 + frac * c6)))));

    let bias = Simd::<i64, SIMD_WIDTH>::splat(1023);
    let floor_i: Simd<i64, SIMD_WIDTH> = unsafe { floor_z.to_int_unchecked() };
    let pow2 =
        Simd::<f64, SIMD_WIDTH>::from_bits((bias + floor_i).cast::<u64>() << Simd::splat(52));

    poly * pow2
}

pub fn softmax(input: &[f64]) -> Vec<f64> {
    let len = input.len();

    // 1. Find max
    let chunks = len / SIMD_WIDTH;
    let mut max_v = Simd::<f64, SIMD_WIDTH>::splat(f64::NEG_INFINITY);

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let v = Simd::<f64, SIMD_WIDTH>::from_slice(&input[offset..]);
        max_v = max_v.simd_max(v);
    }

    let mut max_val = max_v.reduce_max();
    for i in (chunks * SIMD_WIDTH)..len {
        max_val = max_val.max(input[i]);
    }

    // 2. Compute exp(x - max) and sum
    let max_splat = Simd::<f64, SIMD_WIDTH>::splat(max_val);
    let mut output = Vec::with_capacity(len);
    let mut sum_v = Simd::<f64, SIMD_WIDTH>::splat(0.0);

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let v = Simd::<f64, SIMD_WIDTH>::from_slice(&input[offset..]);
        let exp_v = fast_exp(v - max_splat);
        sum_v += exp_v;
        output.extend_from_slice(exp_v.as_array());
    }

    let mut sum = sum_v.reduce_sum();
    for i in (chunks * SIMD_WIDTH)..len {
        let e = (input[i] - max_val).exp();
        sum += e;
        output.push(e);
    }

    // 3. Normalize
    let inv_sum = Simd::<f64, SIMD_WIDTH>::splat(1.0 / sum);

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let v = Simd::<f64, SIMD_WIDTH>::from_slice(&output[offset..]);
        let result = v * inv_sum;
        output[offset..offset + SIMD_WIDTH].copy_from_slice(result.as_array());
    }

    let inv_sum_scalar = 1.0 / sum;
    for i in (chunks * SIMD_WIDTH)..len {
        output[i] *= inv_sum_scalar;
    }

    output
}

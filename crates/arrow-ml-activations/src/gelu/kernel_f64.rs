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

/// Fast tanh via (exp(2x)-1)/(exp(2x)+1)
#[inline(always)]
fn fast_tanh(x: Simd<f64, SIMD_WIDTH>) -> Simd<f64, SIMD_WIDTH> {
    let two = Simd::splat(2.0f64);
    let one = Simd::splat(1.0f64);
    let exp2x = fast_exp(two * x);
    (exp2x - one) / (exp2x + one)
}

/// gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
#[inline]
pub fn gelu(input: &[f64]) -> Vec<f64> {
    let len = input.len();
    let mut output = Vec::with_capacity(len);

    let half = Simd::<f64, SIMD_WIDTH>::splat(0.5);
    let one = Simd::<f64, SIMD_WIDTH>::splat(1.0);
    let sqrt_2_over_pi = Simd::<f64, SIMD_WIDTH>::splat(0.7978845608028654f64);
    let coeff = Simd::<f64, SIMD_WIDTH>::splat(0.044715f64);

    let chunks = len / SIMD_WIDTH;

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let x = Simd::<f64, SIMD_WIDTH>::from_slice(&input[offset..]);
        let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        let result = half * x * (one + fast_tanh(inner));
        output.extend_from_slice(result.as_array());
    }

    for i in (chunks * SIMD_WIDTH)..len {
        let x = input[i];
        let inner = 0.7978845608028654f64 * (x + 0.044715f64 * x * x * x);
        output.push(0.5 * x * (1.0 + inner.tanh()));
    }

    output
}

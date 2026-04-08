use std::simd::prelude::*;
use std::simd::StdFloat;

const SIMD_WIDTH: usize = 8;

/// Fast exp(x) approximation for f32 SIMD.
#[inline(always)]
fn fast_exp(x: Simd<f32, SIMD_WIDTH>) -> Simd<f32, SIMD_WIDTH> {
    let x = x.simd_clamp(Simd::splat(-88.0), Simd::splat(88.0));

    let log2e = Simd::splat(1.4426950408889634f32);
    let ln2 = Simd::splat(0.6931471805599453f32);

    let z = x * log2e;
    let floor_z = z.floor();
    let frac = x - floor_z * ln2;

    let c1 = Simd::splat(1.0f32);
    let c2 = Simd::splat(0.5f32);
    let c3 = Simd::splat(1.0 / 6.0f32);
    let c4 = Simd::splat(1.0 / 24.0f32);
    let poly = c1 + frac * (c1 + frac * (c2 + frac * (c3 + frac * c4)));

    let bias = Simd::<i32, SIMD_WIDTH>::splat(127);
    let floor_i: Simd<i32, SIMD_WIDTH> = unsafe { floor_z.to_int_unchecked() };
    let pow2 =
        Simd::<f32, SIMD_WIDTH>::from_bits((bias + floor_i).cast::<u32>() << Simd::splat(23));

    poly * pow2
}

/// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
#[inline]
pub fn tanh(input: &[f32]) -> Vec<f32> {
    let len = input.len();
    let mut output = Vec::with_capacity(len);
    let two = Simd::<f32, SIMD_WIDTH>::splat(2.0);
    let one = Simd::<f32, SIMD_WIDTH>::splat(1.0);

    let chunks = len / SIMD_WIDTH;

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let x = Simd::<f32, SIMD_WIDTH>::from_slice(&input[offset..]);
        let exp2x = fast_exp(two * x);
        let result = (exp2x - one) / (exp2x + one);
        output.extend_from_slice(result.as_array());
    }

    for i in (chunks * SIMD_WIDTH)..len {
        output.push(input[i].tanh());
    }

    output
}

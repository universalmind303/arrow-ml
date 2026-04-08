use std::simd::prelude::*;
use std::simd::Select;

const SIMD_WIDTH: usize = 8;

#[inline]
pub fn relu(input: &[f32]) -> Vec<f32> {
    let len = input.len();
    let mut output = Vec::with_capacity(len);
    let zero = Simd::<f32, SIMD_WIDTH>::splat(0.0);

    let chunks = len / SIMD_WIDTH;

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let v = Simd::<f32, SIMD_WIDTH>::from_slice(&input[offset..]);
        let result = v.simd_max(zero);
        output.extend_from_slice(result.as_array());
    }

    for i in (chunks * SIMD_WIDTH)..len {
        output.push(if input[i] > 0.0 { input[i] } else { 0.0 });
    }

    debug_assert_eq!(output.len(), len);
    output
}

#[inline]
pub fn leaky_relu(input: &[f32], alpha: f32) -> Vec<f32> {
    let len = input.len();
    let mut output = Vec::with_capacity(len);
    let zero = Simd::<f32, SIMD_WIDTH>::splat(0.0);
    let alpha_v = Simd::<f32, SIMD_WIDTH>::splat(alpha);

    let chunks = len / SIMD_WIDTH;

    for i in 0..chunks {
        let offset = i * SIMD_WIDTH;
        let v = Simd::<f32, SIMD_WIDTH>::from_slice(&input[offset..]);
        let mask = v.simd_gt(zero);
        let negative = v * alpha_v;
        let result = mask.select(v, negative);
        output.extend_from_slice(result.as_array());
    }

    for i in (chunks * SIMD_WIDTH)..len {
        let x = input[i];
        output.push(if x > 0.0 { x } else { alpha * x });
    }

    output
}

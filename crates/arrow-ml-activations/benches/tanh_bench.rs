use arrow::array::Float32Array;
use arrow_ml_activations::tanh::tanh;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn naive_tanh_f32(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| x.tanh()).collect()
}

fn bench_tanh(c: &mut Criterion) {
    let mut group = c.benchmark_group("tanh_f32");
    for &size in &[256, 1024, 4096, 16384, 65536, 262144] {
        let data: Vec<f32> = (0..size)
            .map(|i| (i as f32 - size as f32 / 2.0) * 0.01)
            .collect();
        let array = Float32Array::from(data.clone());

        group.bench_with_input(BenchmarkId::new("arrow_tanh", size), &size, |bench, _| {
            bench.iter(|| tanh(&array))
        });

        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| naive_tanh_f32(&data))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_tanh);
criterion_main!(benches);

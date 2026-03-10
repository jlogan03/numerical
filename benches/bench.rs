use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("title", |b| {
        let input = black_box([1.0; 20]);
        b.iter(|| {
            // black_box(
            //     butter2_bank
            //         .iter_mut()
            //         .zip(input.iter())
            //         .for_each(|(f, v)| {
            //             f.update(*v);
            //         }),
            // )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

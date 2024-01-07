#![allow(non_snake_case)]
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use itertools::Itertools;
use ndarray::{array, Array1, Array2};
use ndarray_rand::RandomExt;
use rand::{distributions::Uniform, Rng};
use twirs::{QuadAsterism, Twirs};

fn twirl() -> (Twirs<f64, QuadAsterism>, Array2<f64>, Array2<f64>) {
    let mut rng = rand::thread_rng();

    let shape = (15, 2);
    let angle: f64 = rng.gen();
    let rot = array![[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]];
    let offset = Array1::random(2, Uniform::new(0., 1.));

    let pixels = Array2::random(shape, Uniform::new(0., 1.));
    let radecs_vec = pixels
        .rows()
        .into_iter()
        .flat_map(|r| rot.dot(&r) + offset.view())
        .collect_vec();
    let radecs = Array2::from_shape_vec(shape, radecs_vec).unwrap();

    (Twirs::quads(pixels.clone(), radecs.clone()), pixels, radecs)
}

fn wcs_benchmark(c: &mut Criterion) {
    let mut wcs = c.benchmark_group("wcs");
    wcs.sample_size(10);

    let (twirl, _, _) = twirl();
    wcs.bench_function("wcs blocking", |b| {
        b.iter_batched(|| twirl.clone(), |t| t.compute_wcs(), BatchSize::SmallInput)
    });

    wcs.bench_function("wcs parallel", |b| {
        b.iter_batched(
            || twirl.clone(),
            |t| t.compute_wcs_par(),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, wcs_benchmark);
criterion_main!(benches);

extern crate nalgebra as na;
extern crate rudolf;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use na::{MatrixMN, Vector3, U3};
use rudolf::core::{Filter, Model, Noise, State};
use rudolf::filters::kalman::KalmanFilter;

pub fn bench_kalman_predict_f32_3d(c: &mut Criterion) {
    let mut filter = KalmanFilter::<f32, U3> {
        state: State::<f32, U3> {
            mean: Vector3::from_element(2.0),
            cov: MatrixMN::<f32, U3, U3>::identity(),
        },
        model: Model::<f32, U3> {
            obs: MatrixMN::<f32, U3, U3>::identity(),
            ctrl: MatrixMN::<f32, U3, U3>::identity(),
        },
        noise: Noise::<f32, U3> {
            obs: MatrixMN::<f32, U3, U3>::identity(),
            ctrl: MatrixMN::<f32, U3, U3>::identity(),
        },
    };

    let new_ctrl = Vector3::from_element(1.0);

    c.bench_function("filters::KalmanFilter::predict<f32, U3>", |b| {
        b.iter(|| filter.predict(black_box(&new_ctrl)))
    });
}

pub fn bench_kalman_update_f32_3d(c: &mut Criterion) {
    let mut filter = KalmanFilter::<f32, U3> {
        state: State::<f32, U3> {
            mean: Vector3::from_element(2.0),
            cov: MatrixMN::<f32, U3, U3>::identity(),
        },
        model: Model::<f32, U3> {
            obs: MatrixMN::<f32, U3, U3>::identity(),
            ctrl: MatrixMN::<f32, U3, U3>::identity(),
        },
        noise: Noise::<f32, U3> {
            obs: MatrixMN::<f32, U3, U3>::identity(),
            ctrl: MatrixMN::<f32, U3, U3>::identity(),
        },
    };

    let new_obs = Vector3::from_element(1.0);

    c.bench_function("filters::KalmanFilter::update<f32, U3>", |b| {
        b.iter(|| filter.update(black_box(&new_obs)))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_kalman_predict_f32_3d, bench_kalman_update_f32_3d
}
criterion_main!(benches);

extern crate nalgebra as na;
extern crate rudolf;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use na::{Matrix3, Vector3};
use rudolf::core::{Filter, Model, Noise, State};
use rudolf::filters::kalman::KalmanFilter;

pub fn bench_kalman_predict_f32_s3o3(c: &mut Criterion) {
    let mut filter = KalmanFilter::<f32, 3, 3> {
        state: State::<f32, 3> {
            mean: Vector3::from_element(2.0),
            cov: Matrix3::zeros(),
        },
        model: Model::<f32, 3, 3> {
            state: Matrix3::identity(),
            ctrl: Matrix3::identity(),
            obs: Matrix3::identity(),
        },
        noise: Noise::<f32, 3, 3> {
            ctrl: Matrix3::identity(),
            obs: Matrix3::identity(),
        },
    };

    let new_ctrl = Vector3::from_element(1.0);

    c.bench_function("filters::KalmanFilter<f32, 3, 3>::predict", |b| {
        b.iter(|| filter.predict(black_box(&new_ctrl)))
    });
}

pub fn bench_kalman_update_f32_s3o3(c: &mut Criterion) {
    let mut filter = KalmanFilter::<f32, 3, 3> {
        state: State::<f32, 3> {
            mean: Vector3::from_element(2.0),
            cov: Matrix3::identity(),
        },
        model: Model::<f32, 3, 3> {
            state: Matrix3::identity(),
            ctrl: Matrix3::identity(),
            obs: Matrix3::identity(),
        },
        noise: Noise::<f32, 3, 3> {
            ctrl: Matrix3::identity(),
            obs: Matrix3::identity(),
        },
    };

    let new_obs = Vector3::from_element(1.0);

    c.bench_function("filters::KalmanFilter<f32, 3, 3>::update", |b| {
        b.iter(|| filter.update(black_box(&new_obs)))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_kalman_predict_f32_s3o3, bench_kalman_update_f32_s3o3
}
criterion_main!(benches);

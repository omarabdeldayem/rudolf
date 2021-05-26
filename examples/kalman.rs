extern crate nalgebra as na;
extern crate rudolf;

use na::{Matrix3, Vector3};
// use rudolf::{State, Models, Noise, KalmanFilter};
use rudolf::core::{Filter, Model, Noise, State};
use rudolf::filters::kalman::KalmanFilter;

fn main() {
    let mut filter = KalmanFilter::<f32, 3> {
        state: State::<f32, 3> {
            mean: Vector3::new(1.0, 1.0, 1.0),
            cov: Matrix3::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        },
        model: Model::<f32, 3> {
            obs: Matrix3::new(1.0, 0.8, 1.5, 2.0, 1.0, 4.0, 1.2, 1.5, 1.0),
            ctrl: Matrix3::new(1.0, 0.5, 0.0, 2.0, 0.0, 4.0, 0.0, 3.0, 1.0),
        },
        noise: Noise::<f32, 3> {
            obs: Matrix3::new(0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01),
            ctrl: Matrix3::new(0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02),
        },
    };

    let new_obs = Vector3::new(2.0, 1.2, 3.4);
    let new_ctrl = Vector3::new(2.0, 1.4, 2.3);

    filter.predict(&new_ctrl);
    filter.update(&new_obs);

    println!("{:?}", filter.state.mean);
}

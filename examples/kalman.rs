extern crate rudolf;
extern crate nalgebra as na;

use na::{Vector3, MatrixN, U3};
// use rudolf::{State, Models, Noise, KalmanFilter};
use rudolf::filters::kalman::KalmanFilter;
use rudolf::core::{State, Models, Noise, Filter};

fn main() {
    let mut filter = KalmanFilter::<f32, U3> {
        state: State::<f32, U3> {
            mean: Vector3::new(1.0, 1.0, 1.0),
            cov: MatrixN::from_element_generic(U3, U3, 0.5),
        },
        models: Models::<f32, U3> {
            obs: MatrixN::from_row_slice_generic(
                U3,
                U3,
                &[1.0, 0.8, 1.5, 2.0, 1.0, 4.0, 1.2, 1.5, 1.0],
            ),
            ctrl: MatrixN::from_row_slice_generic(
                U3,
                U3,
                &[1.0, 0.5, 0.0, 2.0, 0.0, 4.0, 0.0, 3.0, 1.0],
            ),
        },
        noise: Noise::<f32, U3> {
            obs: MatrixN::from_element_generic(U3, U3, 0.01),
            ctrl: MatrixN::from_element_generic(U3, U3, 0.02),
        },
    };

    let new_obs = Vector3::new(2.0, 1.2, 3.4);
    let new_ctrl = Vector3::new(2.0, 1.4, 2.3);

    filter.predict(&new_ctrl);
    filter.update(&new_obs);

    println!("{:?}", filter.state.mean);

}

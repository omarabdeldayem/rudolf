use nalgebra::{Matrix2, Matrix2x4, Matrix4, Vector2, Vector4};
use rudolf::base::{Filter, Model, Noise, State};
use rudolf::filters::kalman::KalmanFilter;

#[path = "utils/utils.rs"]
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut state = State::<f32, 4> {
        mean: Vector4::new(0.0, 0.0, 0.0, 0.0),
        cov: Matrix4::zeros(),
    };
    let mut filter = KalmanFilter::<f32, 4, 2> {
        model: Model::<f32, 4, 2> {
            state: Matrix4::new(
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ),
            ctrl: Matrix4::identity(),
            obs: Matrix2x4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        },
        noise: Noise::<f32, 4, 2> {
            ctrl: Matrix4::identity(),
            obs: Matrix2::identity() * 10.0,
        },
    };

    let raw = utils::gen_noisey_linear();
    let filtered: Vec<(f64, f64)> = raw
        .iter()
        .map(|(x, y)| {
            state = filter.predict(&state, &Vector4::zeros());
            state = filter.update(&state, &Vector2::new(*x as f32, *y as f32));
            (state.mean.x as f64, state.mean.y as f64)
        })
        .collect();

    utils::plot_comparison(
        raw,
        filtered,
        "Linear Model",
        "KalmanFilter",
        "./examples/kalman_f32s3o3.png",
    )
}

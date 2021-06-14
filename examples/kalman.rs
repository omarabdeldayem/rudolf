extern crate nalgebra as na;
extern crate rudolf;

use na::{Matrix2, Matrix2x4, Matrix4, Vector2, Vector4};
use rudolf::core::{Filter, Model, Noise, State};
use rudolf::filters::kalman::KalmanFilter;

use plotters::prelude::*;

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_xorshift::XorShiftRng;

const OUT_FILE_NAME: &'static str = "./examples/kalman_f32s3o3.png";

fn gen_noisey_linear() -> Vec<(f64, f64)> {
    let norm_dist = Normal::new(0.0, 1.0).unwrap();
    let mut x_rand = XorShiftRng::from_seed(*b"1234567887654321");
    let x_iter = norm_dist.sample_iter(&mut x_rand);
    x_iter
        .take(1000)
        .zip(0..50)
        .map(|(y, x)| (x as f64, x as f64 + (y as f64 * 4.0)))
        .collect()
}

fn plot(raw: Vec<(f64, f64)>, filtered: Vec<(f64, f64)>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(OUT_FILE_NAME, (1024, 768)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Linear Model", ("sans-serif", 20))
        .margin(10)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(0f64..50f64, 0f64..50f64)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(
            raw.iter()
                .map(|(x, y)| Circle::new((*x, *y), 3, BLUE.filled())),
        )?
        .label("Measurement");

    chart
        .draw_series(LineSeries::new(filtered, &BLUE))?
        .label("Kalman Filter")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(WHITE.filled())
        .draw()?;

    root.present().expect("Unable to write result to file");
    println!("Result has been saved to {}", OUT_FILE_NAME);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut filter = KalmanFilter::<f32, 4, 2> {
        state: State::<f32, 4> {
            mean: Vector4::new(0.0, 0.0, 0.0, 0.0),
            cov: Matrix4::zeros(),
        },
        model: Model::<f32, 4, 2> {
            state: Matrix4::new(
                1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ),
            ctrl: Matrix4::identity(),
            obs: Matrix2x4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        },
        noise: Noise::<f32, 4, 2> {
            ctrl: Matrix4::identity() * 100.0,
            obs: Matrix2::identity() * 100.0,
        },
    };

    let raw = gen_noisey_linear();
    let filtered: Vec<(f64, f64)> = raw
        .iter()
        .map(|(x, y)| {
            filter.predict(&Vector4::zeros());
            filter.update(&Vector2::new(*x as f32, *y as f32));
            (filter.state.mean.x as f64, filter.state.mean.y as f64)
        })
        .collect();

    plot(raw, filtered)
}

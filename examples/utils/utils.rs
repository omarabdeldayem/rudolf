use plotters::prelude::*;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_xorshift::XorShiftRng;

pub fn gen_noisey_linear() -> Vec<(f64, f64)> {
    let norm_dist = Normal::new(0.0, 1.0).unwrap();
    let mut x_rand = XorShiftRng::from_seed(*b"1234567887654321");
    let x_iter = norm_dist.sample_iter(&mut x_rand);
    x_iter
        .take(1000)
        .zip(0..50)
        .map(|(y, x)| (x as f64, x as f64 + (y as f64 * 4.0)))
        .collect()
}

pub fn plot_comparison(
    raw: Vec<(f64, f64)>,
    filtered: Vec<(f64, f64)>,
    title: &str,
    filtered_label: &str,
    png_out: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(png_out, (1024, 768)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 20))
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
        .label("Measurement")
        .legend(|(x, y)| Circle::new((x, y), 3, BLUE.filled()));

    chart
        .draw_series(LineSeries::new(filtered, &BLUE))?
        .label(filtered_label)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(WHITE.filled())
        .draw()?;

    root.present().expect("Unable to write result to file");
    println!("Result has been saved to {}", png_out);

    Ok(())
}

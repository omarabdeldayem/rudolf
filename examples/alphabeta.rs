use nalgebra::{Matrix1, Vector1};
use rudolf::base::{Filter, Model, State};
use rudolf::filters::alphabeta::AlphaBetaFilter;

#[path = "utils/utils.rs"]
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut state = State::<f64, 1> {
        mean: Vector1::new(0.0),
        cov: Vector1::zeros(),
    };
    let model = Model::<f64, 1, 1> {
        state: Matrix1::new(1.0),
        ctrl: Matrix1::new(0.0),
        obs: Matrix1::new(0.0),
    };
    let mut filter = AlphaBetaFilter::<f64, 1, 1>::new(0.001, 0.00001, 1.0, model);
    let raw = utils::gen_noisey_linear();
    let filtered: Vec<(f64, f64)> = raw
        .iter()
        .map(|(x, y)| {
            state = filter.predict(&state, &Vector1::zeros());
            state = filter.update(&state, &Vector1::new(*y as f64));
            (*x as f64, state.mean.x as f64)
        })
        .collect();

    utils::plot_comparison(
        raw,
        filtered,
        "Linear Model",
        "AlphaBetaFilter",
        "./examples/alphabeta.png",
    )
}

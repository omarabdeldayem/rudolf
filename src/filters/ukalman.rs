use crate::core::{Filter, Noise, NonLinearModel, State, SigmaPointGenerator};
use crate::generators::sigma_points::{MerweGenerator};

use na::{RealField, SVector};

pub struct UKalmanFilter<T, const S: usize, const O: usize, const N: usize>
where
    T: RealField,
{
    pub state: State<T, S>,
    pub model: NonLinearModel<T, S, O>,
    pub noise: Noise<T, S, O>,
    pub generator: MerweGenerator<T, S, N>,
    pub dt: T,
}

impl<T, const S: usize, const O: usize, const N: usize> Filter<T, S, O> for UKalmanFilter<T, S, O, N>
where
    T: RealField + num::NumCast,
{
    fn predict(&mut self, _ctrl: &SVector<T, S>) {
        let mut sigma_points = self.generator.generate_sigmas(&self.state.mean, &self.state.cov);
        // sigma_points.sigmas.row_iter_mut().map(|row| row = ((self.model.state)(row.transpose(), self.dt)).transpose());
        for (i, row) in sigma_points.sigmas.row_iter_mut().enumerate() {
            row = ((self.model.state)(row.transpose(), self.dt)).transpose();
        }
    }

    fn update(&mut self, _obs: &SVector<T, O>) {}
}

use crate::core::{Filter, Model, Noise, State};

use na::{RealField, SMatrix, SVector};

#[derive(Debug)]
pub struct KalmanFilter<T, const D: usize>
where
    T: RealField,
{
    pub state: State<T, D>,
    pub model: Model<T, D>,
    pub noise: Noise<T, D>,
}

impl<T, const D: usize> Filter<T, D> for KalmanFilter<T, D>
where
    T: RealField,
{
    fn predict(&mut self, ctrl: &SVector<T, D>) {
        self.state.mean = (&self.model.obs * &self.state.mean) + (&self.model.ctrl * ctrl);
        self.state.cov =
            &self.model.obs * &self.state.cov * &self.model.obs.transpose() + &self.noise.ctrl;
    }

    fn update(&mut self, obs: &SVector<T, D>) {
        let gain = &self.state.cov
            * &self.model.obs
            * (&self.model.obs * &self.state.cov * &self.model.obs.transpose() + &self.noise.obs)
                .try_inverse()
                .unwrap();
        self.state.mean = &self.state.mean + (&gain * (obs - (&self.model.obs * &self.state.mean)));
        self.state.cov = (SMatrix::<T, D, D>::identity() - &gain * &self.model.obs) * &self.state.cov;
    }
}

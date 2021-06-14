use crate::core::{Filter, Model, Noise, State};

use na::{RealField, SMatrix, SVector};

#[derive(Debug)]
pub struct KalmanFilter<T, const S: usize, const O: usize>
where
    T: RealField,
{
    pub state: State<T, S>,
    pub model: Model<T, S, O>,
    pub noise: Noise<T, S, O>,
}

impl<T, const S: usize, const O: usize> Filter<T, S, O> for KalmanFilter<T, S, O>
where
    T: RealField,
{
    fn predict(&mut self, ctrl: &SVector<T, S>) {
        self.state.mean = (&self.model.state * &self.state.mean) + (&self.model.ctrl * ctrl);
        self.state.cov =
            &self.model.state * &self.state.cov * &self.model.state.transpose() + &self.noise.ctrl;
    }

    fn update(&mut self, obs: &SVector<T, O>) {
        let gain = &self.state.cov
            * &self.model.obs.transpose()
            * ((&self.model.obs * &self.state.cov * &self.model.obs.transpose() + &self.noise.obs)
                .try_inverse()
                .unwrap());
        self.state.mean = &self.state.mean + (&gain * (obs - (&self.model.obs * &self.state.mean)));
        self.state.cov =
            (SMatrix::<T, S, S>::identity() - &gain * &self.model.obs) * &self.state.cov;
    }
}

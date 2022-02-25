use crate::base::{Filter, Model, Noise, State};
use nalgebra::{RealField, SMatrix, SVector};

#[derive(Debug)]
pub struct KalmanFilter<T, const S: usize, const O: usize>
where
    T: RealField,
{
    // pub state: State<T, S>,
    pub model: Model<T, S, O>,
    pub noise: Noise<T, S, O>,
}

impl<T, const S: usize, const O: usize> Filter<T, S, O> for KalmanFilter<T, S, O>
where
    T: RealField,
{
    fn predict(&self, state: &State<T, S>, ctrl: &SVector<T, S>) -> State<T, S> {
        let mean = (&self.model.state * state.mean) + (&self.model.ctrl * ctrl);
        let cov = &self.model.state * state.cov * &self.model.state.transpose() + &self.noise.ctrl;
        State { mean, cov }
    }

    fn update(&self, state: &State<T, S>, obs: &SVector<T, O>) -> State<T, S> {
        let gain = state.cov
            * &self.model.obs.transpose()
            * ((&self.model.obs * state.cov * &self.model.obs.transpose() + &self.noise.obs)
                .try_inverse()
                .unwrap());
        let mean = state.mean + (&gain * (obs - (&self.model.obs * state.mean)));
        let cov = (SMatrix::<T, S, S>::identity() - &gain * &self.model.obs) * state.cov;
        State { mean, cov }
    }
}

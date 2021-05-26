use crate::core::{Filter, Noise, NonlinModel, State};

use na::{RealField, SVector};

pub struct UKalmanFilter<T, const D: usize>
where
    T: RealField,
{
    pub state: State<T, D>,
    pub model: NonlinModel<T, D>,
    pub noise: Noise<T, D>,
    pub dt: T,
}

impl<T, const D: usize> Filter<T, D> for UKalmanFilter<T, D>
where
    T: RealField,
{
    fn predict(&mut self, _ctrl: &SVector<T, D>) {}

    fn update(&mut self, _obs: &SVector<T, D>) {}
}

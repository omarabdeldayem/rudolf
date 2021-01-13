use crate::core::{Filter, Noise, NonlinModel, State};

use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, MatrixN, RealField, VectorN};

pub struct UKalmanFilter<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    pub state: State<T, D>,
    pub model: NonlinModel<T, D>,
    pub noise: Noise<T, D>,
    pub dt: T,
}

impl<T, D> Filter<T, D> for UKalmanFilter<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    fn predict(&mut self, ctrl: &VectorN<T, D>) {}

    fn update(&mut self, obs: &VectorN<T, D>) {}
}

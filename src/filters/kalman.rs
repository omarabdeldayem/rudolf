use crate::core::{Filter, Models, Noise, State};

use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, MatrixN, RealField, VectorN};

#[derive(Debug)]
pub struct KalmanFilter<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    pub state: State<T, D>,
    pub models: Models<T, D>,
    pub noise: Noise<T, D>,
}

impl<T, D> Filter<T, D> for KalmanFilter<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{

    fn predict(&mut self, ctrl: &VectorN<T, D>) {
        self.state.mean = (&self.models.obs * &self.state.mean) + (&self.models.ctrl * ctrl);
        self.state.cov =
            &self.models.obs * &self.state.cov * &self.models.obs.transpose() + &self.noise.ctrl;
    }

    fn update(&mut self, obs: &VectorN<T, D>) {
        let gain = &self.state.cov
            * &self.models.obs
            * (&self.models.obs * &self.state.cov * &self.models.obs.transpose() + &self.noise.obs)
                .try_inverse()
                .unwrap();
        self.state.mean =
            &self.state.mean + (&gain * (obs - (&self.models.obs * &self.state.mean)));
        self.state.cov = (MatrixN::<T, D>::identity() - &gain * &self.models.obs) * &self.state.cov;
    }

}

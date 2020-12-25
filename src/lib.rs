extern crate nalgebra as na;

use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, MatrixN, RealField, VectorN};

pub struct State<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    pub mean: VectorN<T, D>,
    pub cov: MatrixN<T, D>,
}

pub struct Noise<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    pub obs: MatrixN<T, D>,
    pub ctrl: MatrixN<T, D>,
}

pub struct Models<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    pub obs: MatrixN<T, D>,
    pub ctrl: MatrixN<T, D>,
}

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

impl<T, D> KalmanFilter<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    pub fn update(&mut self, obs: &VectorN<T, D>) { let inovation = &self.state.cov
            * &self.models.obs
            * (&self.models.obs * &self.state.cov * &self.models.obs.transpose()
                + &self.noise.obs)
                .try_inverse()
                .unwrap();
        self.state.mean = &self.state.mean
            + (&inovation * (obs - (&self.models.obs * &self.state.mean)));
        self.state.cov =
            (MatrixN::<T, D>::identity() - &inovation * &self.models.obs) * &self.state.cov;
    }

    pub fn predict(&mut self, ctrl: &VectorN<T, D>) {
        self.state.mean =
            (&self.models.obs * &self.state.mean) + (&self.models.ctrl * ctrl);
        self.state.cov =
            &self.models.obs * &self.state.cov * &self.models.obs.transpose()
                + &self.noise.ctrl;
    }
}


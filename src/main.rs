extern crate nalgebra as na;

use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, MatrixN, RealField, VectorN};

struct State<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    mean: VectorN<T, D>,
    cov: MatrixN<T, D>,
}

struct Noise<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    observation: MatrixN<T, D>,
    control: MatrixN<T, D>,
}

struct Models<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    observation: MatrixN<T, D>,
    control: MatrixN<T, D>,
}

struct KalmanFilter<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    state: State<T, D>,
    models: Models<T, D>,
    noise: Noise<T, D>,
}

impl<T, D> KalmanFilter<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    fn update(&mut self, observation: &VectorN<T, D>) {
        let inovation = &self.state.cov
            * &self.models.observation
            * (&self.models.observation * &self.state.cov * &self.models.observation.transpose()
                + &self.noise.observation)
                .try_inverse()
                .unwrap();
        self.state.mean = &self.state.mean
            + (&inovation * (observation - (&self.models.observation * &self.state.mean)));
        self.state.cov =
            (MatrixN::<T, D>::identity() - &inovation * &self.models.observation) * &self.state.cov;
    }

    fn predict(&mut self, control: &VectorN<T, D>) {
        self.state.mean =
            (&self.models.observation * &self.state.mean) + (&self.models.control * control);
        self.state.cov =
            &self.models.observation * &self.state.cov * &self.models.observation.transpose()
                + &self.noise.control;
    }
}

fn main() {}

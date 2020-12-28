extern crate nalgebra as na;

use na::linalg::Cholesky;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, MatrixN, RealField, VectorN};

pub trait SigmaPointGenerator<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    fn generate_sigmas(&self, mean: &VectorN<T, D>, cov: &MatrixN<T, D>) -> VectorN<T, D>;
    fn generate_weights(&self);
}

pub struct SimplexGenerator<T>
where
    T: RealField,
{
    pub kappa: T,
}

impl<T, D> SigmaPointGenerator<T, D> for SimplexGenerator<T>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    fn generate_sigmas(&self, mean: &VectorN<T, D>, cov: &MatrixN<T, D>) -> VectorN<T, D> {
        let sqrt_cov = Cholesky::<T, D>::new(*cov);
    }

    fn generate_weights(&self) {

    }
}

pub struct JulierGenerator<T>
where
    T: RealField,
{
    pub scale_factor: T,
}

impl<T, D> SigmaPointGenerator<T, D> for JulierGenerator<T>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    fn generate_sigmas(&self, mean: &VectorN<T, D>, cov: &MatrixN<T, D>) -> VectorN<T, D> {
        let sqrt_cov = Cholesky::<T, D>::new(*cov);
    }

    fn generate_weights(&self) {

    }
}

pub struct MerweGenerator<T>
where
    T: RealField,
{
    pub alpha: T,
}

impl<T, D> SigmaPointGenerator<T, D> for MerweGenerator<T>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    fn generate_sigmas(&self, mean: &VectorN<T, D>, cov: &MatrixN<T, D>) -> VectorN<T, D> {
        let sqrt_cov = Cholesky::<T, D>::new(*cov);
    }

    fn generate_weights(&self) {

    }
}

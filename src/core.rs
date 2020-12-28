extern crate nalgebra as na;

use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, MatrixN, RealField, VectorN};

#[derive(Debug)]
pub struct State<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    pub mean: VectorN<T, D>,
    pub cov: MatrixN<T, D>,
}

#[derive(Debug)]
pub struct Noise<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    pub obs: MatrixN<T, D>,
    pub ctrl: MatrixN<T, D>,
}

#[derive(Debug)]
pub struct Models<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    pub obs: MatrixN<T, D>,
    pub ctrl: MatrixN<T, D>,
}

pub trait Filter<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D>,
{
    fn predict(&mut self, obs: &VectorN<T, D>);
    fn update(&mut self, ctrl: &VectorN<T, D>);
}

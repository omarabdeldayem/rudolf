extern crate nalgebra as na;

use na::{RealField, SMatrix, SVector};

#[derive(Debug)]
pub struct State<T, const D: usize>
where
    T: RealField,
{
    pub mean: SVector<T, D>,
    pub cov: SMatrix<T, D, D>,
}

#[derive(Debug)]
pub struct Noise<T, const D: usize>
where
    T: RealField,
{
    pub obs: SMatrix<T, D, D>,
    pub ctrl: SMatrix<T, D, D>,
}

#[derive(Debug)]
pub struct Model<T, const D: usize>
where
    T: RealField,
{
    pub obs: SMatrix<T, D, D>,
    pub ctrl: SMatrix<T, D, D>,
}

#[derive(Debug)]
pub struct NonlinModel<T, const D: usize>
where
    T: RealField,
{
    pub obs: fn(SVector<T, D>, T) -> SVector<T, D>,
    pub ctrl: fn(SVector<T, D>) -> SMatrix<T, D, D>,
}

pub trait Filter<T, const D: usize>
where
    T: RealField,
{
    fn predict(&mut self, obs: &SVector<T, D>);
    fn update(&mut self, ctrl: &SVector<T, D>);
}

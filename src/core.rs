extern crate nalgebra as na;

use na::{RealField, SMatrix, SVector};

#[derive(Debug)]
pub struct State<T, const S: usize>
where
    T: RealField,
{
    pub mean: SVector<T, S>,
    pub cov: SMatrix<T, S, S>,
}

#[derive(Debug)]
pub struct Noise<T, const S: usize, const M: usize>
where
    T: RealField,
{
    pub ctrl: SMatrix<T, S, S>,
    pub obs: SMatrix<T, M, M>,
}

#[derive(Debug)]
pub struct Model<T, const S: usize, const M: usize>
where
    T: RealField,
{
    pub state: SMatrix<T, S, S>,
    pub ctrl: SMatrix<T, S, S>,
    pub obs: SMatrix<T, M, S>,
}

#[derive(Debug)]
pub struct NonlinModel<T, const D: usize>
where
    T: RealField,
{
    pub ctrl: fn(SVector<T, D>) -> SMatrix<T, D, D>,
    pub obs: fn(SVector<T, D>, T) -> SVector<T, D>,
}

pub trait Filter<T, const S: usize, const M: usize>
where
    T: RealField,
{
    fn predict(&mut self, ctrl: &SVector<T, S>);
    fn update(&mut self, obs: &SVector<T, M>);
}

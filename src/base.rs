use nalgebra::{RealField, SMatrix, SVector};

#[derive(Debug)]
pub struct State<T, const S: usize>
where
    T: RealField,
{
    pub mean: SVector<T, S>,
    pub cov: SMatrix<T, S, S>,
}

#[derive(Debug)]
pub struct Noise<T, const S: usize, const O: usize>
where
    T: RealField,
{
    pub ctrl: SMatrix<T, S, S>,
    pub obs: SMatrix<T, O, O>,
}

#[derive(Debug)]
pub struct Model<T, const S: usize, const O: usize>
where
    T: RealField,
{
    pub state: SMatrix<T, S, S>,
    pub ctrl: SMatrix<T, S, S>,
    pub obs: SMatrix<T, O, S>,
}

#[derive(Debug)]
pub struct NonLinearModel<T, const S: usize, const O: usize>
where
    T: RealField,
{
    pub state: fn(SVector<T, S>, T) -> SVector<T, S>,
    pub ctrl: fn(SVector<T, S>, T) -> SVector<T, S>,
    pub obs: fn(SVector<T, O>, T) -> SVector<T, O>,
}

pub trait Filter<T, const S: usize, const O: usize>
where
    T: RealField,
{
    fn predict(&mut self, state: &State<T, S>, ctrl: &SVector<T, S>) -> State<T, S>;
    fn update(&mut self, state: &State<T, S>, obs: &SVector<T, O>) -> State<T, S>;
}

#[derive(Debug)]
pub struct SigmaPoints<T, const S: usize, const N: usize>
where
    T: RealField,
{
    pub mean_weights: SVector<T, N>,
    pub cov_weights: SVector<T, N>,
    pub sigmas: SMatrix<T, N, S>,
}

pub trait SigmaPointGenerator<T, const S: usize, const N: usize>
where
    T: RealField,
{
    fn generate_sigmas(&self, mean: &SVector<T, S>, cov: &SMatrix<T, S, S>)
        -> SigmaPoints<T, S, N>;
}

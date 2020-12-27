extern crate nalgebra as na;

use na::{Dim, DimName, RealField};

enum MatrixSqrtMethod {
    Cholesky,
    BlockedSchur,
}

pub trait SigmaPointGenerator<T, D>
where
    T: RealField,
    D: Dim + DimName,
{
    fn generate(&self);
}

pub struct SimplexGenerator<T, D>
where
    T: RealField,
    D: Dim + DimName,
{
    pub sqrt_method: MatrixSqrtMethod,
}

pub struct JulierGenerator<T, D>
where
    T: RealField,
    D: Dim + DimName,
{
    pub scale_factor: T,
    pub sqrt_method: MatrixSqrtMethod,
}


pub struct MerweGenerator<T, D>
where
    T: RealField,
    D: Dim + DimName,
{
    pub sqrt_method: MatrixSqrtMethod,
}

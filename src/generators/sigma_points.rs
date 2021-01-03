extern crate nalgebra as na;

use na::allocator::Allocator;
use na::linalg::Cholesky;
use na::{DefaultAllocator, Dim, DimName, MatrixMN, MatrixN, RealField, VectorN};

pub struct SigmaPoints<T, DState, DSigma>
where
    T: RealField,
    DState: Dim + DimName,
    DSigma: Dim + DimName,
    DefaultAllocator: Allocator<T, DSigma> + Allocator<T, DSigma, DState>,
{
    pub weights: VectorN<T, DSigma>,
    pub sigmas: MatrixMN<T, DSigma, DState>,
}

pub trait SigmaPointGenerator<T, DState, DSigma>
where
    T: RealField,
    DState: Dim + DimName,
    DSigma: Dim + DimName,
    DefaultAllocator: Allocator<T, DState>
        + Allocator<T, DState, DState>
        + Allocator<T, DSigma, DState>
        + Allocator<T, DSigma>,
{
    fn generate_sigmas(
        &self,
        mean: &VectorN<T, DState>,
        cov: &MatrixN<T, DState>,
    ) -> SigmaPoints<T, DState, DSigma>;
    fn generate_weights(&self);
}

pub struct SimplexGenerator<T>
where
    T: RealField,
{
    pub kappa: T,
}

// impl<T, D> SigmaPointGenerator<T, D> for SimplexGenerator<T>
// where
//     T: RealField,
//     D: Dim + DimName,
//     DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
// {
//     fn generate_sigmas(&self, mean: &VectorN<T, D>, cov: &MatrixN<T, D>) -> VectorN<T, D> {
//         let sqrt_cov = Cholesky::<T, D>::new(*cov);
//     }

//     fn generate_weights(&self) {}
// }

pub struct JulierGenerator<T>
where
    T: RealField,
{
    pub scale_factor: T,
}

// impl<T, D> SigmaPointGenerator<T, D> for JulierGenerator<T>
// where
//     T: RealField,
//     D: Dim + DimName,
//     DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
// {
//     fn generate_sigmas(&self, mean: &VectorN<T, D>, cov: &MatrixN<T, D>) -> VectorN<T, D> {
//         let sqrt_cov = Cholesky::<T, D>::new(*cov);
//     }

//     fn generate_weights(&self) {}
// }

pub struct MerweGenerator<T>
where
    T: RealField,
{
    pub alpha: T,
    pub beta: T,
    pub kappa: T,
    pub dim: T,
}

impl<T, DState, DSigma> SigmaPointGenerator<T, DState, DSigma> for MerweGenerator<T>
where
    T: RealField,
    DState: Dim + DimName,
    DSigma: Dim + DimName,
    DefaultAllocator: Allocator<T, DState>
        + Allocator<T, DState, DState>
        + Allocator<T, DSigma, DState>
        + Allocator<T, DSigma>,
{
    fn generate_sigmas(&self, mean: &VectorN<T, DState>, cov: &MatrixN<T, DState>) -> SigmaPoints<T, DState, DSigma> {
        let lambda = (self.alpha * self.alpha) * (self.dim + self.kappa) - self.dim;
        let scaled_cov = cov.scale(self.dim + lambda);
        let sqrt_cov = Cholesky::<T, DState>::new(scaled_cov).unwrap().unpack();
        let mut sigmas = MatrixMN::<T, DSigma, DState>::from_row_slice(mean.as_slice());

        //
        for i in 1..mean.len() {
            let mut slice = sigmas.row_mut(i);
            slice += sqrt_cov.row(i - 1);
        }

        //
        for i in mean.len()..sigmas.nrows() {
            let mut slice = sigmas.row_mut(i);
            slice -= sqrt_cov.row(i - mean.len());
        }

        let weights = VectorN::<T, DSigma>::zeros();

        SigmaPoints::<T, DState, DSigma> {
            weights: weights,
            sigmas: sigmas,
        }
    }

    fn generate_weights(&self) {}
}

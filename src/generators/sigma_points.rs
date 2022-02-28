use crate::base::{SigmaPointGenerator, SigmaPoints};
use nalgebra::{RealField, SMatrix, SVector};

pub struct SimplexGenerator<T>
where
    T: RealField,
{
    pub kappa: T,
}

pub struct JulierGenerator<T>
where
    T: RealField,
{
    pub scale_factor: T,
}

pub struct MerweGenerator<T, const S: usize, const N: usize>
where
    T: RealField,
{
    pub alpha: T,
    pub beta: T,
    pub kappa: T,
    pub dim: T,
}

impl<T, const S: usize, const N: usize> SigmaPointGenerator<T, S, N> for MerweGenerator<T, S, N>
where
    T: RealField + num::NumCast,
{
    fn generate_sigmas(
        &self,
        mean: &SVector<T, S>,
        cov: &SMatrix<T, S, S>,
    ) -> SigmaPoints<T, S, N> {
        let lambda = (self.alpha * self.alpha) * (self.dim + self.kappa) - self.dim;
        let scaled_cov = cov.scale(self.dim + lambda);
        let sqrt_cov = scaled_cov.cholesky().unwrap().unpack();
        let mut sigmas = SMatrix::<T, N, S>::from_row_slice(mean.as_slice());

        for i in 1..mean.len() {
            let mut slice = sigmas.row_mut(i);
            slice += sqrt_cov.row(i - 1);
        }

        for i in mean.len()..sigmas.nrows() {
            let mut slice = sigmas.row_mut(i);
            slice -= sqrt_cov.row(i - mean.len());
        }

        let weight_sf = num::traits::cast::<u8, T>(1).unwrap()
            / (num::traits::cast::<u8, T>(2).unwrap() * (self.dim + lambda));
        let mut mean_weights = SVector::<T, N>::from_element(weight_sf);
        let mut cov_weights = SVector::<T, N>::from_element(weight_sf);

        mean_weights[0] = lambda / (self.dim + lambda);
        cov_weights[0] = (lambda / (self.dim + lambda)) - (self.alpha * self.alpha)
            + self.beta
            + num::traits::cast(1).unwrap();

        SigmaPoints::<T, S, N> {
            mean_weights,
            cov_weights,
            sigmas,
        }
    }
}

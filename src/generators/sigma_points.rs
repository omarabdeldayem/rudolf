extern crate nalgebra as na;

use na::{RealField, SMatrix, SVector};

pub struct SigmaPoints<T, const DSTATE: usize, const DSIGMA: usize>
where
    T: RealField,
{
    pub mean_weights: SVector<T, DSIGMA>,
    pub cov_weights: SVector<T, DSIGMA>,
    pub sigmas: SMatrix<T, DSIGMA, DSTATE>,
}

pub trait SigmaPointGenerator<T, const DSTATE: usize, const DSIGMA: usize>
where
    T: RealField,
{
    fn generate_sigmas(
        &self,
        mean: &SVector<T, DSTATE>,
        cov: &SMatrix<T, DSTATE, DSTATE>,
    ) -> SigmaPoints<T, DSTATE, DSIGMA>;
}

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

pub struct MerweGenerator<T>
where
    T: RealField,
{
    pub alpha: T,
    pub beta: T,
    pub kappa: T,
    pub dim: T,
}

impl<T, const DSTATE: usize, const DSIGMA: usize> SigmaPointGenerator<T, DSTATE, DSIGMA>
    for MerweGenerator<T>
where
    T: RealField + num::NumCast,
{
    fn generate_sigmas(
        &self,
        mean: &SVector<T, DSTATE>,
        cov: &SMatrix<T, DSTATE, DSTATE>,
    ) -> SigmaPoints<T, DSTATE, DSIGMA> {
        let lambda = (self.alpha * self.alpha) * (self.dim + self.kappa) - self.dim;
        let scaled_cov = cov.scale(self.dim + lambda);
        let sqrt_cov = scaled_cov.cholesky().unwrap().unpack();
        let mut sigmas = SMatrix::<T, DSIGMA, DSTATE>::from_row_slice(mean.as_slice());

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

        let weight_sf = num::traits::cast::<u8, T>(1).unwrap()
            / (num::traits::cast::<u8, T>(2).unwrap() * (self.dim + lambda));
        let mut mean_weights = SVector::<T, DSIGMA>::from_element(weight_sf);
        let mut cov_weights = SVector::<T, DSIGMA>::from_element(weight_sf);

        mean_weights[0] = lambda / (self.dim + lambda);
        cov_weights[0] = (lambda / (self.dim + lambda)) - (self.alpha * self.alpha)
            + self.beta
            + num::traits::cast(1).unwrap();

        SigmaPoints::<T, DSTATE, DSIGMA> {
            mean_weights: mean_weights,
            cov_weights: cov_weights,
            sigmas: sigmas,
        }
    }
}

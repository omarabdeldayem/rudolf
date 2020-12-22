extern crate nalgebra as na;

use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, MatrixN, RealField, Vector3, VectorN, U3};

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
    obs: MatrixN<T, D>,
    ctrl: MatrixN<T, D>,
}

struct Models<T, D>
where
    T: RealField,
    D: Dim + DimName,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    obs: MatrixN<T, D>,
    ctrl: MatrixN<T, D>,
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
    fn update(&mut self, obs: &VectorN<T, D>) {
        let inovation = &self.state.cov
            * &self.models.obs
            * (&self.models.obs * &self.state.cov * &self.models.obs.transpose()
                + &self.noise.obs)
                .try_inverse()
                .unwrap();
        self.state.mean = &self.state.mean
            + (&inovation * (obs - (&self.models.obs * &self.state.mean)));
        self.state.cov =
            (MatrixN::<T, D>::identity() - &inovation * &self.models.obs) * &self.state.cov;
    }

    fn predict(&mut self, ctrl: &VectorN<T, D>) {
        self.state.mean =
            (&self.models.obs * &self.state.mean) + (&self.models.ctrl * ctrl);
        self.state.cov =
            &self.models.obs * &self.state.cov * &self.models.obs.transpose()
                + &self.noise.ctrl;
    }
}

fn main() {
    let mut filter = KalmanFilter::<f32, U3> {
        state: State::<f32, U3> {
            mean: Vector3::new(1.0, 1.0, 1.0),
            cov: MatrixN::from_element_generic(U3, U3, 0.5),
        },
        models: Models::<f32, U3> {
            obs: MatrixN::from_row_slice_generic(
                U3,
                U3,
                &[1.0, 0.8, 1.5, 2.0, 1.0, 4.0, 1.2, 1.5, 1.0],
            ),
            ctrl: MatrixN::from_row_slice_generic(
                U3,
                U3,
                &[1.0, 0.5, 0.0, 2.0, 0.0, 4.0, 0.0, 3.0, 1.0],
            ),
        },
        noise: Noise::<f32, U3> {
            obs: MatrixN::from_element_generic(U3, U3, 0.01),
            ctrl: MatrixN::from_element_generic(U3, U3, 0.02),
        },
    };

    let new_obs = Vector3::new(2.0, 1.2, 3.4);
    let new_ctrl = Vector3::new(2.0, 1.4, 2.3);

    filter.predict(&new_ctrl);
    filter.update(&new_obs);

    println!("{:?}", filter.state.mean);
}

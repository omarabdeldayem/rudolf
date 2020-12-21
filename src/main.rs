extern crate nalgebra as na;

use na::{RealField, DMatrix, DVector};

struct State<T: RealField> {
    mean: DVector<T>,
    cov: DMatrix<T>,
}

struct Noise<T: RealField> {
    observation: DMatrix<T>,
    control: DMatrix<T>,
}

struct Models<T: RealField> {
    observation: DMatrix<T>,
    control: DMatrix<T>,
}

struct KalmanFilter<T: RealField> {
    state: State<T>,
    models: Models<T>,
    noise: Noise<T>,
}

impl<T> KalmanFilter<T>
where
    T: RealField,
{
    fn update(&mut self, observation: &DVector<T>) {
        let inovation: DMatrix<T> = &self.state.cov * &self.models.observation
            * (&self.models.observation * &self.state.cov * &self.models.observation.transpose()
                + &self.noise.observation)
            .try_inverse()
            .unwrap();
        self.state.mean = &self.state.mean + (&inovation * (observation - (&self.models.observation * &self.state.mean)));
        // this needs compile-time dimensional genericity to properly size the identity
        // self.state.cov = (DMatrix::identity() - &inovation * &self.models.observation) * self.state.cov;
    }

    fn predict(&mut self, control: &DVector<T>) {
        self.state.mean = (&self.models.observation * &self.state.mean) + (&self.models.control * control);
        self.state.cov = &self.models.observation * &self.state.cov * &self.models.observation.transpose() + &self.noise.control;
    }
}

fn main() {
    // let a = array![1.0, 2.0, 3.0];
    // let b = array![4.0, 5.0, 6.0];

    // println!("{:?}", a);
    // println!("{:?}", b);
}

use ndarray::prelude::*;
use num_traits::Float;

extern crate num_traits;

struct State<T: Float + 'static> {
    mean: Array1<T>,
    cov: Array2<T>,
}

struct Noise<T: Float + 'static> {
    measurement: Array2<T>,
    process: Array2<T>,
}

struct KalmanFilter<T: Float + 'static> {
    init_state: State<T>,
    trans: Array2<T>,
    obs: Array2<T>,
    noise: Noise<T>,
}

impl<T> KalmanFilter<T>
where
    T: Float + 'static,
{
    fn update(&self, state: State<T>) {

    }
}

impl<T> State<T>
where
    T: Float + 'static,
{

    fn filter(self: &mut Self, kf: &KalmanFilter<T>) {
        self.mean = kf.trans.dot(&self.mean);
    }

}

fn main() {
    let a = array![1.0, 2.0, 3.0];
    let b = array![4.0, 5.0, 6.0];

    println!("{:?}", a);
    println!("{:?}", b);
}

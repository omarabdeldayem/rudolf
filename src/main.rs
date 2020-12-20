use ndarray::prelude::*;

struct State<T: ::ndarray::RawData, D> {
    mean: Array1<T>,
    cov: ArrayBase<T, D>,
}

struct Noise<T: ::ndarray::RawData, D> {
    measurement: ArrayBase<T, D>,
    process: ArrayBase<T, D>,
}

struct KalmanFilter<T: ::ndarray::RawData, D> {
    init_state: State<T, D>,
    trans: ArrayBase<T, D>,
    obs: ArrayBase<T, D>,
    noise: Noise<T, D>,
}

impl<T: ::ndarray::RawData, D> KalmanFilter<T, D> {
    fn update(state: State<T, D>) {

    }
}

impl<T: ::ndarray::RawData, D> State<T, D> {
    fn filter(kf: KalmanFilter<T, D>) {

    }
}

fn main() {
    let a = array![1.0, 2.0, 3.0];
    let b = array![4.0, 5.0, 6.0];

    println!("{:?}", a);
    println!("{:?}", b);
}

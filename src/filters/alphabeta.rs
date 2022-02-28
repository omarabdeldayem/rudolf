use crate::base::{Filter, Model, State};
use nalgebra::{RealField, SMatrix, SVector};

pub struct AlphaBetaFilter<T, const S: usize, const O: usize>
where
    T: RealField,
{
    pub alpha: T,
    pub beta: T,
    pub time_step: T,
    pub model: Model<T, S, O>,
}

impl<T, const S: usize, const O: usize> AlphaBetaFilter<T, S, O>
where
    T: RealField,
{
    pub fn new(alpha: T, beta: T, time_step: T, model: Model<T, S, O>) -> AlphaBetaFilter<T, S, O> {
        AlphaBetaFilter {
            alpha,
            beta,
            time_step,
            model,
        }
    }
}

impl<T, const S: usize, const O: usize> Filter<T, S, O> for AlphaBetaFilter<T, S, O>
where
    T: RealField,
{
    fn predict(&mut self, state: &State<T, S>, ctrl: &SVector<T, S>) -> State<T, S> {
        let ctrl = self.model.state.diagonal() + ctrl;
        let predicted = state.mean + ctrl * self.time_step;

        State {
            mean: predicted,
            cov: state.cov,
        }
    }

    fn update(&mut self, state: &State<T, S>, obs: &SVector<T, O>) -> State<T, S> {
        let innovation = state.mean - (self.model.obs.transpose() * obs);
        let update = state.mean + (innovation * self.alpha);
        self.model.state +=
            SMatrix::<T, S, S>::from_diagonal(&(innovation * self.beta / self.time_step));

        State {
            mean: update,
            cov: state.cov,
        }
    }
}

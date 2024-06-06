use ndarray::Array2;
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};

use crate::{
    params::{OpenESParams, Params},
    states::{OpenESState, State},
};

#[derive(Debug)]
pub enum Algo {
    OpenES(usize, usize),
    // OtherAlgo,
}

impl Algo {
    pub fn default_params(&self) -> Params {
        match self {
            Algo::OpenES(_, _) => Params::OpenES(OpenESParams::default_params()),
            _ => Params::OtherParams,
        }
    }

    pub fn init_algorithm(&self, params: &Params) -> State {
        match (self, params) {
            (Algo::OpenES(_, num_dims), Params::OpenES(_)) => {
                State::OpenES(OpenESState::init_state(num_dims, params))
            }
            _ => State::OtherState,
        }
    }

    pub fn ask(&self, state: &State, params: &Params) -> Array2<f32> {
        match (self, state, params) {
            (Algo::OpenES(popsize, num_dims), State::OpenES(state), Params::OpenES(params)) => {
                let mut pop: Array2<f32> = Array2::random((*popsize, *num_dims), StandardNormal);
                // Given the shapes and following operation,
                // each row of the population is an independent draw
                // from a multivariate normal distribution with mean 'state.mean'
                // and standard deviation 'state.sigma'.
                // Alternatively, each column corresponds to draws from
                // a normal distribution with mean 'state.mean[j]'
                // and standard deviation 'state.sigma[j]'.
                pop = pop * &state.sigma + &state.mean;
                pop = pop.map(|x| x.clamp(params.clip_min, params.clip_max));
                pop
            }
            _ => unimplemented!(),
        }
    }

    pub fn tell(
        &self,
        pop: Array2<f32>,
        fitness: Array2<f32>,
        state: State,
        params: &Params,
    ) -> State {
        match (self, state, params) {
            (Algo::OpenES(popsize, _), State::OpenES(state), Params::OpenES(params)) => {
                // Reconstruct z
                let noise: Array2<f32> = (pop - &state.mean) / &state.sigma;
                // The logic of the following operation is to have a vector same length as
                // state.mean and state.sigma that will serve as pseudo-grads for the update
                // so the dot operation weights the population (z) with its fitness
                // and then scaling back by state.sigma
                let grads: Array2<f32> =
                    1.0 / (*popsize as f32 * &state.sigma) * noise.t().dot(&fitness).t();
                let grads_ewa: Array2<f32> =
                    params.beta_1 * &state.grads_ewa + (1.0 - params.beta_1) * &grads;
                let grads_sq_ewa: Array2<f32> =
                    params.beta_2 * &state.grads_sq_ewa + (1.0 - params.beta_2) * (&grads * &grads);
                // Initially, the ewas might be small, even more so given their initialization,
                // i.e.
                let grads_ewa_adj: Array2<f32> =
                    &grads_ewa / (1.0 - params.beta_1.powi(&state.gen_counter + 1));
                let grads_sq_ewa_adj: Array2<f32> =
                    &grads_sq_ewa / (1.0 - params.beta_2.powi(&state.gen_counter + 1));
                let mean: Array2<f32> = &state.mean
                    - params.learning_rate * &grads_ewa_adj
                        / (&grads_sq_ewa_adj * &grads_sq_ewa_adj)
                    + params.eps;
                // TODO:
                // Implement decay on learning_rate
                // Enhance decay on sigma
                let sigma = &state.sigma * 0.99; // max to not converge at zero...
                State::OpenES(OpenESState {
                    mean,
                    sigma,
                    grads_ewa,
                    grads_sq_ewa,
                    gen_counter: &state.gen_counter + 1,
                    ..state
                })
            }
            _ => unimplemented!(),
        }
    }
}

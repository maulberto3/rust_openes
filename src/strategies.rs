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

    pub fn ask(&self, state: &State) -> Array2<f32> {
        match (self, state) {
            (Algo::OpenES(popsize, num_dims), State::OpenES(state)) => {
                let mut pop: Array2<f32> = Array2::random((*popsize, *num_dims), StandardNormal);
                // Given the shapes and following operation,
                // each row of the population is an independent draw
                // from a multivariate normal distribution with mean 'state.mean'
                // and standard deviation 'state.sigma'.
                // Alternatively, each column corresponds to draws from
                // a normal distribution with mean 'state.mean[j]'
                // and standard deviation 'state.sigma[j]'.
                pop = pop * &state.sigma + &state.mean;
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
    ) -> (Array2<f32>, State) {
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
                let m: Array2<f32> = (1.0 - params.beta_1) * &grads + params.beta_1 * &state.m;
                let v: Array2<f32> =
                    (1.0 - params.beta_2) * (&grads.map(|x| x.powi(2))) + params.beta_2 * &state.v;
                let mhat: Array2<f32> = &m / (1.0 - params.beta_1.powi(&state.gen_counter + 1));
                let vhat: Array2<f32> = &v / (1.0 - params.beta_2.powi(&state.gen_counter + 1));
                let mean_new: Array2<f32> = &state.mean
                    - params.learning_rate * &mhat / (vhat.map(|x| x.powf(0.5)) + params.eps);
                (
                    mean_new,
                    State::OpenES(OpenESState {
                        m,
                        v,
                        gen_counter: &state.gen_counter + 1,
                        ..state
                    }),
                )

                // TODO: continue with update of optim

                // def update(self, state: OptState, params: OptParams) -> OptState:
                // """Exponentially decay the learning rate if desired."""
                // lrate = exp_decay(state.lrate, params.lrate_decay, params.lrate_limit)
                // return state.replace(lrate=lrate)

                // def exp_decay(
                //     param: chex.Array, param_decay: chex.Array, param_limit: chex.Array
                // ) -> chex.Array:
                //     """Exponentially decay parameter & clip by minimal value."""
                //     param = param * param_decay
                //     param = jnp.maximum(param, param_limit)
                //     return param
            }
            _ => unimplemented!(),
        }
    }
}

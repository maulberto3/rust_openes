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
        state: State,
        pop: Array2<f32>,
        fitness: Array2<f32>,
        params: &Params,
    ) -> State {
        match (self, state) {
            (Algo::OpenES(popsize, _), State::OpenES(state)) => {
                // Reconstruct z
                let noise: Array2<f32> = (pop - &state.mean) / &state.sigma;
                // The logic of the following operation is to have a vector same length as 
                // state.mean and state.sigma that will serve as pseudo-grads for the update
                // so the dot operation is weigting the population (z) given its fitness
                // and then scaling back by state.sigma
                let theta_grad: Array2<f32> = 1.0 / (*popsize as f32 * &state.sigma) * noise.t().dot(&fitness).t();

                //
                // TODO: Implement step function
                // 
                // Isolate Optimzers in optims.rs?
                // struct OptParams {
                //     beta_1: f64,
                //     beta_2: f64,
                //     eps: f64,
                //     learning_rate: f64,
                // }
                
                // struct OptState {
                //     m: Array1<f64>,      // First moment vector
                //     v: Array1<f64>,      // Second moment vector
                //     gen_counter: usize,  // Generation counter
                // }

                // impl OptState {
                //     fn new(num_dims: usize) -> Self {
                //         OptState {
                //             m: Array1::zeros(num_dims),
                //             v: Array1::zeros(num_dims),
                //             gen_counter: 0,
                //         }
                //     }
                
                //     fn step(&self, mean: &Array1<f64>, grads: &Array1<f64>, params: &OptParams) -> (Array1<f64>, OptState) {
                //         let beta_1 = params.beta_1;
                //         let beta_2 = params.beta_2;
                //         let eps = params.eps;
                
                //         let m = (1.0 - beta_1) * grads + beta_1 * &self.m;
                //         let v = (1.0 - beta_2) * (grads * grads) + beta_2 * &self.v;
                
                //         let gen_counter = self.gen_counter + 1;
                //         let mhat = &m / (1.0 - beta_1.powi(gen_counter as i32));
                //         let vhat = &v / (1.0 - beta_2.powi(gen_counter as i32));
                
                //         let mean_new = mean - &(params.learning_rate * &mhat / (vhat.mapv(f64::sqrt) + eps));
                
                //         let new_state = OptState {
                //             m,
                //             v,
                //             gen_counter,
                //         };
                
                //         (mean_new, new_state)
                //     }
                // }

                
                

                //             class Adam(Optimizer):
                // def __init__(self, num_dims: int):
                //     """JAX-Compatible Adam Optimizer (Kingma & Ba, 2015)
                //     Reference: https://arxiv.org/abs/1412.6980"""
                //     super().__init__(num_dims)
                //     self.opt_name = "adam"

                // @property
                // def params_opt(self) -> Dict[str, float]:
                //     """Return default Adam parameters."""
                //     return {
                //         "beta_1": 0.99,
                //         "beta_2": 0.999,
                //         "eps": 1e-8,
                //     }

                // def initialize_opt(self, params: OptParams) -> OptState:
                //     """Initialize the m, v trace of the optimizer."""
                //     return OptState(
                //         m=jnp.zeros(self.num_dims),
                //         v=jnp.zeros(self.num_dims),
                //         lrate=params.lrate_init,
                //     )

                // def step_opt(
                //     self,
                //     mean: chex.Array,
                //     grads: chex.Array,
                //     state: OptState,
                //     params: OptParams,
                // ) -> Tuple[chex.Array, OptState]:
                //     """Perform a simple Adam GD step."""
                //     m = (1 - params.beta_1) * grads + params.beta_1 * state.m
                //     v = (1 - params.beta_2) * (grads ** 2) + params.beta_2 * state.v
                //     mhat = m / (1 - params.beta_1 ** (state.gen_counter + 1))
                //     vhat = v / (1 - params.beta_2 ** (state.gen_counter + 1))
                //     mean_new = mean - state.lrate * mhat / (jnp.sqrt(vhat) + params.eps)
                //     return mean_new, state.replace(
                //         m=m, v=v, gen_counter=state.gen_counter + 1
                //     )

                State::OpenES(state)
            }
            _ => unimplemented!(),
        }
    }
}

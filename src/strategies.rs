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
                pop = pop * &state.sigma + &state.mean;
                pop
            }
            _ => unimplemented!(),
        }
    }

    // fn tell(&self, state: State, pop: Array2<f32>, fitness: Array2<f32>, params: &Params) -> State {
    //     match state {
    //         State::OpenES(state) => {
    //             let popsize;
    //             let num_dims;
    //             if let Algo::OpenES(size, dims) = &self {
    //                 popsize = *size;
    //                 num_dims = *dims;
    //             } else {
    //                 // TODO: Enchance this
    //                 // return Array2::<f32>::ones((3, 3));
    //                 unimplemented!()
    //             }
    //             // let noise: Array2<f32> = (pop - &state.mean) / &state.sigma;
    //             // let theta_grad_1: Array2<f32> = 1.0 / (popsize as f32 * &state.sigma);// * noise.t().dot(&fitness);
    //             // dbg!(&theta_grad_1.shape());
    //             // let theta_grad_2: Array2<f32> = noise.t().dot(&fitness);
    //             // dbg!(&theta_grad_2.shape());

    //             //             class Adam(Optimizer):
    //             // def __init__(self, num_dims: int):
    //             //     """JAX-Compatible Adam Optimizer (Kingma & Ba, 2015)
    //             //     Reference: https://arxiv.org/abs/1412.6980"""
    //             //     super().__init__(num_dims)
    //             //     self.opt_name = "adam"

    //             // @property
    //             // def params_opt(self) -> Dict[str, float]:
    //             //     """Return default Adam parameters."""
    //             //     return {
    //             //         "beta_1": 0.99,
    //             //         "beta_2": 0.999,
    //             //         "eps": 1e-8,
    //             //     }

    //             // def initialize_opt(self, params: OptParams) -> OptState:
    //             //     """Initialize the m, v trace of the optimizer."""
    //             //     return OptState(
    //             //         m=jnp.zeros(self.num_dims),
    //             //         v=jnp.zeros(self.num_dims),
    //             //         lrate=params.lrate_init,
    //             //     )

    //             // def step_opt(
    //             //     self,
    //             //     mean: chex.Array,
    //             //     grads: chex.Array,
    //             //     state: OptState,
    //             //     params: OptParams,
    //             // ) -> Tuple[chex.Array, OptState]:
    //             //     """Perform a simple Adam GD step."""
    //             //     m = (1 - params.beta_1) * grads + params.beta_1 * state.m
    //             //     v = (1 - params.beta_2) * (grads ** 2) + params.beta_2 * state.v
    //             //     mhat = m / (1 - params.beta_1 ** (state.gen_counter + 1))
    //             //     vhat = v / (1 - params.beta_2 ** (state.gen_counter + 1))
    //             //     mean_new = mean - state.lrate * mhat / (jnp.sqrt(vhat) + params.eps)
    //             //     return mean_new, state.replace(
    //             //         m=m, v=v, gen_counter=state.gen_counter + 1
    //             //     )

    //             State::OpenES(state)
    //         }
    //         _ => unimplemented!(),
    //     }
    // }
}

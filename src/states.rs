use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

use crate::params::Params;

#[derive(Debug)]
pub enum State {
    OpenES(OpenESState),
    OtherState,
}

#[derive(Debug)]
pub struct OpenESState {
    pub mean: Array2<f32>,
    pub sigma: Array2<f32>,
    // // optimizer state...
    pub best_member: Array2<f32>,
    // pub best_fitness: f32,  // jnp.finfo(jnp.float32).max,
    // pub gen_counter: usize
}

impl OpenESState {
    pub fn init_state(num_dims: &usize, params: &Params) -> Self {
        match params {
            Params::OpenES(params) => {
                let mean = Array2::<f32>::random(
                    (1, *num_dims),
                    Uniform::<f32>::new(params.init_min, params.init_max),
                );
                let sigma = Array2::ones((1, *num_dims));
                let best_member = mean.clone();
                OpenESState {
                    mean,
                    sigma,
                    // opt_state,
                    best_member,
                }
            }
            _ => unimplemented!(),
        }
    }
}

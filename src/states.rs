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
    pub best_member: Array2<f32>,
    // pub best_fitness: f32,  // jnp.finfo(jnp.float32).max,
    // pub gen_counter: usize
    // optimizer state...
    pub grads_ewa: Array2<f32>,
    pub grads_sq_ewa: Array2<f32>,
    pub gen_counter: i32,
}

impl OpenESState {
    pub fn init_state(num_dims: &usize, params: &Params) -> Self {
        if let Params::OpenES(params) = params {
            let mean: Array2<f32> = Array2::<f32>::random(
                (1, *num_dims),
                Uniform::<f32>::new(params.init_min, params.init_max),
            );
            // TODO: `sigma=jnp.ones(self.num_dims) * params.sigma_init,`
            let sigma: Array2<f32> = Array2::ones((1, *num_dims));
            let best_member: Array2<f32> = mean.clone();
            // optim init state
            let grads_ewa: Array2<f32> = Array2::zeros((1, *num_dims));
            let grads_sq_ewa: Array2<f32> = Array2::zeros((1, *num_dims));
            let gen_counter = 0;

            OpenESState {
                mean,
                sigma,
                best_member,
                grads_ewa,
                grads_sq_ewa,
                gen_counter,
            }
        } else {
            unreachable!("Expected Params::OpenES")
        }
    }

    // pub fn step(&self, mean: &Array1<f64>, grads: &Array1<f64>, params: &Params) -> (Array1<f64>, OptState) {
    //     if let Params::OpenES(params) = params {
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
    //     } else {
    //         unreachable!("Expected Params::OpenES")
    //     }
    // }
}

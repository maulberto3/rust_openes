
// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let popsize = 20;
//         let num_dims = 50;
//         let algo = OpenES::new(popsize, num_dims);
//         assert_eq!(algo.popsize, popsize);
//         assert_eq!(algo.num_dims, num_dims);
//     }
// }

// pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
// use_antithetic_sampling: bool,
// opt_name: String, // = "adam",
// pub lrate_init: f32, // = 0.05,
// lrate_decay: f32, // = 1.0,
// lrate_limit: f32, // = 0.001,
// sigma_init: f32, // = 0.03,
// sigma_decay: f32, // = 1.0,
// sigma_limit: f32, // = 0.01,
// mean_decay: f32, // = 0.0,
// n_devices: Optional[int] = None,
// **fitness_kwargs: Union[bool, int, float]

// #[derive(Debug)]
// struct OpenES {
//     pub popsize: usize,
//     pub num_dims: usize,
// }

// #[derive(Debug)]
// struct OpenESParams {
//     pub init_min: f32,
//     pub init_max: f32,
// }

// #[derive(Debug)]
// struct OpenESState {
//     pub mean: Vec<f32>,
// }

// impl OpenES {
//     fn new(popsize: usize, num_dims: usize) -> Self {
//         OpenES { popsize, num_dims }
//     }

//     fn default_params(&self) -> OpenESParams {
//         OpenESParams {
//             init_min: 0.0,
//             init_max: 1.0,
//         }
//     }

//     fn init_algorithm(&self, params: &OpenESParams) -> OpenESState {
//         let mut mean: Vec<f32> = Vec::new();
//         let mut rng = rand::thread_rng();
//         let distr = Uniform::new(params.init_min as f64, params.init_max as f64).unwrap();
//         for _i in 0..self.num_dims {
//             mean.push(distr.sample(&mut rng) as f32);
//         }
//         OpenESState { mean }
//     }
// }

// pub fn work() -> Result<()> {
//     let popsize = 20;
//     let num_dims = 20;
//     let algo = OpenES::new(popsize, num_dims);
//     dbg!(&algo);

//     let params = algo.default_params();
//     dbg!(&params);

//     let state = algo.init_algorithm(&params);
//     dbg!(&state);

//     Ok(())
// }

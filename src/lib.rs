use anyhow::Result;
use rand::prelude::*;
use rayon::prelude::*;
use statrs::distribution::{Normal, Uniform};

#[derive(Debug)]
enum EvoAlgo {
    OpenES(usize, usize),
    Other,
}

#[derive(Debug)]
enum EvoParams {
    OpenES(OpenESParams),
    Other,
}

#[derive(Debug)]
pub struct OpenESParams {
    pub init_min: f64,
    pub init_max: f64,
    pub sigma_init: f64,
}

impl OpenESParams {
    fn default_params() -> Self {
        OpenESParams {
            init_min: -1.0,
            init_max: 1.0,
            sigma_init: 1.0,
        }
    }
}

#[derive(Debug)]
enum EvoState {
    OpenES(OpenESState),
    Other,
}

#[derive(Debug, Clone)]
pub struct OpenESState {
    pub mean: Vec<f64>,
    pub sigma: Vec<f64>,
    // optimizer state...
    pub best_member: Vec<f64>,
}

impl OpenESState {
    fn init_state(algo: &EvoAlgo, params: &EvoParams) -> Self {
        match (algo, params) {
            (EvoAlgo::OpenES(_, num_dims), EvoParams::OpenES(params)) => {
                let distr = Uniform::new(params.init_min, params.init_max).unwrap();
                let mean: Vec<f64> = (0..*num_dims)
                    .into_par_iter()
                    .map(|_| {
                        let mut rng = rand::thread_rng();
                        distr.sample(&mut rng)
                    })
                    .collect();
                let sigma = vec![1.0 * params.sigma_init; *num_dims];
                let best_member = mean.clone();
                OpenESState {
                    mean,
                    sigma,
                    best_member,
                }
            }
            _ => unimplemented!(),
        }
    }
}

impl EvoAlgo {
    fn default_params(&self) -> EvoParams {
        match self {
            EvoAlgo::OpenES(_, _) => EvoParams::OpenES(OpenESParams::default_params()),
            _ => EvoParams::Other,
        }
    }

    fn init_algorithm(&self, params: &EvoParams) -> EvoState {
        match self {
            EvoAlgo::OpenES(_, _) => EvoState::OpenES(OpenESState::init_state(self, params)),
            _ => EvoState::Other,
        }
    }

    fn ask(&self, state: EvoState) -> (Vec<Vec<f64>>, EvoState) {
        match (self, state) {
            (EvoAlgo::OpenES(popsize, _), EvoState::OpenES(state)) => {
                let pop: Vec<Vec<f64>> = (0..*popsize)
                    .into_par_iter() // Convert the iterator into a parallel iterator
                    .map(|_indiv| {
                        let mut rng = rand::thread_rng();
                        state
                            .mean
                            .iter()
                            .zip(state.sigma.iter())
                            .map(|(mean, sigma)| {
                                let distr = Normal::new(*mean, *sigma).unwrap();
                                distr.sample(&mut rng)
                            })
                            .collect()
                    }) // Use map to generate population in parallel
                    .collect(); // Collect the results into a vector

                (pop, EvoState::OpenES(state))
            }
            _ => unimplemented!(),
        }
    }
}

pub fn work() -> Result<()> {
    let popsize = 5;
    let num_dims = 5;
    let openes = EvoAlgo::OpenES(popsize, num_dims);
    // dbg!(&openes);

    let params = openes.default_params();
    // dbg!(&params);
    if let EvoParams::OpenES(params) = &params {
        dbg!(&params.init_min);
        // dbg!(&params.init_max);
    }

    let state = openes.init_algorithm(&params);
    // dbg!(&state);
    if let EvoState::OpenES(state) = &state {
        dbg!(&state.mean);
        // dbg!(&state.sigma);
        // dbg!(&state.best_member);
    }

    let num_iters = 7;
    let x: (Vec<Vec<f64>>, EvoState) = openes.ask(state);
    // for _i in 0..num_iters {
    //     x = openes.ask(state);
    //     break;
    // }
    dbg!(&x);

    Ok(())
}

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

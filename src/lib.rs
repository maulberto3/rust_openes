use ::rand::distributions::Uniform;
use anyhow::Result;
use ndarray::{Array2, Axis};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};
use rayon::prelude::*;

#[derive(Debug)]
enum Algo {
    OpenES(usize, usize),
    OtherAlgo,
}

impl Algo {
    fn default_params(&self) -> Params {
        match self {
            Algo::OpenES(_, _) => Params::OpenES(OpenESParams::default_params()),
            _ => Params::OtherParams,
        }
    }

    fn init_algorithm(&self, popsize: &usize, num_dims: &usize, params: &Params) -> State {
        match params {
            Params::OpenES(_) => State::OpenES(OpenESState::init_state(popsize, num_dims, params)),
            _ => State::OtherState,
        }
    }

    fn ask(&self, popsize: &usize, num_dims: &usize, state: &State) -> Array2<f32> {
        match state {
            State::OpenES(state) => {
                let mut pop: Array2<f32> = Array2::random((*popsize, *num_dims), StandardNormal);
                pop = pop * &state.sigma + &state.mean;
                pop
            }
            _ => unimplemented!(),
        }
    }

    // fn tell(&self, state: State, pop: &Vec<Vec<f64>>, fitness: &Vec<f64>) -> State {
    //     match (self, state) {
    //         (Algo::OpenES(popsize, _), State::OpenES(state)) => {
    //             // and defintely here
    //             let noise = pop
    //                 .par_iter()
    //                 .map(|candidate| candidate.par_iter().map(|x| {}));

    //             State::OpenES(state)
    //         }
    //         _ => unimplemented!(),
    //     }
    // }
}

#[derive(Debug)]
enum Params {
    OpenES(OpenESParams),
    OtherParams,
}

#[derive(Debug)]
pub struct OpenESParams {
    // opt_params
    pub sigma_init: f32,
    pub sigma_decay: f32,
    pub sigma_limit: f32,
    pub init_min: f32,
    pub init_max: f32,
    pub clip_min: f32,
    pub clip_max: f32,
}

impl OpenESParams {
    fn default_params() -> Self {
        OpenESParams {
            // opt_params
            sigma_init: 1.0,
            sigma_decay: 0.999,
            sigma_limit: 0.04,
            init_min: -1.0,
            init_max: 1.0,
            clip_min: f32::NEG_INFINITY,
            clip_max: f32::INFINITY,
        }
    }
}

#[derive(Debug)]
enum State {
    OpenES((OpenESState)),
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
    fn init_state(popsize: &usize, num_dims: &usize, params: &Params) -> Self {
        match params {
            Params::OpenES(params) => {
                let mean = Array2::<f32>::random(
                    (*popsize, *num_dims),
                    Uniform::<f32>::new(params.init_min, params.init_max),
                );
                let sigma = Array2::ones((*popsize, *num_dims));
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

pub fn work() -> Result<()> {
    let popsize = 5;
    let num_dims = 5;
    let open_es = Algo::OpenES(popsize, num_dims);
    // dbg!(&open_es);

    let params = open_es.default_params();
    // dbg!(&params);

    let state = open_es.init_algorithm(&popsize, &num_dims, &params);
    // println!("{:6.4?}", &state);
    // dbg!(&state);

    let pop = open_es.ask(&popsize, &num_dims, &state);
    dbg!(&pop);

    fn fitness(pop: &Array2<f32>, popsize: &usize) -> Array2<f32> {
        pop.map_axis(Axis(0), |row| row.mapv(|elem| elem.powi(2)).sum())
            .into_shape((*popsize, 1))
            .unwrap()
    }

    let fit = fitness(&pop, &popsize);
    dbg!(&fit);

    // let num_iters = 7;
    // for _i in 0..num_iters {
    // pop, state = open_es.ask(state);
    // fit = fitness(&pop);
    // state = open_es.tell(state, &pop, &fit, &params);
    // state.best_member, state.best_fitness

    //     break;
    // }

    Ok(())
}

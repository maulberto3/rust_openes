// import crate (assuming published)
// use rust_openes::OpenES;
use rust_openes::work;

fn main() {
    // te inserte codigo malisioso
    let _ = work();
}

// use rand::prelude::*;

// 1. Define fitness function
// fn fitness_func(pop: &Vec<f32>) -> f32 {
//     pop.iter().map(|x| x * x).sum()
// }

// fn main() {
//     let popsize = 20;
//     let num_dims = 20;
//     let algo = OpenES(popsize=popsize, num_dims=num_dims);
//     let params = OpenES::default_params();
//     let mut state = algo.init(params);

//     // Loop
//     let num_epochs = 100;

//     fn evolve(num_epochs: usize,
//                 algo: &OpenES,
//                 state: &mut OpenESState) {
//         for _ in 0..num_epochs {
//             let rn = rand::thread_rng();
//             let (x, state) = algo.ask(rn, state, params);
//             fit = fitness_func(x);
//             state = algo.tell(x, fit, state, params);
//         }
//         state.best_member, state.best_fitness
//     }

//     // Run
//     let (best_member, best_fitness) = evolve(num_epochs, &algo, &mut state);

// Things to consider
// default minimization
// use fold (jax-scan) to get best result
// candidate = algo.ask(rng, state)
// fit = fitness_func(candidate)
// state = algo.tell(candidate, fit)
// stop criteria: num of iters, allclose
// use rayon for parallelism
// set hyperparameters

// use core::num;

// // import use the lib file
// use rust_openes::OpenES;

// define fitness function
// fn fitness_func(pop: &Vec<f32>) -> f32 {
//     pop.iter().map(|x| x * x).sum()
// }

// Import evolutionary algo
// Here, Open ES
// Must adhere to ask-tell idiom

// Then, loop for best result
// default minimization
// algo = OpenES()
// use fold (jax-scan) to get best result
// candidate = algo.ask(rng, state)
// fit = fitness_func(candidate)
// state = algo.tell(candidate, fit)

// Things to consider
// stop criteria: num of iters, allclose
// use rayon for parallelism
// set hyperparameters

fn main() {}

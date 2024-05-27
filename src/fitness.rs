use ndarray::{Array2, Axis};
use rayon::iter::{ParallelBridge, ParallelIterator};

pub fn square_and_sum(pop: &Array2<f32>) -> Array2<f32> {
    // pop.map_axis(Axis(0), |row| row.mapv(|elem| elem.powi(2)).sum())
    //     .into_shape((*popsize, 1))
    //     .unwrap()

    let popsize = pop.shape()[0];
    let fitness_values: Vec<f32> = pop
        .axis_iter(Axis(0))
        .par_bridge() // Parallelize the iteration
        .map(|row| row.map(|elem| elem.powi(2)).sum())
        .collect();
    Array2::from_shape_vec((popsize, 1), fitness_values).unwrap()
}

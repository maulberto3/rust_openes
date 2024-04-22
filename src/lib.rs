/// Creates the OpenES struct
#[derive(Debug)]
pub struct OpenES {
    popsize: usize,
    num_dims: usize,
    // pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
    // use_antithetic_sampling: bool,
    // opt_name: String, // = "adam",
    // lrate_init: f32, // = 0.05,
    // lrate_decay: f32, // = 1.0,
    // lrate_limit: f32, // = 0.001,
    // sigma_init: f32, // = 0.03,
    // sigma_decay: f32, // = 1.0,
    // sigma_limit: f32, // = 0.01,
    // mean_decay: f32, // = 0.0,
    // n_devices: Optional[int] = None,
    // **fitness_kwargs: Union[bool, int, float]
}

/// Implements the OpenES algorithm
/// Population size and number of dimensions are required
/// let popsize = 20;
/// let num_dims = 50;
/// let algo = OpenES::new(popsize, num_dims);
/// assert_eq!(algo.popsize, popsize);
/// assert_eq!(algo.num_dims, num_dims);
impl OpenES {
    pub fn new(popsize: usize, num_dims: usize) -> Self {
        Self { popsize, num_dims }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let popsize = 20;
        let num_dims = 50;
        let algo = OpenES::new(popsize, num_dims);
        assert_eq!(algo.popsize, popsize);
        assert_eq!(algo.num_dims, num_dims);
    }
}

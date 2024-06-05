use anyhow::Result;
use ndarray::Array2;

mod fitness;
use fitness::square_and_sum;

mod params;

mod states;

mod strategies;
use strategies::Algo;

/// `Result<T, Error>`
///
/// This is a reasonable return type to use throughout your application but also
/// for `fn main`; if you do, failures will be printed along with any
/// [context][Context] and a backtrace if one was captured.
///
/// `anyhow::Result` may be used with one *or* two type parameters.
///
/// ```rust
/// use anyhow::Result;
///
/// fn demo1() -> Result<T> {...}
///            // ^ equivalent to std::result::Result<T, anyhow::Error>
///
/// fn demo2() -> Result<T, OtherError> {...}
///            // ^ equivalent to std::result::Result<T, OtherError>
/// ```
///
/// # Example
///
/// ```
/// use anyhow::Result;
///
/// fn main() -> Result<()> {
///     # return Ok(());
///     let config = std::fs::read_to_string("cluster.json")?;
///     let map: ClusterMap = serde_json::from_str(&config)?;
///     println!("cluster info: {:#?}", map);
///     Ok(())
/// }
/// ```
pub fn work() -> Result<()> {
    // Step 1: Choose Algorithm
    let (popsize, num_dims) = (5, 4);
    let open_es = Algo::OpenES(popsize, num_dims);
    // dbg!(&open_es);

    // Step 2: Get its (default) Parmeters and...
    let params = open_es.default_params();
    // dbg!(&params);

    // Step 3: Initiate its State
    let state = open_es.init_algorithm(&params);
    // println!("{:+6.4?}", &state);

    // Step 4: Ask-Tell
    let pop: Array2<f32> = open_es.ask(&state);
    // println!("{:+.4}", &pop);

    let fitness: Array2<f32> = square_and_sum(&pop);
    // println!("{:+.4}", &fitness);

    let _ = open_es.tell(pop, fitness, state, &params);
    // dbg!(&state);

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

#[cfg(test)]
mod tests {
    use crate::work;

    #[test]
    // TODO: implement integration tests, similar to Robert Lange
    fn it_works() {
        _ = work();
    }
}

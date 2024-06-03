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
/// # const IGNORE: &str = stringify! {
/// fn demo1() -> Result<T> {...}
///            // ^ equivalent to std::result::Result<T, anyhow::Error>
///
/// fn demo2() -> Result<T, OtherError> {...}
///            // ^ equivalent to std::result::Result<T, OtherError>
/// # };
/// ```
///
/// # Example
///
/// ```
/// # pub trait Deserialize {}
/// #
/// # mod serde_json {
/// #     use super::Deserialize;
/// #     use std::io;
/// #
/// #     pub fn from_str<T: Deserialize>(json: &str) -> io::Result<T> {
/// #         unimplemented!()
/// #     }
/// # }
/// #
/// # #[derive(Debug)]
/// # struct ClusterMap;
/// #
/// # impl Deserialize for ClusterMap {}
/// #
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
    // This line should in doc as a comment...?
    let popsize = 5;
    let num_dims = 4;
    let open_es = Algo::OpenES(popsize, num_dims);
    // dbg!(&open_es);

    let params = open_es.default_params();
    // dbg!(&params);

    let state = open_es.init_algorithm(&params);
    // println!("{:+6.4?}", &state);

    let pop: Array2<f32> = open_es.ask(&state);
    println!("{:+.4}", &pop);

    let fitness: Array2<f32> = square_and_sum(&pop);
    println!("{:+.4}", &fitness);

    // let state = open_es.tell(state, pop, fitness, &params);
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

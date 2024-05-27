use anyhow::Result;
use ndarray::Array2;

mod fitness;
use fitness::square_and_sum;

mod params;

mod states;

mod strategies;
use strategies::Algo;


pub fn work() -> Result<()> {
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

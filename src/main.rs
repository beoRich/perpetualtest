use clap::Parser;
use perpetual::Matrix;
use std::error::Error;
#[derive(Parser)]
struct Cli {
    budget: f32,
}

fn main() -> Result<(), Box<dyn Error>> {
    //let budget = Cli::parse().budget;

    let (data, y) = perpetualtest::preprocess("resources/titanic.csv")?;

    let matrix = Matrix::new(&data, y.len(), 5);

    let budgets = vec![0.5, 0.8, 1.0];

    let (results, errors): (Vec<_>, Vec<_>) = budgets
        .into_iter()
        .map(|budget| perpetualtest::train_and_predict(&y, &matrix, budget))
        .partition(Result::is_ok);

    let results: Vec<_> = results.into_iter().map(Result::unwrap).collect();
    let errors: Vec<_> = errors.into_iter().map(Result::unwrap_err).collect();

    results.into_iter().for_each(|train_predict_result| {
        perpetualtest::calculate_results(
            train_predict_result.proba,
            &y,
            train_predict_result.budget,
        )
    });
    if !errors.is_empty() {
        eprintln!("Errors:");
        for error in errors {
            eprintln!("{}", error);
        }
    }
    Ok(())
}

use clap::Parser;
use perpetual::Matrix;
use std::error::Error;
#[derive(Parser)]
struct Cli {
    budget: f32,
}

mod config;
mod postprocessing;
mod prediction;
mod preprocessing;
mod training;

fn main() -> Result<(), Box<dyn Error>> {
    //let budget = Cli::parse().budget;

    let (data, y) = preprocessing::read_preprocess("resources/titanic.csv")?;

    let matrix = Matrix::new(&data, y.len(), 5);

    let budgets = vec![0.5, 0.8, 1.0];

    let (results, errors): (Vec<_>, Vec<_>) = budgets
        .into_iter()
        .map(|budget| training::train(&y, &matrix, budget))
        .partition(Result::is_ok);

    let train_result: Vec<_> = results.into_iter().map(Result::unwrap).collect();
    let prediction_results = train_result
        .into_iter()
        .map(|train_result| prediction::predict(train_result, &matrix));

    prediction_results
        .into_iter()
        .for_each(|prediction_result| {
            postprocessing::calculate_results(
                prediction_result.predicted_values,
                &y,
                prediction_result.train_result.budget,
            )
        });
    if !errors.is_empty() {
        eprintln!("Errors:");
        for error in errors {
            match error {
                Ok(_) => println!("Operation successful!"),
                Err(e) => {
                    // Print the error. Box<dyn Error> implements Debug automatically.
                    println!("Error occurred: {}", e);
                }
            }
        }
    }
    Ok(())
}

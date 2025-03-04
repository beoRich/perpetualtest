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

    let errors: Vec<Box<dyn Error>> = budgets
        .into_iter()
        .map(|budget| perpetualtest::train_and_predict(&y, &matrix, budget))
        .filter_map(Result::err)
        .collect();

    if errors.is_empty() {
        Ok(())
    } else {
        // Combine errors into a single message
        let error_messages = errors
            .into_iter()
            .map(|e| e.to_string())
            .collect::<Vec<String>>()
            .join("; ");

        Err(error_messages.into()) // Convert to Box<dyn Error>
    }
}

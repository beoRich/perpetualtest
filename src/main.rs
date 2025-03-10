use clap::Parser;
use perpetual::Matrix;
use std::error::Error;

mod config;
mod parsing;
mod postprocessing;
mod prediction;
mod preprocessing;
mod training;

fn main() -> Result<(), Box<dyn Error>> {
    let cli = parsing::Cli::parse();
    let mode = cli.mode;
    let budget = cli.budget;

    let (data, y) = preprocessing::read_preprocess(config::TRAINING_DATA)?;
    let matrix = Matrix::new(&data, y.len(), 5);

    match mode {
        parsing::Mode::Train => {
            println!("Train Only:");
            let _ = training::train(&y, &matrix, budget);
        }
        parsing::Mode::Predict => {
            println!("Predict Only:");
            pred_post_process(&y, &matrix, None);
        }
        parsing::Mode::TrainPredict => {
            println!("Predict and train:");
            let train_result = training::train(&y, &matrix, budget);
            pred_post_process(&y, &matrix, Some(train_result?));
        }
    }

    Ok(())
}

fn pred_post_process(
    y: &Vec<f64>,
    matrix: &Matrix<f64>,
    train_maybe: Option<training::TrainResult>,
) {
    let prediction_result = prediction::predict(train_maybe, &matrix);
    postprocessing::calculate_results(
        prediction_result.predicted_values,
        &y,
        prediction_result.model.budget,
    )
}

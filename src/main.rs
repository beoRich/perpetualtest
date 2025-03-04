use clap::Parser;
use perpetual::objective::Objective;
use perpetual::{Matrix, PerpetualBooster};
use polars::prelude::*;
use std::sync::Arc;

#[derive(Parser)]
struct Cli {
    budget: f32,
}

fn proba_to_indicator(prob: f64) -> u8 {
    if prob >= 0.5 { 1 } else { 0 }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    //let budget = Cli::parse().budget;

    let features_and_target = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"];

    let features_and_target_arc: Arc<[PlSmallStr]> = features_and_target
        .iter()
        .map(|&s| PlSmallStr::from(s)) // Convert &str to PlSmallStr
        .collect::<Vec<PlSmallStr>>() // Collect into Vec
        .into_boxed_slice() // Convert Vec<PlSmallStr> to Box<[PlSmallStr]>
        .into(); // Convert Box<[PlSmallStr]> to Arc<[PlSmallStr]>

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_columns(Some(features_and_target_arc))
        .try_into_reader_with_file_path(Some("resources/titanic.csv".into()))?
        .finish()
        .unwrap();

    //println!("{}", df.head(Some(5)));

    // Get data in column major format...
    let id_vars: Vec<&str> = Vec::new();
    let mdf = df.unpivot(["Pclass", "Age", "SibSp", "Parch", "Fare"], id_vars)?;

    let data = Vec::from_iter(
        mdf.select_at_idx(1)
            .expect("Invalid column")
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN)),
    );

    let y = Vec::from_iter(
        df.column("Survived")?
            .cast(&DataType::Float64)?
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN)),
    );

    // Create Matrix from ndarray.
    let matrix = Matrix::new(&data, y.len(), 5);

    let budget = 0.8;

    // Create booster.
    // To provide parameters generate a default booster, and then use
    // the relevant `set_` methods for any parameters you would like to
    // adjust.
    let mut model = PerpetualBooster::default()
        .set_objective(Objective::LogLoss)
        .set_budget(budget);
    model.fit(&matrix, &y, None)?;

    //println!( "Model prediction(log loss): {:?} ...", &model.predict(&matrix, true)[0..10] );

    let proba_results = &model.predict_proba(&matrix, true);
    let indicator: Vec<_> = proba_results
        .iter()
        .map(|&p| proba_to_indicator(p))
        .collect();

    let compare = y.into_iter().zip(indicator.into_iter());

    let diff_vec = compare
        .into_iter()
        .map(|(x, y)| (x - y as f64).abs())
        .collect::<Vec<_>>();

    let amount: u32 = diff_vec.len() as u32;
    let diff_sum: u32 = diff_vec.into_iter().map(|x| x as u32).sum();
    println!(
        "Budget: {}, Score: {}/{}",
        budget,
        amount - diff_sum,
        amount
    );

    Ok(())
}

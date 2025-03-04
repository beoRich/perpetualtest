use perpetual::objective::Objective;
use perpetual::{Matrix, PerpetualBooster};
use polars::datatypes::{DataType, PlSmallStr};
use polars::io::SerReader;
use polars::prelude::{CsvReadOptions, UnpivotDF};
use std::error::Error;
use std::sync::Arc;

#[derive(Debug)]
pub struct TrainPredictResults {
    pub proba: Vec<f64>,
    pub budget: f32,
}

fn proba_to_indicator(prob: f64) -> u8 {
    if prob >= 0.5 { 1 } else { 0 }
}

pub fn preprocess(file_path: &str) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
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
        .try_into_reader_with_file_path(Some(file_path.into()))?
        .finish()
        .unwrap();

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

    // model fit requires float64 type later
    let y = Vec::from_iter(
        df.column("Survived")?
            .cast(&DataType::Float64)?
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN)),
    );
    Ok((data, y))
}
pub fn train_and_predict(
    y: &Vec<f64>,
    matrix: &Matrix<f64>,
    budget: f32,
) -> Result<TrainPredictResults, Box<dyn Error>> {
    // Create booster.
    // To provide parameters generate a default booster, and then use
    // the relevant `set_` methods for any parameters you would like to
    // adjust.
    let mut model = PerpetualBooster::default()
        .set_objective(Objective::LogLoss)
        .set_budget(budget);
    model.fit(&matrix, &y, None)?;

    let proba_results = model.predict_proba(&matrix, true);
    Ok(TrainPredictResults {
        proba: proba_results,
        budget,
    })
}

pub fn calculate_results(proba_results: Vec<f64>, y: &Vec<f64>, budget: f32) {
    let indicator: Vec<_> = proba_results
        .iter()
        .map(|&p| crate::proba_to_indicator(p))
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
}

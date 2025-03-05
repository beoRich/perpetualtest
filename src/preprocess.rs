use polars::datatypes::{DataType, PlSmallStr};
use polars::frame::DataFrame;
use polars::prelude::{CsvReadOptions, SerReader, UnpivotDF};
use std::error::Error;
use std::sync::Arc;

pub fn preprocess(file_path: &str) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let df = read_csv(file_path)?;

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

fn read_csv(file_path: &str) -> Result<DataFrame, Box<dyn Error>> {
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
    Ok(df)
}

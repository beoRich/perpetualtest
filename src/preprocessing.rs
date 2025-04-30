use crate::config;
use polars::datatypes::{DataType, PlSmallStr};
use polars::frame::DataFrame;
use polars::prelude::{ChunkApply, CsvReadOptions, SerReader, UnpivotDF};
use std::borrow::Cow;
use std::error::Error;
use std::sync::Arc;

use plotlars::{Plot, Rgb, ScatterPlot};

pub fn read_preprocess(file_path: &str) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let features_and_target_arc: Arc<[PlSmallStr]> = config::FEATURES_AND_TARGET
        .iter()
        .map(|&s| PlSmallStr::from(s)) // Convert &str to PlSmallStr
        .collect::<Vec<PlSmallStr>>() // Collect into Vec
        .into_boxed_slice() // Convert Vec<PlSmallStr> to Box<[PlSmallStr]>
        .into(); // Convert Box<[PlSmallStr]> to Arc<[PlSmallStr]>

    let mut df = CsvReadOptions::default()
        .with_has_header(true)
        .with_columns(Some(features_and_target_arc))
        .try_into_reader_with_file_path(Some(file_path.into()))?
        .finish()
        .unwrap();
    df.try_apply("Sex", |s| {
        Ok(s.str()?.apply_values(|value| match value {
            "male" => Cow::from("0"),
            "female" => Cow::from("1"),
            _ => Cow::from("2"),
        }))
    })?;
    let mut df_clone = df.clone();
    df.try_apply("Sex", |s| s.cast(&DataType::Float64))?;
    df_clone.try_apply("Survived", |s| s.cast(&DataType::String))?;
    ScatterPlot::builder()
        .data(&df_clone)
        .x("Sex")
        .y("Age")
        .group("Survived")
        .opacity(0.5)
        .size(12)
        .colors(vec![Rgb(178, 34, 34), Rgb(65, 105, 225), Rgb(255, 140, 0)])
        .plot_title("Titanic Passengers Sex vs Age")
        .x_title("Sex")
        .y_title("Age")
        .legend_title("Survived")
        .build()
        .plot();
    println!("{}", df.head(Some(20)));
    preprocess(df)
}

pub fn preprocess(df: DataFrame) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    // Get data in column major format...
    let id_vars: Vec<&str> = Vec::new();
    println!("Before unpivot{}", df);
    let mut mdf = df.unpivot(config::FEATURES, id_vars)?;
    mdf.try_apply("value", |s| s.cast(&DataType::Float64))?;
    println!("After unpivot {}", mdf);

    let data = Vec::from_iter(
        mdf.select_at_idx(1)
            .expect("Invalid column")
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN)),
    );

    // model.fit requires float64 type later
    let y = Vec::from_iter(
        df.column("Survived")?
            .cast(&DataType::Float64)?
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN)),
    );
    Ok((data, y))
}

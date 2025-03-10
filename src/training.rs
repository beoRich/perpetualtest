use crate::config;
use perpetual::booster::booster::ImportanceMethod;
use perpetual::objective::Objective;
use perpetual::{Matrix, PerpetualBooster};
use std::collections::HashMap;
use std::error::Error;

pub struct TrainResult {
    pub model: PerpetualBooster,
    pub feature_importance: HashMap<usize, f32>,
    pub budget: f32,
}

pub fn train(
    y: &Vec<f64>,
    matrix: &Matrix<f64>,
    budget: f32,
) -> Result<TrainResult, Box<dyn Error>> {
    // Create booster.
    // To provide parameters generate a default booster, and then use
    // the relevant `set_` methods for any parameters you would like to
    // adjust.
    let mut model = PerpetualBooster::default()
        .set_objective(Objective::LogLoss)
        .set_budget(budget);
    model.fit(&matrix, &y, None)?;
    model.save_booster(config::MODEL_PATH)?;
    // feature importance returns a raw map
    let feature_importance = model.calculate_feature_importance(ImportanceMethod::TotalCover, true);
    Ok(TrainResult {
        model,
        budget,
        feature_importance,
    })
}

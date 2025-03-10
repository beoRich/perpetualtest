use crate::config;
use crate::training::TrainResult;
use perpetual::{Matrix, PerpetualBooster};

pub struct PredictionResult {
    pub model: PerpetualBooster,
    pub predicted_values: Vec<f64>,
}
pub fn predict(train_result_input: Option<TrainResult>, matrix: &Matrix<f64>) -> PredictionResult {
    let model = match train_result_input {
        Some(train_result) => train_result.model,
        None => read_model(),
    };
    let predicted_values = model.predict_proba(&matrix, true);
    PredictionResult {
        model,
        predicted_values,
    }
}

fn read_model() -> PerpetualBooster {
    PerpetualBooster::load_booster(config::MODEL_PATH).unwrap()
}

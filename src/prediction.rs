use crate::training::TrainResult;
use perpetual::Matrix;

pub struct PredictionResult {
    pub train_result: TrainResult,
    pub predicted_values: Vec<f64>,
}
pub fn predict(train_result: TrainResult, matrix: &Matrix<f64>) -> PredictionResult {
    let predicted_values = train_result.model.predict_proba(&matrix, true);
    PredictionResult {
        train_result,
        predicted_values,
    }
}

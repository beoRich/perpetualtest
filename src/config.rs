pub const TRAINING_DATA: &str = "resources/data/titanic.csv";
pub const MODEL_PATH: &str = "resources/model/latest.pb";
pub const FEATURES: [&str; 6] = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex"];
pub const FEATURES_AND_TARGET: [&str; 7] =
    ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "Sex"];

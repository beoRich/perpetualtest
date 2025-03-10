use clap::Parser;

#[derive(Parser)]
pub struct Cli {
    #[clap(value_enum, default_value_t = Mode::TrainPredict)]
    pub mode: Mode,
    #[clap(long, default_value_t = 1.0)]
    pub budget: f32,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum Mode {
    Train,
    Predict,
    TrainPredict,
}

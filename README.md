# Perpetualtest Demo

## About

This is a demo cli project for exploring perpetual (https://github.com/perpetual-ml/perpetual), and a blueprint of how
to structure a classification ml project in rust.
It trains a classification model on the titanic dataset and predicts the survival.

## Requirements

Rust nightly
`rustup install nightly
cargo +nightly install racer`

## Usage

Budget Argument defaults to 1.0 if omitted

### Train Only with budget override

`cargo run -- train --budget 1.3`

### Predict Only (uses the latest trained model)

`cargo run -- predict`

### Train and Predict (with default budget)

`cargo run`
or
`cargo run train-predict`

### Train and Predict with budget override

`cargo run -- --budget 1.3`
or
`cargo run train-predict --budget 1.4`


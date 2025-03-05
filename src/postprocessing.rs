use crate::postprocessing;

pub fn calculate_results(proba_results: Vec<f64>, y: &Vec<f64>, budget: f32) {
    let indicator: Vec<_> = proba_results
        .iter()
        .map(|&p| postprocessing::proba_to_indicator(p))
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

fn proba_to_indicator(prob: f64) -> u8 {
    if prob >= 0.5 { 1 } else { 0 }
}

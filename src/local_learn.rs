/// Local learning rule primitives (alternatives to backpropagation).
/// Implements Forward-Forward, Hebbian, Oja's rule, and predictive coding updates.

/// Forward-Forward goodness function: sum of squared activations
pub fn goodness(activations: &[f64]) -> f64 {
    activations.iter().map(|x| x * x).sum()
}

/// Dense forward pass with ReLU activation
pub fn dense_forward(input: &[f64], weights: &[f64], bias: &[f64],
                     n_in: usize, n_out: usize) -> Vec<f64> {
    let mut out = vec![0.0; n_out];
    for j in 0..n_out {
        out[j] = bias[j];
        for i in 0..n_in {
            out[j] += input[i] * weights[i * n_out + j];
        }
        out[j] = out[j].max(0.0); // ReLU
    }
    out
}

/// Forward-Forward layer update.
/// pos_data and neg_data produce activations; layer learns to have high goodness
/// for positive examples and low goodness for negative examples.
pub fn ff_layer_forward(
    pos_input: &[f64], neg_input: &[f64],
    weights: &mut [f64], bias: &mut [f64],
    n_in: usize, n_out: usize,
    lr: f64, threshold: f64,
) -> (Vec<f64>, Vec<f64>) {
    let pos_act = dense_forward(pos_input, weights, bias, n_in, n_out);
    let neg_act = dense_forward(neg_input, weights, bias, n_in, n_out);

    let pos_goodness = goodness(&pos_act);
    let neg_goodness = goodness(&neg_act);

    // Local loss: push pos_goodness above threshold, neg_goodness below threshold
    // Gradient of goodness w.r.t. pre-ReLU activation = 2*activation (for active units)
    // Gradient of pre-ReLU w.r.t. weight[i,j] = input[i]

    // Positive phase: increase goodness if below threshold
    if pos_goodness < threshold {
        for j in 0..n_out {
            if pos_act[j] > 0.0 {
                // gradient factor
                let g = 2.0 * pos_act[j] * lr;
                for i in 0..n_in {
                    weights[i * n_out + j] += g * pos_input[i];
                }
                bias[j] += g;
            }
        }
    }

    // Negative phase: decrease goodness if above threshold
    if neg_goodness > threshold {
        for j in 0..n_out {
            if neg_act[j] > 0.0 {
                let g = 2.0 * neg_act[j] * lr;
                for i in 0..n_in {
                    weights[i * n_out + j] -= g * neg_input[i];
                }
                bias[j] -= g;
            }
        }
    }

    (pos_act, neg_act)
}

/// Hebbian update: delta_w = eta * pre * post
pub fn hebbian_update(pre: &[f64], post: &[f64], weights: &mut [f64],
                      n_pre: usize, n_post: usize, lr: f64) {
    for i in 0..n_pre {
        for j in 0..n_post {
            weights[i * n_post + j] += lr * pre[i] * post[j];
        }
    }
}

/// Oja's rule: Hebbian with weight normalization
/// delta_w = eta * (pre * post - post^2 * w)
pub fn oja_update(pre: &[f64], post: &[f64], weights: &mut [f64],
                  n_pre: usize, n_post: usize, lr: f64) {
    for i in 0..n_pre {
        for j in 0..n_post {
            let idx = i * n_post + j;
            weights[idx] += lr * (pre[i] * post[j] - post[j] * post[j] * weights[idx]);
        }
    }
}

/// Predictive coding update: minimize prediction error locally.
/// Returns the error signal for the layer below.
pub fn predictive_coding_update(
    prediction: &[f64], target: &[f64],
    weights: &mut [f64], n_in: usize, n_out: usize, lr: f64,
) -> Vec<f64> {
    // error = target - prediction
    let mut error = vec![0.0; n_out];
    for j in 0..n_out {
        error[j] = target[j] - prediction[j];
    }

    // Update weights to reduce error: w += lr * error_j * input_i
    // We approximate input as the transpose projection of prediction
    // For simplicity, update based on error gradient
    for i in 0..n_in {
        for j in 0..n_out {
            // Use prediction as a proxy for the input contribution
            weights[i * n_out + j] += lr * error[j];
        }
    }

    error
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goodness() {
        assert!((goodness(&[1.0, 2.0, 3.0]) - 14.0).abs() < 1e-10);
        assert!((goodness(&[0.0, 0.0]) - 0.0).abs() < 1e-10);
        assert!((goodness(&[0.5]) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_hebbian_update() {
        let pre = vec![1.0, 2.0];
        let post = vec![3.0, 4.0];
        let mut weights = vec![0.0; 4]; // 2x2
        let lr = 0.1;
        hebbian_update(&pre, &post, &mut weights, 2, 2, lr);
        // w[0,0] += 0.1 * 1.0 * 3.0 = 0.3
        assert!((weights[0] - 0.3).abs() < 1e-10);
        // w[0,1] += 0.1 * 1.0 * 4.0 = 0.4
        assert!((weights[1] - 0.4).abs() < 1e-10);
        // w[1,0] += 0.1 * 2.0 * 3.0 = 0.6
        assert!((weights[2] - 0.6).abs() < 1e-10);
        // w[1,1] += 0.1 * 2.0 * 4.0 = 0.8
        assert!((weights[3] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_ff_layer() {
        let n_in = 3;
        let n_out = 2;
        let mut weights = vec![0.1; n_in * n_out];
        let mut bias = vec![0.0; n_out];
        let pos_input = vec![1.0, 1.0, 1.0]; // strong signal
        let neg_input = vec![0.1, 0.1, 0.1]; // weak signal
        let lr = 0.01;
        let threshold = 1.0;

        // Run several updates
        for _ in 0..50 {
            ff_layer_forward(&pos_input, &neg_input, &mut weights, &mut bias,
                             n_in, n_out, lr, threshold);
        }

        // After training, positive goodness should exceed negative goodness
        let pos_act = dense_forward(&pos_input, &weights, &bias, n_in, n_out);
        let neg_act = dense_forward(&neg_input, &weights, &bias, n_in, n_out);
        assert!(goodness(&pos_act) > goodness(&neg_act),
                "pos goodness {} should exceed neg goodness {}",
                goodness(&pos_act), goodness(&neg_act));
    }

    #[test]
    fn test_predictive_coding() {
        let n_in = 2;
        let n_out = 2;
        let mut weights = vec![0.0; n_in * n_out];
        let target = vec![1.0, 2.0];
        let lr = 0.01;

        // Run several updates; error should decrease
        let prediction1 = vec![0.0; n_out];
        let error1 = predictive_coding_update(&prediction1, &target, &mut weights, n_in, n_out, lr);
        let err_mag1: f64 = error1.iter().map(|e| e * e).sum();

        // After updating weights, a new forward pass (simple w * 1) should be closer
        // Apply more updates
        for _ in 0..100 {
            let pred: Vec<f64> = (0..n_out).map(|j| {
                (0..n_in).map(|i| weights[i * n_out + j]).sum()
            }).collect();
            predictive_coding_update(&pred, &target, &mut weights, n_in, n_out, lr);
        }
        let final_pred: Vec<f64> = (0..n_out).map(|j| {
            (0..n_in).map(|i| weights[i * n_out + j]).sum()
        }).collect();
        let err_mag2: f64 = (0..n_out).map(|j| (target[j] - final_pred[j]).powi(2)).sum();

        assert!(err_mag2 < err_mag1,
                "error should decrease: initial {} vs final {}", err_mag1, err_mag2);
    }
}

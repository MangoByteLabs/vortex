/// ODE solver primitives for neural ODEs, liquid neural networks, and CfC cells.

/// Euler method (1st order)
pub fn euler_solve(f: &dyn Fn(f64, &[f64]) -> Vec<f64>, y0: &[f64],
                   t_start: f64, t_end: f64, steps: usize) -> Vec<f64> {
    let dt = (t_end - t_start) / steps as f64;
    let mut y = y0.to_vec();
    let mut t = t_start;
    for _ in 0..steps {
        let dy = f(t, &y);
        for i in 0..y.len() {
            y[i] += dt * dy[i];
        }
        t += dt;
    }
    y
}

/// RK4 solver (4th order Runge-Kutta)
pub fn rk4_solve(f: &dyn Fn(f64, &[f64]) -> Vec<f64>, y0: &[f64],
                 t_start: f64, t_end: f64, steps: usize) -> Vec<f64> {
    let dt = (t_end - t_start) / steps as f64;
    let mut y = y0.to_vec();
    let mut t = t_start;
    let n = y.len();
    for _ in 0..steps {
        let k1 = f(t, &y);
        let y2: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * dt * k1[i]).collect();
        let k2 = f(t + 0.5 * dt, &y2);
        let y3: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * dt * k2[i]).collect();
        let k3 = f(t + 0.5 * dt, &y3);
        let y4: Vec<f64> = (0..n).map(|i| y[i] + dt * k3[i]).collect();
        let k4 = f(t + dt, &y4);
        for i in 0..n {
            y[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += dt;
    }
    y
}

/// Adaptive RK45 (Dormand-Prince) with error control
pub fn rk45_solve(f: &dyn Fn(f64, &[f64]) -> Vec<f64>, y0: &[f64],
                  t_start: f64, t_end: f64, rtol: f64, atol: f64) -> Vec<f64> {
    // Dormand-Prince coefficients
    let a2 = 1.0 / 5.0;
    let a3 = 3.0 / 10.0;
    let a4 = 4.0 / 5.0;
    let a5 = 8.0 / 9.0;

    let b21 = 1.0 / 5.0;
    let b31 = 3.0 / 40.0; let b32 = 9.0 / 40.0;
    let b41 = 44.0 / 45.0; let b42 = -56.0 / 15.0; let b43 = 32.0 / 9.0;
    let b51 = 19372.0 / 6561.0; let b52 = -25360.0 / 2187.0; let b53 = 64448.0 / 6561.0; let b54 = -212.0 / 729.0;
    let b61 = 9017.0 / 3168.0; let b62 = -355.0 / 33.0; let b63 = 46732.0 / 5247.0; let b64 = 49.0 / 176.0; let b65 = -5103.0 / 18656.0;

    // 5th order weights
    let c1 = 35.0 / 384.0; let c3 = 500.0 / 1113.0; let c4 = 125.0 / 192.0; let c5 = -2187.0 / 6784.0; let c6 = 11.0 / 84.0;
    // 4th order weights (for error estimate)
    let d1 = 5179.0 / 57600.0; let d3 = 7571.0 / 16695.0; let d4 = 393.0 / 640.0; let d5 = -92097.0 / 339200.0; let d6 = 187.0 / 2100.0; let d7 = 1.0 / 40.0;

    let n = y0.len();
    let mut y = y0.to_vec();
    let mut t = t_start;
    let mut dt = (t_end - t_start) / 100.0; // initial step guess
    let dt_min = (t_end - t_start) * 1e-12;

    while t < t_end - dt_min {
        if t + dt > t_end { dt = t_end - t; }

        let k1 = f(t, &y);
        let y2: Vec<f64> = (0..n).map(|i| y[i] + dt * b21 * k1[i]).collect();
        let k2 = f(t + a2 * dt, &y2);
        let y3: Vec<f64> = (0..n).map(|i| y[i] + dt * (b31 * k1[i] + b32 * k2[i])).collect();
        let k3 = f(t + a3 * dt, &y3);
        let y4: Vec<f64> = (0..n).map(|i| y[i] + dt * (b41 * k1[i] + b42 * k2[i] + b43 * k3[i])).collect();
        let k4 = f(t + a4 * dt, &y4);
        let y5: Vec<f64> = (0..n).map(|i| y[i] + dt * (b51 * k1[i] + b52 * k2[i] + b53 * k3[i] + b54 * k4[i])).collect();
        let k5 = f(t + a5 * dt, &y5);
        let y6: Vec<f64> = (0..n).map(|i| y[i] + dt * (b61 * k1[i] + b62 * k2[i] + b63 * k3[i] + b64 * k4[i] + b65 * k5[i])).collect();
        let k6 = f(t + dt, &y6);

        // 5th order solution
        let y5th: Vec<f64> = (0..n).map(|i| {
            y[i] + dt * (c1 * k1[i] + c3 * k3[i] + c4 * k4[i] + c5 * k5[i] + c6 * k6[i])
        }).collect();

        // 4th order solution for error estimate
        let y7: Vec<f64> = (0..n).map(|i| {
            y[i] + dt * (c1 * k1[i] + c3 * k3[i] + c4 * k4[i] + c5 * k5[i] + c6 * k6[i])
        }).collect();
        let k7 = f(t + dt, &y7);
        let y4th: Vec<f64> = (0..n).map(|i| {
            y[i] + dt * (d1 * k1[i] + d3 * k3[i] + d4 * k4[i] + d5 * k5[i] + d6 * k6[i] + d7 * k7[i])
        }).collect();

        // Error estimate
        let err: f64 = (0..n).map(|i| {
            let sc = atol + rtol * y5th[i].abs().max(y[i].abs());
            ((y5th[i] - y4th[i]) / sc).powi(2)
        }).sum::<f64>() / n as f64;
        let err = err.sqrt();

        if err <= 1.0 || dt <= dt_min {
            // Accept step
            t += dt;
            y = y5th;
            // Increase step size
            if err > 0.0 {
                dt *= (0.9 * (1.0 / err).powf(0.2)).min(5.0);
            } else {
                dt *= 5.0;
            }
        } else {
            // Reject step, reduce dt
            dt *= (0.9 * (1.0 / err).powf(0.25)).max(0.1);
        }
    }
    y
}

/// Liquid neural network cell: dh/dt = (-h + tanh(W_h @ h + W_x @ x)) / tau
pub fn liquid_cell(h: &[f64], x: &[f64], w_h: &[f64], w_x: &[f64],
                   tau: &[f64], n_hidden: usize, n_input: usize) -> Vec<f64> {
    let mut dh = vec![0.0; n_hidden];
    for j in 0..n_hidden {
        let mut activation = 0.0;
        for i in 0..n_hidden {
            activation += w_h[i * n_hidden + j] * h[i];
        }
        for i in 0..n_input {
            activation += w_x[i * n_hidden + j] * x[i];
        }
        dh[j] = (-h[j] + activation.tanh()) / tau[j];
    }
    dh
}

/// Closed-form continuous-time (CfC) approximation.
/// Analytical solution avoiding ODE solver:
/// h_new = h * exp(-dt/tau) + (1 - exp(-dt/tau)) * tanh(W_h @ h + W_x @ x)
pub fn cfc_cell(h: &[f64], x: &[f64], w_h: &[f64], w_x: &[f64],
                tau: &[f64], dt: f64, n_hidden: usize, n_input: usize) -> Vec<f64> {
    let mut h_new = vec![0.0; n_hidden];
    for j in 0..n_hidden {
        let decay = (-dt / tau[j]).exp();
        let mut activation = 0.0;
        for i in 0..n_hidden {
            activation += w_h[i * n_hidden + j] * h[i];
        }
        for i in 0..n_input {
            activation += w_x[i * n_hidden + j] * x[i];
        }
        h_new[j] = h[j] * decay + (1.0 - decay) * activation.tanh();
    }
    h_new
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_exponential_decay() {
        // dy/dt = -y, y(0) = 1 => y(1) = e^{-1} ≈ 0.3679
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let result = euler_solve(&f, &[1.0], 0.0, 1.0, 10000);
        let expected = (-1.0_f64).exp();
        assert!((result[0] - expected).abs() < 0.001,
                "Euler result {} vs expected {}", result[0], expected);
    }

    #[test]
    fn test_rk4_accuracy() {
        // Same problem: dy/dt = -y, y(0) = 1
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let steps = 100;
        let euler = euler_solve(&f, &[1.0], 0.0, 1.0, steps);
        let rk4 = rk4_solve(&f, &[1.0], 0.0, 1.0, steps);
        let expected = (-1.0_f64).exp();

        let euler_err = (euler[0] - expected).abs();
        let rk4_err = (rk4[0] - expected).abs();
        assert!(rk4_err < euler_err,
                "RK4 error {} should be less than Euler error {}", rk4_err, euler_err);
        assert!(rk4_err < 1e-8, "RK4 should be very accurate, got error {}", rk4_err);
    }

    #[test]
    fn test_liquid_cell() {
        let n_hidden = 2;
        let n_input = 1;
        let h = vec![0.5, -0.3];
        let x = vec![1.0];
        let w_h = vec![0.1, 0.2, -0.1, 0.3]; // 2x2
        let w_x = vec![0.5, -0.4]; // 1x2
        let tau = vec![1.0, 1.0];

        let dh = liquid_cell(&h, &x, &w_h, &w_x, &tau, n_hidden, n_input);
        assert_eq!(dh.len(), n_hidden);

        // dh[0] = (-h[0] + tanh(w_h[0,0]*h[0] + w_h[1,0]*h[1] + w_x[0,0]*x[0])) / tau[0]
        let act0: f64 = 0.1 * 0.5 + (-0.1) * (-0.3) + 0.5 * 1.0; // 0.05 + 0.03 + 0.5 = 0.58
        let expected_dh0 = (-0.5 + act0.tanh()) / 1.0;
        assert!((dh[0] - expected_dh0).abs() < 1e-10,
                "dh[0] = {} vs expected {}", dh[0], expected_dh0);
    }

    #[test]
    fn test_cfc_cell() {
        // CfC should approximate Euler solution of liquid cell ODE
        let n_hidden = 2;
        let n_input = 1;
        let h = vec![0.0, 0.0];
        let x = vec![1.0];
        let w_h = vec![0.0, 0.0, 0.0, 0.0]; // no recurrence for simpler test
        let w_x = vec![0.5, -0.3];
        let tau = vec![1.0, 1.0];
        let dt = 0.01;
        let steps = 100; // total time = 1.0

        // Euler integration of liquid cell
        let x2 = x.clone();
        let w_h2 = w_h.clone();
        let w_x2 = w_x.clone();
        let tau2 = tau.clone();
        let f = move |_t: f64, h_state: &[f64]| -> Vec<f64> {
            liquid_cell(h_state, &x2, &w_h2, &w_x2, &tau2, n_hidden, n_input)
        };
        let euler_result = euler_solve(&f, &h, 0.0, 1.0, steps);

        // CfC in one big step (dt=1.0)
        let cfc_result = cfc_cell(&h, &x, &w_h, &w_x, &tau, 1.0, n_hidden, n_input);

        // Both should converge toward tanh(W_x @ x) for zero initial state and no recurrence
        // They won't match exactly but should be in the same ballpark
        for j in 0..n_hidden {
            assert!((euler_result[j] - cfc_result[j]).abs() < 0.5,
                    "dim {}: Euler {} vs CfC {} differ too much", j, euler_result[j], cfc_result[j]);
        }
    }
}

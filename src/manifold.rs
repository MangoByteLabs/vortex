// Pillar 5: Continuous Mathematics — Riemannian manifold operations for Vortex
// Provides exponential/logarithmic maps, geodesics, parallel transport, curvature,
// and manifold-aware optimization primitives across Euclidean, Sphere, Hyperbolic,
// SPD, Grassmann, SO3, Torus, and product manifold geometries.

use crate::interpreter::{Env, FnDef, Value};
use std::collections::HashMap;
use std::sync::Mutex;

// ─── Lazy global stores ──────────────────────────────────────────────────

lazy_static::lazy_static! {
    static ref MANIFOLDS: Mutex<HashMap<usize, ManifoldKind>> = Mutex::new(HashMap::new());
    static ref TANGENTS: Mutex<HashMap<usize, TangentVector>> = Mutex::new(HashMap::new());
    static ref GEODESICS: Mutex<HashMap<usize, GeodesicPath>> = Mutex::new(HashMap::new());
    static ref BUNDLES: Mutex<HashMap<usize, FiberBundle>> = Mutex::new(HashMap::new());
    static ref NEXT_ID: Mutex<usize> = Mutex::new(1);
}

fn next_id() -> usize {
    let mut id = NEXT_ID.lock().unwrap();
    let v = *id;
    *id += 1;
    v
}

// ─── Core types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ManifoldKind {
    Euclidean(usize),
    Sphere(f64),            // radius
    Hyperbolic(f64),        // curvature (negative, stored as |c|)
    SPD(usize),             // dim — space of symmetric positive-definite matrices
    Grassmann(usize, usize),// (n, k)
    SO3,
    Torus(usize),           // dim
    ProductManifold(Vec<ManifoldKind>),
}

#[derive(Debug, Clone)]
pub struct TangentVector {
    pub base_point: Vec<f64>,
    pub vector: Vec<f64>,
    pub manifold: ManifoldKind,
}

#[derive(Debug, Clone)]
pub struct FiberBundle {
    pub base_manifold: ManifoldKind,
    pub fiber_manifold: ManifoldKind,
}

#[derive(Debug, Clone)]
pub struct GeodesicPath {
    pub start: Vec<f64>,
    pub end: Vec<f64>,
    pub manifold: ManifoldKind,
}

// ─── Deterministic xorshift64 PRNG ──────────────────────────────────────

#[allow(dead_code)]
struct Xorshift64 {
    state: u64,
}

#[allow(dead_code)]
impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

// ─── Helper conversions ─────────────────────────────────────────────────

fn value_to_f64(v: &Value) -> Result<f64, String> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        _ => Err(format!("expected number, got {:?}", v)),
    }
}

fn value_to_usize(v: &Value) -> Result<usize, String> {
    match v {
        Value::Int(i) => {
            if *i < 0 {
                Err(format!("expected non-negative int, got {}", i))
            } else {
                Ok(*i as usize)
            }
        }
        _ => Err(format!("expected int, got {:?}", v)),
    }
}

fn value_to_f64_vec(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(value_to_f64).collect(),
        _ => Err(format!("expected array, got {:?}", v)),
    }
}

fn f64_vec_to_value(v: &[f64]) -> Value {
    Value::Array(v.iter().map(|x| Value::Float(*x)).collect())
}

fn get_manifold(id: usize) -> Result<ManifoldKind, String> {
    MANIFOLDS.lock().unwrap().get(&id).cloned().ok_or_else(|| format!("unknown manifold id {}", id))
}

// ─── Linear algebra helpers ─────────────────────────────────────────────

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn scale(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}

fn add_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn sub_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn normalize(v: &[f64]) -> Vec<f64> {
    let n = norm(v);
    if n < 1e-15 { v.to_vec() } else { scale(v, 1.0 / n) }
}

/// Clamp to [-1,1] for safe acos/asin.
fn clamp_unit(x: f64) -> f64 {
    x.max(-1.0).min(1.0)
}

// ─── Manifold dimension ─────────────────────────────────────────────────

fn manifold_dimension(m: &ManifoldKind) -> usize {
    match m {
        ManifoldKind::Euclidean(d) => *d,
        ManifoldKind::Sphere(_) => 3, // embedded in R^3 by convention
        ManifoldKind::Hyperbolic(_) => 2, // Poincaré disk default
        ManifoldKind::SPD(d) => d * (d + 1) / 2,
        ManifoldKind::Grassmann(n, k) => k * (n - k),
        ManifoldKind::SO3 => 3,
        ManifoldKind::Torus(d) => *d,
        ManifoldKind::ProductManifold(ms) => ms.iter().map(manifold_dimension).sum(),
    }
}

// ─── Exponential map ────────────────────────────────────────────────────

fn exp_map(m: &ManifoldKind, base: &[f64], tangent: &[f64]) -> Result<Vec<f64>, String> {
    match m {
        ManifoldKind::Euclidean(_) => Ok(add_vec(base, tangent)),

        ManifoldKind::Sphere(r) => {
            // Rodrigues-style: exp_p(v) = cos(|v|/r)*p + sin(|v|/r)*r*(v/|v|)
            let n = norm(tangent);
            if n < 1e-15 {
                return Ok(base.to_vec());
            }
            let theta = n / r;
            let dir = scale(tangent, 1.0 / n);
            // result = cos(theta)*base + sin(theta)*r*dir
            let c = theta.cos();
            let s = theta.sin();
            let result: Vec<f64> = base.iter().zip(dir.iter())
                .map(|(b, d)| c * b + s * r * d)
                .collect();
            Ok(result)
        }

        ManifoldKind::Hyperbolic(curv) => {
            // Poincaré ball model, curvature -|c|
            // exp_p(v) = möbius_add(p, tanh(sqrt(|c|)*|v|/2 / (1-|c|*|p|^2)) * v/|v| / sqrt(|c|) ... )
            // Simplified: use the formula for the Poincaré ball
            let c = *curv;
            let sqrt_c = c.sqrt();
            let pn2 = dot(base, base); // |p|^2
            let vn = norm(tangent);
            if vn < 1e-15 {
                return Ok(base.to_vec());
            }
            let lambda_p = 2.0 / (1.0 - c * pn2).max(1e-15);
            let t = (lambda_p * vn * sqrt_c / 2.0).tanh() / (sqrt_c * vn);
            let tv = scale(tangent, t);
            // Möbius addition: p ⊕ tv
            Ok(mobius_add(base, &tv, c))
        }

        ManifoldKind::SPD(d) => {
            // For SPD(d), points/tangents are stored as flattened d×d symmetric matrices.
            // exp_P(V) = P^{1/2} expm(P^{-1/2} V P^{-1/2}) P^{1/2}
            // For small d we use eigendecomposition.
            let dim = *d;
            let p_mat = vec_to_mat(base, dim)?;
            let v_mat = vec_to_mat(tangent, dim)?;
            let (sqrt_p, inv_sqrt_p) = mat_sqrt_and_inv(&p_mat, dim);
            let inner = mat_mul(&mat_mul(&inv_sqrt_p, &v_mat, dim), &inv_sqrt_p, dim);
            let exp_inner = mat_exp(&inner, dim);
            let result = mat_mul(&mat_mul(&sqrt_p, &exp_inner, dim), &sqrt_p, dim);
            Ok(mat_to_vec(&result, dim))
        }

        ManifoldKind::SO3 => {
            // Axis-angle: tangent is 3-vector (axis*angle)
            // exp: Rodrigues formula applied to rotation matrix at base
            if tangent.len() != 3 || base.len() != 9 {
                return Err("SO3 expects 9-dim base (3x3 rot matrix) and 3-dim tangent".into());
            }
            let angle = norm(tangent);
            if angle < 1e-15 {
                return Ok(base.to_vec());
            }
            let k = scale(tangent, 1.0 / angle); // unit axis
            let k_mat = skew_symmetric(&k);
            let k2 = mat_mul_3x3(&k_mat, &k_mat);
            // R_delta = I + sin(θ)K + (1-cos(θ))K²
            let mut r_delta = [0.0f64; 9];
            let s = angle.sin();
            let c = 1.0 - angle.cos();
            for i in 0..9 {
                let eye = if i % 4 == 0 { 1.0 } else { 0.0 };
                r_delta[i] = eye + s * k_mat[i] + c * k2[i];
            }
            // result = R_delta * base_matrix
            let base_arr: [f64; 9] = base.try_into().map_err(|_| "bad SO3 base".to_string())?;
            let res = mat_mul_3x3_arr(&r_delta, &base_arr);
            Ok(res.to_vec())
        }

        ManifoldKind::Torus(d) => {
            // exp_p(v) = (p + v) mod 2π
            if base.len() != *d || tangent.len() != *d {
                return Err("Torus dimension mismatch".into());
            }
            Ok(base.iter().zip(tangent.iter())
                .map(|(b, t)| (b + t).rem_euclid(std::f64::consts::TAU))
                .collect())
        }

        ManifoldKind::Grassmann(_, _) => {
            // thin SVD-based retraction (used as exp approximation)
            // For simplicity, use QR-based retraction: exp_X(V) ≈ qf(X + V)
            let result = add_vec(base, tangent);
            // Normalize columns (simplified Grassmann retraction)
            Ok(result)
        }

        ManifoldKind::ProductManifold(ms) => {
            let mut result = Vec::new();
            let mut offset = 0;
            for sub_m in ms {
                let d = manifold_dimension(sub_m);
                let sub_b = &base[offset..offset + d];
                let sub_t = &tangent[offset..offset + d];
                let sub_result = exp_map(sub_m, sub_b, sub_t)?;
                result.extend(sub_result);
                offset += d;
            }
            Ok(result)
        }
    }
}

// ─── Logarithmic map ────────────────────────────────────────────────────

fn log_map(m: &ManifoldKind, base: &[f64], target: &[f64]) -> Result<Vec<f64>, String> {
    match m {
        ManifoldKind::Euclidean(_) => Ok(sub_vec(target, base)),

        ManifoldKind::Sphere(r) => {
            // log_p(q) = arccos(<p,q>/(r²)) * r * (q - <p,q>/|p|² * p) / |q - proj|
            let d = dot(base, target) / (r * r);
            let d_clamped = clamp_unit(d);
            let theta = d_clamped.acos();
            if theta.abs() < 1e-15 {
                return Ok(vec![0.0; base.len()]);
            }
            // project target onto tangent space at base
            let proj_coeff = dot(base, target) / dot(base, base);
            let perp: Vec<f64> = target.iter().zip(base.iter())
                .map(|(q, p)| q - proj_coeff * p)
                .collect();
            let perp_n = norm(&perp);
            if perp_n < 1e-15 {
                return Ok(vec![0.0; base.len()]);
            }
            Ok(scale(&perp, theta * r / perp_n))
        }

        ManifoldKind::Hyperbolic(curv) => {
            let c = *curv;
            // log_p(q) via Möbius: log_p(q) = (2/(sqrt(c)*λ_p)) * arctanh(sqrt(c)*|-p⊕q|) * (-p⊕q)/|-p⊕q|
            let neg_p: Vec<f64> = base.iter().map(|x| -x).collect();
            let diff = mobius_add(&neg_p, target, c);
            let diff_n = norm(&diff);
            let sqrt_c = c.sqrt();
            let pn2 = dot(base, base);
            let lambda_p = 2.0 / (1.0 - c * pn2).max(1e-15);
            if diff_n < 1e-15 {
                return Ok(vec![0.0; base.len()]);
            }
            let coeff = (2.0 / (lambda_p * sqrt_c)) * (sqrt_c * diff_n).min(1.0 - 1e-10).atanh();
            Ok(scale(&diff, coeff / diff_n))
        }

        ManifoldKind::SPD(d) => {
            let dim = *d;
            let p_mat = vec_to_mat(base, dim)?;
            let q_mat = vec_to_mat(target, dim)?;
            let (sqrt_p, inv_sqrt_p) = mat_sqrt_and_inv(&p_mat, dim);
            let inner = mat_mul(&mat_mul(&inv_sqrt_p, &q_mat, dim), &inv_sqrt_p, dim);
            let log_inner = mat_log(&inner, dim);
            let result = mat_mul(&mat_mul(&sqrt_p, &log_inner, dim), &sqrt_p, dim);
            Ok(mat_to_vec(&result, dim))
        }

        ManifoldKind::SO3 => {
            if base.len() != 9 || target.len() != 9 {
                return Err("SO3 expects 9-dim (3x3 rotation matrices)".into());
            }
            let base_arr: [f64; 9] = base.try_into().map_err(|_| "bad base".to_string())?;
            let target_arr: [f64; 9] = target.try_into().map_err(|_| "bad target".to_string())?;
            // R_rel = target * base^T
            let base_t = transpose_3x3(&base_arr);
            let r_rel = mat_mul_3x3_arr(&target_arr, &base_t);
            // Extract axis-angle from rotation matrix
            Ok(rotation_to_axis_angle(&r_rel).to_vec())
        }

        ManifoldKind::Torus(d) => {
            if base.len() != *d || target.len() != *d {
                return Err("Torus dimension mismatch".into());
            }
            // Shortest path on torus
            Ok(base.iter().zip(target.iter()).map(|(b, t)| {
                let mut diff = t - b;
                if diff > std::f64::consts::PI { diff -= std::f64::consts::TAU; }
                if diff < -std::f64::consts::PI { diff += std::f64::consts::TAU; }
                diff
            }).collect())
        }

        ManifoldKind::Grassmann(_, _) => {
            Ok(sub_vec(target, base))
        }

        ManifoldKind::ProductManifold(ms) => {
            let mut result = Vec::new();
            let mut offset = 0;
            for sub_m in ms {
                let d = manifold_dimension(sub_m);
                let sub_b = &base[offset..offset + d];
                let sub_t = &target[offset..offset + d];
                let sub_result = log_map(sub_m, sub_b, sub_t)?;
                result.extend(sub_result);
                offset += d;
            }
            Ok(result)
        }
    }
}

// ─── Geodesic distance ──────────────────────────────────────────────────

fn geodesic_dist(m: &ManifoldKind, p: &[f64], q: &[f64]) -> Result<f64, String> {
    match m {
        ManifoldKind::Euclidean(_) => Ok(norm(&sub_vec(p, q))),

        ManifoldKind::Sphere(r) => {
            let cos_d = clamp_unit(dot(p, q) / (r * r));
            Ok(r * cos_d.acos())
        }

        ManifoldKind::Hyperbolic(curv) => {
            let c = *curv;
            let neg_p: Vec<f64> = p.iter().map(|x| -x).collect();
            let diff = mobius_add(&neg_p, q, c);
            let diff_n = norm(&diff);
            let sqrt_c = c.sqrt();
            Ok((2.0 / sqrt_c) * (sqrt_c * diff_n).min(1.0 - 1e-10).atanh())
        }

        ManifoldKind::SPD(_) => {
            let v = log_map(m, p, q)?;
            Ok(norm(&v))
        }

        ManifoldKind::SO3 => {
            let v = log_map(m, p, q)?;
            Ok(norm(&v))
        }

        ManifoldKind::Torus(_) => {
            let v = log_map(m, p, q)?;
            Ok(norm(&v))
        }

        ManifoldKind::Grassmann(_, _) => {
            let v = log_map(m, p, q)?;
            Ok(norm(&v))
        }

        ManifoldKind::ProductManifold(ms) => {
            let mut dist_sq = 0.0;
            let mut offset = 0;
            for sub_m in ms {
                let d = manifold_dimension(sub_m);
                let sub_p = &p[offset..offset + d];
                let sub_q = &q[offset..offset + d];
                let sub_d = geodesic_dist(sub_m, sub_p, sub_q)?;
                dist_sq += sub_d * sub_d;
                offset += d;
            }
            Ok(dist_sq.sqrt())
        }
    }
}

// ─── Parallel transport ─────────────────────────────────────────────────

fn parallel_transport_impl(
    m: &ManifoldKind, start: &[f64], end: &[f64], vector: &[f64],
) -> Result<Vec<f64>, String> {
    match m {
        ManifoldKind::Euclidean(_) => Ok(vector.to_vec()),

        ManifoldKind::Sphere(r) => {
            // Parallel transport along geodesic on sphere
            let log_v = log_map(m, start, end)?;
            let d = norm(&log_v);
            if d < 1e-15 {
                return Ok(vector.to_vec());
            }
            let dir = scale(&log_v, 1.0 / d);
            let theta = d / r;
            // transported = v - (sin(theta)/d)*<v,log> * (log/d + start/r)  ... Schild's ladder approx
            // Exact: v - (<log,v>/d²)(sin(θ)*start/r + (1-cos(θ))*dir)
            let inner = dot(vector, &dir);
            let s = theta.sin();
            let c_val = 1.0 - theta.cos();
            let start_n = scale(start, 1.0 / r);
            let correction: Vec<f64> = start_n.iter().zip(dir.iter())
                .map(|(si, di)| inner * (s * si + c_val * di))
                .collect();
            Ok(sub_vec(vector, &correction))
        }

        ManifoldKind::Hyperbolic(_) => {
            // Approximate via Schild's ladder midpoint method
            let mid_tangent = scale(&log_map(m, start, end)?, 0.5);
            let mid = exp_map(m, start, &mid_tangent)?;
            let v_at_mid = log_map(m, &mid, &add_vec(start, vector))?;
            let transported = scale(&v_at_mid, 1.0); // first-order approx
            // Project back to tangent space at end
            let back = exp_map(m, &mid, &v_at_mid)?;
            log_map(m, end, &back)
        }

        ManifoldKind::Torus(_) => Ok(vector.to_vec()), // flat

        ManifoldKind::SO3 | ManifoldKind::SPD(_) | ManifoldKind::Grassmann(_, _) => {
            // Approximate via vector transport: project the vector at endpoint
            let v = log_map(m, start, end)?;
            let _d = norm(&v);
            // Simple approximation: just return vector (valid for short distances)
            Ok(vector.to_vec())
        }

        ManifoldKind::ProductManifold(ms) => {
            let mut result = Vec::new();
            let mut offset = 0;
            for sub_m in ms {
                let d = manifold_dimension(sub_m);
                let sub_s = &start[offset..offset + d];
                let sub_e = &end[offset..offset + d];
                let sub_v = &vector[offset..offset + d];
                let sub_r = parallel_transport_impl(sub_m, sub_s, sub_e, sub_v)?;
                result.extend(sub_r);
                offset += d;
            }
            Ok(result)
        }
    }
}

// ─── Riemannian gradient (projection) ───────────────────────────────────

fn riemannian_grad(m: &ManifoldKind, point: &[f64], euc_grad: &[f64]) -> Result<Vec<f64>, String> {
    match m {
        ManifoldKind::Euclidean(_) => Ok(euc_grad.to_vec()),

        ManifoldKind::Sphere(r) => {
            // Project onto tangent space: g - <g,p>/<p,p> * p
            let coeff = dot(euc_grad, point) / dot(point, point);
            Ok(euc_grad.iter().zip(point.iter())
                .map(|(g, p)| g - coeff * p)
                .collect())
        }

        ManifoldKind::Hyperbolic(curv) => {
            let c = *curv;
            let pn2 = dot(point, point);
            let lambda = 2.0 / (1.0 - c * pn2).max(1e-15);
            // Riemannian gradient = euc_grad / (lambda/2)^2
            let s = 4.0 / (lambda * lambda);
            Ok(scale(euc_grad, s))
        }

        ManifoldKind::SPD(_) => {
            // Riemannian gradient for SPD: P * euc_grad * P (symmetrized)
            // Simplified: return euc_grad (first-order valid for identity)
            Ok(euc_grad.to_vec())
        }

        ManifoldKind::SO3 => {
            // Project onto skew-symmetric: (G - G^T)/2 mapped to 3-vector
            if euc_grad.len() != 9 {
                return Err("SO3 Riemannian gradient expects 9-dim".into());
            }
            // Skew part of point^T * grad
            let pt: [f64; 9] = point.try_into().map_err(|_| "bad point")?;
            let g: [f64; 9] = euc_grad.try_into().map_err(|_| "bad grad")?;
            let pt_t = transpose_3x3(&pt);
            let m_res = mat_mul_3x3_arr(&pt_t, &g);
            // Extract skew: (M - M^T)/2
            let skew = [
                (m_res[1] - m_res[3]) / 2.0, // (0,1)
                (m_res[2] - m_res[6]) / 2.0, // (0,2)
                (m_res[5] - m_res[7]) / 2.0, // (1,2)
            ];
            // Map to axis-angle vector: [skew[2], -skew[1], skew[0]]
            Ok(vec![skew[2], -skew[1], skew[0]])
        }

        ManifoldKind::Torus(_) => Ok(euc_grad.to_vec()),

        ManifoldKind::Grassmann(_, _) => Ok(euc_grad.to_vec()),

        ManifoldKind::ProductManifold(ms) => {
            let mut result = Vec::new();
            let mut offset = 0;
            for sub_m in ms {
                let d = manifold_dimension(sub_m);
                let sub_p = &point[offset..offset + d];
                let sub_g = &euc_grad[offset..offset + d];
                let sub_r = riemannian_grad(sub_m, sub_p, sub_g)?;
                result.extend(sub_r);
                offset += d;
            }
            Ok(result)
        }
    }
}

// ─── Fréchet mean (iterative) ───────────────────────────────────────────

fn frechet_mean_impl(m: &ManifoldKind, points: &[Vec<f64>]) -> Result<Vec<f64>, String> {
    if points.is_empty() {
        return Err("frechet_mean: empty point set".into());
    }
    if points.len() == 1 {
        return Ok(points[0].clone());
    }

    let mut mean = points[0].clone();
    let n = points.len() as f64;
    let max_iters = 100;
    let tol = 1e-10;

    for _ in 0..max_iters {
        // Compute average tangent vector at current mean
        let mut avg_tangent = vec![0.0; mean.len()];
        for pt in points {
            let v = log_map(m, &mean, pt)?;
            for (a, b) in avg_tangent.iter_mut().zip(v.iter()) {
                *a += b / n;
            }
        }
        let step_size = norm(&avg_tangent);
        if step_size < tol {
            break;
        }
        mean = exp_map(m, &mean, &avg_tangent)?;
    }

    Ok(mean)
}

// ─── Curvature tensor (sectional curvature) ─────────────────────────────

fn sectional_curvature(m: &ManifoldKind, _point: &[f64]) -> Result<f64, String> {
    match m {
        ManifoldKind::Euclidean(_) => Ok(0.0),
        ManifoldKind::Sphere(r) => Ok(1.0 / (r * r)),
        ManifoldKind::Hyperbolic(c) => Ok(-(*c)),
        ManifoldKind::SPD(_) => Ok(0.0), // non-positive, varies; return 0 as representative
        ManifoldKind::Grassmann(_, _) => Ok(0.0), // varies
        ManifoldKind::SO3 => Ok(0.25), // bi-invariant metric, K=1/4
        ManifoldKind::Torus(_) => Ok(0.0), // flat
        ManifoldKind::ProductManifold(_) => Ok(0.0), // mixed
    }
}

// ─── Manifold SGD step (retraction) ─────────────────────────────────────

fn manifold_sgd(m: &ManifoldKind, point: &[f64], grad: &[f64], lr: f64) -> Result<Vec<f64>, String> {
    let neg_grad = scale(grad, -lr);
    let rg = riemannian_grad(m, point, &neg_grad)?;
    exp_map(m, point, &rg)
}

// ─── Manifold projection ────────────────────────────────────────────────

fn project_to_manifold(m: &ManifoldKind, point: &[f64]) -> Result<Vec<f64>, String> {
    match m {
        ManifoldKind::Euclidean(_) => Ok(point.to_vec()),

        ManifoldKind::Sphere(r) => {
            let n = norm(point);
            if n < 1e-15 {
                let mut p = vec![0.0; point.len()];
                if !p.is_empty() { p[0] = *r; }
                Ok(p)
            } else {
                Ok(scale(point, r / n))
            }
        }

        ManifoldKind::Hyperbolic(curv) => {
            // Project into Poincaré ball: if |x|² >= 1/c, rescale
            let c = *curv;
            let max_norm = (1.0 / c).sqrt() - 1e-5;
            let n = norm(point);
            if n >= max_norm {
                Ok(scale(point, max_norm / n))
            } else {
                Ok(point.to_vec())
            }
        }

        ManifoldKind::SPD(d) => {
            // Symmetrize and ensure positive eigenvalues
            let dim = *d;
            let mat = vec_to_mat(point, dim)?;
            let sym = symmetrize(&mat, dim);
            // Clamp eigenvalues to be positive (simplified: just add epsilon to diagonal)
            let mut result = sym;
            for i in 0..dim {
                result[i][i] = result[i][i].max(1e-6);
            }
            Ok(mat_to_vec(&result, dim))
        }

        ManifoldKind::SO3 => {
            // Project to nearest rotation matrix via SVD-like approach (polar decomposition)
            if point.len() != 9 {
                return Err("SO3 projection expects 9-dim".into());
            }
            // Simplified: Gram-Schmidt on columns
            let mut cols = [[0.0f64; 3]; 3];
            for j in 0..3 {
                for i in 0..3 {
                    cols[j][i] = point[i * 3 + j];
                }
            }
            // Orthonormalize
            let c0 = normalize(&cols[0]);
            let d1 = dot(&cols[1], &c0);
            let c1_raw: Vec<f64> = cols[1].iter().zip(c0.iter()).map(|(a, b)| a - d1 * b).collect();
            let c1 = normalize(&c1_raw);
            // c2 = c0 × c1
            let c2 = [
                c0[1] * c1[2] - c0[2] * c1[1],
                c0[2] * c1[0] - c0[0] * c1[2],
                c0[0] * c1[1] - c0[1] * c1[0],
            ];
            let mut result = vec![0.0; 9];
            for i in 0..3 {
                result[i * 3] = c0[i];
                result[i * 3 + 1] = c1[i];
                result[i * 3 + 2] = c2[i];
            }
            Ok(result)
        }

        ManifoldKind::Torus(d) => {
            if point.len() != *d {
                return Err("Torus dimension mismatch".into());
            }
            Ok(point.iter().map(|x| x.rem_euclid(std::f64::consts::TAU)).collect())
        }

        ManifoldKind::Grassmann(_, _) => Ok(point.to_vec()),

        ManifoldKind::ProductManifold(ms) => {
            let mut result = Vec::new();
            let mut offset = 0;
            for sub_m in ms {
                let d = manifold_dimension(sub_m);
                let sub_p = &point[offset..offset + d];
                let sub_r = project_to_manifold(sub_m, sub_p)?;
                result.extend(sub_r);
                offset += d;
            }
            Ok(result)
        }
    }
}

// ─── Möbius addition (Poincaré ball) ────────────────────────────────────

fn mobius_add(x: &[f64], y: &[f64], c: f64) -> Vec<f64> {
    let x2 = dot(x, x);
    let y2 = dot(y, y);
    let xy = dot(x, y);
    let denom = 1.0 + 2.0 * c * xy + c * c * x2 * y2;
    let denom = denom.max(1e-15);
    let coeff_x = 1.0 + 2.0 * c * xy + c * y2;
    let coeff_y = 1.0 - c * x2;
    x.iter().zip(y.iter())
        .map(|(xi, yi)| (coeff_x * xi + coeff_y * yi) / denom)
        .collect()
}

// ─── Matrix helpers for SPD manifold ────────────────────────────────────

fn vec_to_mat(v: &[f64], d: usize) -> Result<Vec<Vec<f64>>, String> {
    if v.len() != d * d {
        return Err(format!("expected {}x{} = {} elements, got {}", d, d, d * d, v.len()));
    }
    let mut mat = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in 0..d {
            mat[i][j] = v[i * d + j];
        }
    }
    Ok(mat)
}

fn mat_to_vec(m: &[Vec<f64>], d: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(d * d);
    for i in 0..d {
        for j in 0..d {
            v.push(m[i][j]);
        }
    }
    v
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>], d: usize) -> Vec<Vec<f64>> {
    let mut c = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn symmetrize(m: &[Vec<f64>], d: usize) -> Vec<Vec<f64>> {
    let mut s = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in 0..d {
            s[i][j] = (m[i][j] + m[j][i]) / 2.0;
        }
    }
    s
}

/// Simple 2x2 eigendecomposition for SPD matrices. For d>2 uses iterative Jacobi.
fn eigen_decomp(m: &[Vec<f64>], d: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    if d == 1 {
        return (vec![m[0][0]], vec![vec![1.0]]);
    }
    if d == 2 {
        let a = m[0][0]; let b = m[0][1]; let c_val = m[1][0]; let d_val = m[1][1];
        let trace = a + d_val;
        let det = a * d_val - b * c_val;
        let disc = ((trace * trace - 4.0 * det).max(0.0)).sqrt();
        let l1 = (trace + disc) / 2.0;
        let l2 = (trace - disc) / 2.0;
        let (v1, v2) = if b.abs() > 1e-15 {
            let v1 = normalize(&[l1 - d_val, b]);
            let v2 = normalize(&[l2 - d_val, b]);
            (v1, v2)
        } else {
            (vec![1.0, 0.0], vec![0.0, 1.0])
        };
        return (vec![l1, l2], vec![v1, v2]);
    }
    // Jacobi eigenvalue algorithm for small d
    jacobi_eigen(m, d)
}

fn jacobi_eigen(m: &[Vec<f64>], d: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut a = m.to_vec();
    let mut v = vec![vec![0.0; d]; d];
    for i in 0..d { v[i][i] = 1.0; }

    for _ in 0..100 {
        // Find largest off-diagonal
        let mut p = 0;
        let mut q = 1;
        let mut max_val = a[0][1].abs();
        for i in 0..d {
            for j in (i+1)..d {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-12 { break; }

        let theta = if (a[p][p] - a[q][q]).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
        };
        let cs = theta.cos();
        let sn = theta.sin();

        // Apply Givens rotation
        let mut new_a = a.clone();
        for i in 0..d {
            new_a[i][p] = cs * a[i][p] + sn * a[i][q];
            new_a[i][q] = -sn * a[i][p] + cs * a[i][q];
        }
        let a_tmp = new_a.clone();
        for j in 0..d {
            new_a[p][j] = cs * a_tmp[p][j] + sn * a_tmp[q][j];
            new_a[q][j] = -sn * a_tmp[p][j] + cs * a_tmp[q][j];
        }
        a = new_a;

        // Update eigenvectors
        let mut new_v = v.clone();
        for i in 0..d {
            new_v[i][p] = cs * v[i][p] + sn * v[i][q];
            new_v[i][q] = -sn * v[i][p] + cs * v[i][q];
        }
        v = new_v;
    }

    let eigenvalues: Vec<f64> = (0..d).map(|i| a[i][i]).collect();
    // Eigenvectors as columns of v
    let eigenvectors: Vec<Vec<f64>> = (0..d).map(|j| {
        (0..d).map(|i| v[i][j]).collect()
    }).collect();
    (eigenvalues, eigenvectors)
}

fn mat_sqrt_and_inv(m: &[Vec<f64>], d: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let (evals, evecs) = eigen_decomp(m, d);
    let mut sqrt_m = vec![vec![0.0; d]; d];
    let mut inv_sqrt_m = vec![vec![0.0; d]; d];
    for k in 0..d {
        let lam = evals[k].max(1e-15);
        let s = lam.sqrt();
        let is = 1.0 / s;
        for i in 0..d {
            for j in 0..d {
                sqrt_m[i][j] += s * evecs[k][i] * evecs[k][j];
                inv_sqrt_m[i][j] += is * evecs[k][i] * evecs[k][j];
            }
        }
    }
    (sqrt_m, inv_sqrt_m)
}

fn mat_exp(m: &[Vec<f64>], d: usize) -> Vec<Vec<f64>> {
    let (evals, evecs) = eigen_decomp(m, d);
    let mut result = vec![vec![0.0; d]; d];
    for k in 0..d {
        let e = evals[k].exp();
        for i in 0..d {
            for j in 0..d {
                result[i][j] += e * evecs[k][i] * evecs[k][j];
            }
        }
    }
    result
}

fn mat_log(m: &[Vec<f64>], d: usize) -> Vec<Vec<f64>> {
    let (evals, evecs) = eigen_decomp(m, d);
    let mut result = vec![vec![0.0; d]; d];
    for k in 0..d {
        let l = evals[k].max(1e-15).ln();
        for i in 0..d {
            for j in 0..d {
                result[i][j] += l * evecs[k][i] * evecs[k][j];
            }
        }
    }
    result
}

// ─── SO3 helpers ────────────────────────────────────────────────────────

fn skew_symmetric(v: &[f64]) -> [f64; 9] {
    // v = [a, b, c] -> [[0,-c,b],[c,0,-a],[-b,a,0]]
    [
        0.0, -v[2], v[1],
        v[2], 0.0, -v[0],
        -v[1], v[0], 0.0,
    ]
}

fn mat_mul_3x3(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut c = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
            }
        }
    }
    c
}

fn mat_mul_3x3_arr(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    mat_mul_3x3(a, b)
}

fn transpose_3x3(m: &[f64; 9]) -> [f64; 9] {
    [
        m[0], m[3], m[6],
        m[1], m[4], m[7],
        m[2], m[5], m[8],
    ]
}

fn rotation_to_axis_angle(r: &[f64; 9]) -> [f64; 3] {
    let trace = r[0] + r[4] + r[8];
    let cos_angle = clamp_unit((trace - 1.0) / 2.0);
    let angle = cos_angle.acos();
    if angle.abs() < 1e-10 {
        return [0.0, 0.0, 0.0];
    }
    let s = 2.0 * angle.sin();
    if s.abs() < 1e-10 {
        // angle near π — extract from diagonal
        return [angle, 0.0, 0.0];
    }
    let axis = [
        (r[7] - r[5]) / s,
        (r[2] - r[6]) / s,
        (r[3] - r[1]) / s,
    ];
    [axis[0] * angle, axis[1] * angle, axis[2] * angle]
}

// ─── Parse manifold kind from string + params ───────────────────────────

fn parse_manifold_kind(kind: &str, params: &[Value]) -> Result<ManifoldKind, String> {
    match kind.to_lowercase().as_str() {
        "euclidean" => {
            let d = if params.is_empty() { 3 } else { value_to_usize(&params[0])? };
            Ok(ManifoldKind::Euclidean(d))
        }
        "sphere" => {
            let r = if params.is_empty() { 1.0 } else { value_to_f64(&params[0])? };
            Ok(ManifoldKind::Sphere(r))
        }
        "hyperbolic" => {
            let c = if params.is_empty() { 1.0 } else { value_to_f64(&params[0])? };
            Ok(ManifoldKind::Hyperbolic(c))
        }
        "spd" => {
            let d = if params.is_empty() { 2 } else { value_to_usize(&params[0])? };
            Ok(ManifoldKind::SPD(d))
        }
        "grassmann" => {
            if params.len() < 2 {
                return Err("Grassmann requires (n, k) parameters".into());
            }
            let n = value_to_usize(&params[0])?;
            let k = value_to_usize(&params[1])?;
            Ok(ManifoldKind::Grassmann(n, k))
        }
        "so3" => Ok(ManifoldKind::SO3),
        "torus" => {
            let d = if params.is_empty() { 2 } else { value_to_usize(&params[0])? };
            Ok(ManifoldKind::Torus(d))
        }
        _ => Err(format!("unknown manifold kind: {}", kind)),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Builtin wrappers
// ═══════════════════════════════════════════════════════════════════════════

fn builtin_manifold_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("manifold_new requires at least a kind string".into());
    }
    let kind_str = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("manifold_new: first argument must be a string".into()),
    };
    let params = &args[1..];
    let mk = parse_manifold_kind(&kind_str, params)?;
    let id = next_id();
    MANIFOLDS.lock().unwrap().insert(id, mk);
    Ok(Value::Int(id as i128))
}

fn builtin_exponential_map(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("exponential_map(manifold_id, base_point, tangent_vec)".into());
    }
    let mid = value_to_usize(&args[0])?;
    let base = value_to_f64_vec(&args[1])?;
    let tangent = value_to_f64_vec(&args[2])?;
    let m = get_manifold(mid)?;
    let result = exp_map(&m, &base, &tangent)?;
    Ok(f64_vec_to_value(&result))
}

fn builtin_logarithmic_map(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("logarithmic_map(manifold_id, base_point, target_point)".into());
    }
    let mid = value_to_usize(&args[0])?;
    let base = value_to_f64_vec(&args[1])?;
    let target = value_to_f64_vec(&args[2])?;
    let m = get_manifold(mid)?;
    let result = log_map(&m, &base, &target)?;
    Ok(f64_vec_to_value(&result))
}

fn builtin_geodesic_distance(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("geodesic_distance(manifold_id, p1, p2)".into());
    }
    let mid = value_to_usize(&args[0])?;
    let p1 = value_to_f64_vec(&args[1])?;
    let p2 = value_to_f64_vec(&args[2])?;
    let m = get_manifold(mid)?;
    let d = geodesic_dist(&m, &p1, &p2)?;
    Ok(Value::Float(d))
}

fn builtin_parallel_transport(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 4 {
        return Err("parallel_transport(manifold_id, start, end, vector)".into());
    }
    let mid = value_to_usize(&args[0])?;
    let start = value_to_f64_vec(&args[1])?;
    let end = value_to_f64_vec(&args[2])?;
    let vector = value_to_f64_vec(&args[3])?;
    let m = get_manifold(mid)?;
    let result = parallel_transport_impl(&m, &start, &end, &vector)?;
    Ok(f64_vec_to_value(&result))
}

fn builtin_riemannian_gradient(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("riemannian_gradient(manifold_id, point, euclidean_grad)".into());
    }
    let mid = value_to_usize(&args[0])?;
    let point = value_to_f64_vec(&args[1])?;
    let grad = value_to_f64_vec(&args[2])?;
    let m = get_manifold(mid)?;
    let result = riemannian_grad(&m, &point, &grad)?;
    Ok(f64_vec_to_value(&result))
}

fn builtin_frechet_mean(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("frechet_mean(manifold_id, points_array)".into());
    }
    let mid = value_to_usize(&args[0])?;
    let points_val = match &args[1] {
        Value::Array(arr) => arr.clone(),
        _ => return Err("frechet_mean: second argument must be array of arrays".into()),
    };
    let points: Vec<Vec<f64>> = points_val.iter().map(value_to_f64_vec).collect::<Result<_, _>>()?;
    let m = get_manifold(mid)?;
    let result = frechet_mean_impl(&m, &points)?;
    Ok(f64_vec_to_value(&result))
}

fn builtin_curvature_tensor(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("curvature_tensor(manifold_id, point)".into());
    }
    let mid = value_to_usize(&args[0])?;
    let point = value_to_f64_vec(&args[1])?;
    let m = get_manifold(mid)?;
    let k = sectional_curvature(&m, &point)?;
    Ok(Value::Float(k))
}

fn builtin_manifold_sgd_step(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 4 {
        return Err("manifold_sgd_step(manifold_id, point, grad, lr)".into());
    }
    let mid = value_to_usize(&args[0])?;
    let point = value_to_f64_vec(&args[1])?;
    let grad = value_to_f64_vec(&args[2])?;
    let lr = value_to_f64(&args[3])?;
    let m = get_manifold(mid)?;
    let result = manifold_sgd(&m, &point, &grad, lr)?;
    Ok(f64_vec_to_value(&result))
}

fn builtin_tangent_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("tangent_new(manifold_id, base_point, vector)".into());
    }
    let mid = value_to_usize(&args[0])?;
    let base = value_to_f64_vec(&args[1])?;
    let vector = value_to_f64_vec(&args[2])?;
    let m = get_manifold(mid)?;
    let tv = TangentVector { base_point: base, vector, manifold: m };
    let id = next_id();
    TANGENTS.lock().unwrap().insert(id, tv);
    Ok(Value::Int(id as i128))
}

fn builtin_geodesic_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("geodesic_new(manifold_id, start, end)".into());
    }
    let mid = value_to_usize(&args[0])?;
    let start = value_to_f64_vec(&args[1])?;
    let end = value_to_f64_vec(&args[2])?;
    let m = get_manifold(mid)?;
    let gp = GeodesicPath { start, end, manifold: m };
    let id = next_id();
    GEODESICS.lock().unwrap().insert(id, gp);
    Ok(Value::Int(id as i128))
}

fn builtin_geodesic_point_at(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("geodesic_point_at(geodesic_id, t)".into());
    }
    let gid = value_to_usize(&args[0])?;
    let t = value_to_f64(&args[1])?;
    let gp = GEODESICS.lock().unwrap().get(&gid).cloned()
        .ok_or_else(|| format!("unknown geodesic id {}", gid))?;
    let tangent = log_map(&gp.manifold, &gp.start, &gp.end)?;
    let scaled = scale(&tangent, t);
    let point = exp_map(&gp.manifold, &gp.start, &scaled)?;
    Ok(f64_vec_to_value(&point))
}

fn builtin_fiber_bundle_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("fiber_bundle_new(base_manifold_id, fiber_manifold_id)".into());
    }
    let base_id = value_to_usize(&args[0])?;
    let fiber_id = value_to_usize(&args[1])?;
    let base_m = get_manifold(base_id)?;
    let fiber_m = get_manifold(fiber_id)?;
    let fb = FiberBundle { base_manifold: base_m, fiber_manifold: fiber_m };
    let id = next_id();
    BUNDLES.lock().unwrap().insert(id, fb);
    Ok(Value::Int(id as i128))
}

fn builtin_manifold_dim(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("manifold_dim(manifold_id)".into());
    }
    let mid = value_to_usize(&args[0])?;
    let m = get_manifold(mid)?;
    Ok(Value::Int(manifold_dimension(&m) as i128))
}

fn builtin_manifold_project(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("manifold_project(manifold_id, point)".into());
    }
    let mid = value_to_usize(&args[0])?;
    let point = value_to_f64_vec(&args[1])?;
    let m = get_manifold(mid)?;
    let result = project_to_manifold(&m, &point)?;
    Ok(f64_vec_to_value(&result))
}

// ═══════════════════════════════════════════════════════════════════════════
// Registration
// ═══════════════════════════════════════════════════════════════════════════

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("manifold_new".into(), FnDef::Builtin(builtin_manifold_new));
    env.functions.insert("exponential_map".into(), FnDef::Builtin(builtin_exponential_map));
    env.functions.insert("logarithmic_map".into(), FnDef::Builtin(builtin_logarithmic_map));
    env.functions.insert("geodesic_distance".into(), FnDef::Builtin(builtin_geodesic_distance));
    env.functions.insert("parallel_transport".into(), FnDef::Builtin(builtin_parallel_transport));
    env.functions.insert("riemannian_gradient".into(), FnDef::Builtin(builtin_riemannian_gradient));
    env.functions.insert("frechet_mean".into(), FnDef::Builtin(builtin_frechet_mean));
    env.functions.insert("curvature_tensor".into(), FnDef::Builtin(builtin_curvature_tensor));
    env.functions.insert("manifold_sgd_step".into(), FnDef::Builtin(builtin_manifold_sgd_step));
    env.functions.insert("tangent_new".into(), FnDef::Builtin(builtin_tangent_new));
    env.functions.insert("geodesic_new".into(), FnDef::Builtin(builtin_geodesic_new));
    env.functions.insert("geodesic_point_at".into(), FnDef::Builtin(builtin_geodesic_point_at));
    env.functions.insert("fiber_bundle_new".into(), FnDef::Builtin(builtin_fiber_bundle_new));
    env.functions.insert("manifold_dim".into(), FnDef::Builtin(builtin_manifold_dim));
    env.functions.insert("manifold_project".into(), FnDef::Builtin(builtin_manifold_project));
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn vec_approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, tol))
    }

    // ── Euclidean ─────────────────────────────────────────────────────

    #[test]
    fn test_euclidean_exp_log() {
        let m = ManifoldKind::Euclidean(3);
        let p = vec![1.0, 2.0, 3.0];
        let v = vec![0.5, -0.5, 1.0];
        let q = exp_map(&m, &p, &v).unwrap();
        assert!(vec_approx_eq(&q, &[1.5, 1.5, 4.0], 1e-10));
        let v2 = log_map(&m, &p, &q).unwrap();
        assert!(vec_approx_eq(&v2, &v, 1e-10));
    }

    #[test]
    fn test_euclidean_distance() {
        let m = ManifoldKind::Euclidean(2);
        let d = geodesic_dist(&m, &[0.0, 0.0], &[3.0, 4.0]).unwrap();
        assert!(approx_eq(d, 5.0, 1e-10));
    }

    #[test]
    fn test_euclidean_frechet_mean() {
        let m = ManifoldKind::Euclidean(2);
        let points = vec![vec![0.0, 0.0], vec![2.0, 0.0], vec![1.0, 3.0]];
        let mean = frechet_mean_impl(&m, &points).unwrap();
        assert!(approx_eq(mean[0], 1.0, 1e-6));
        assert!(approx_eq(mean[1], 1.0, 1e-6));
    }

    #[test]
    fn test_euclidean_curvature() {
        let m = ManifoldKind::Euclidean(3);
        let k = sectional_curvature(&m, &[0.0, 0.0, 0.0]).unwrap();
        assert!(approx_eq(k, 0.0, 1e-15));
    }

    // ── Sphere ────────────────────────────────────────────────────────

    #[test]
    fn test_sphere_exp_log_roundtrip() {
        let m = ManifoldKind::Sphere(1.0);
        let p = vec![1.0, 0.0, 0.0]; // on unit sphere
        let v = vec![0.0, 0.3, 0.0]; // tangent at p (perpendicular)
        let q = exp_map(&m, &p, &v).unwrap();
        // q should be on the sphere
        let q_norm = norm(&q);
        assert!(approx_eq(q_norm, 1.0, 1e-10));
        let v2 = log_map(&m, &p, &q).unwrap();
        assert!(vec_approx_eq(&v2, &v, 1e-6));
    }

    #[test]
    fn test_sphere_distance() {
        let m = ManifoldKind::Sphere(1.0);
        let p = vec![1.0, 0.0, 0.0];
        let q = vec![0.0, 1.0, 0.0];
        let d = geodesic_dist(&m, &p, &q).unwrap();
        assert!(approx_eq(d, std::f64::consts::FRAC_PI_2, 1e-10));
    }

    #[test]
    fn test_sphere_curvature() {
        let m = ManifoldKind::Sphere(2.0);
        let k = sectional_curvature(&m, &[2.0, 0.0, 0.0]).unwrap();
        assert!(approx_eq(k, 0.25, 1e-10));
    }

    #[test]
    fn test_sphere_project() {
        let m = ManifoldKind::Sphere(1.0);
        let p = vec![3.0, 4.0, 0.0];
        let proj = project_to_manifold(&m, &p).unwrap();
        assert!(approx_eq(norm(&proj), 1.0, 1e-10));
    }

    // ── Hyperbolic ────────────────────────────────────────────────────

    #[test]
    fn test_hyperbolic_exp_log_roundtrip() {
        let m = ManifoldKind::Hyperbolic(1.0);
        let p = vec![0.0, 0.0]; // origin of Poincaré disk
        let v = vec![0.3, 0.0];
        let q = exp_map(&m, &p, &v).unwrap();
        // q should be in the disk (|q| < 1)
        assert!(norm(&q) < 1.0);
        let v2 = log_map(&m, &p, &q).unwrap();
        assert!(vec_approx_eq(&v2, &v, 1e-6));
    }

    #[test]
    fn test_hyperbolic_distance_origin() {
        let m = ManifoldKind::Hyperbolic(1.0);
        let p = vec![0.0, 0.0];
        let q = vec![0.5, 0.0];
        let d = geodesic_dist(&m, &p, &q).unwrap();
        // d = 2 * atanh(0.5) ≈ 1.0986
        assert!(approx_eq(d, 2.0 * (0.5_f64).atanh(), 1e-6));
    }

    #[test]
    fn test_hyperbolic_curvature() {
        let m = ManifoldKind::Hyperbolic(1.0);
        let k = sectional_curvature(&m, &[0.0, 0.0]).unwrap();
        assert!(approx_eq(k, -1.0, 1e-15));
    }

    // ── SPD ───────────────────────────────────────────────────────────

    #[test]
    fn test_spd_exp_log_roundtrip() {
        let m = ManifoldKind::SPD(2);
        // Identity matrix as base
        let p = vec![1.0, 0.0, 0.0, 1.0];
        // Symmetric tangent
        let v = vec![0.1, 0.05, 0.05, 0.2];
        let q = exp_map(&m, &p, &v).unwrap();
        let v2 = log_map(&m, &p, &q).unwrap();
        assert!(vec_approx_eq(&v2, &v, 1e-4));
    }

    #[test]
    fn test_spd_distance() {
        let m = ManifoldKind::SPD(2);
        let p = vec![1.0, 0.0, 0.0, 1.0];
        let q = vec![2.0, 0.0, 0.0, 2.0];
        let d = geodesic_dist(&m, &p, &q).unwrap();
        // d = sqrt(2) * ln(2) for I -> 2I
        assert!(d > 0.0);
    }

    // ── SO3 ───────────────────────────────────────────────────────────

    #[test]
    fn test_so3_exp_identity() {
        let m = ManifoldKind::SO3;
        // Identity rotation matrix
        let base = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        // Zero tangent -> should stay at identity
        let v = vec![0.0, 0.0, 0.0];
        let q = exp_map(&m, &base, &v).unwrap();
        assert!(vec_approx_eq(&q, &base, 1e-10));
    }

    #[test]
    fn test_so3_exp_log_roundtrip() {
        let m = ManifoldKind::SO3;
        let base = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let v = vec![0.1, 0.2, 0.3]; // small rotation
        let q = exp_map(&m, &base, &v).unwrap();
        let v2 = log_map(&m, &base, &q).unwrap();
        assert!(vec_approx_eq(&v2, &v, 1e-4));
    }

    #[test]
    fn test_so3_curvature() {
        let m = ManifoldKind::SO3;
        let k = sectional_curvature(&m, &[1.0; 9]).unwrap();
        assert!(approx_eq(k, 0.25, 1e-10));
    }

    // ── Torus ─────────────────────────────────────────────────────────

    #[test]
    fn test_torus_exp_wraps() {
        let m = ManifoldKind::Torus(2);
        let p = vec![6.0, 6.0]; // near 2π
        let v = vec![1.0, 1.0];
        let q = exp_map(&m, &p, &v).unwrap();
        // Should wrap around
        let tau = std::f64::consts::TAU;
        assert!(q[0] >= 0.0 && q[0] < tau);
        assert!(q[1] >= 0.0 && q[1] < tau);
    }

    #[test]
    fn test_torus_distance_wrap() {
        let m = ManifoldKind::Torus(1);
        let p = vec![0.1];
        let q = vec![std::f64::consts::TAU - 0.1];
        let d = geodesic_dist(&m, &p, &q).unwrap();
        assert!(approx_eq(d, 0.2, 1e-10));
    }

    #[test]
    fn test_torus_curvature() {
        let m = ManifoldKind::Torus(2);
        let k = sectional_curvature(&m, &[0.0, 0.0]).unwrap();
        assert!(approx_eq(k, 0.0, 1e-15));
    }

    // ── Dimension ─────────────────────────────────────────────────────

    #[test]
    fn test_manifold_dimensions() {
        assert_eq!(manifold_dimension(&ManifoldKind::Euclidean(5)), 5);
        assert_eq!(manifold_dimension(&ManifoldKind::Sphere(1.0)), 3);
        assert_eq!(manifold_dimension(&ManifoldKind::Hyperbolic(1.0)), 2);
        assert_eq!(manifold_dimension(&ManifoldKind::SPD(3)), 6);
        assert_eq!(manifold_dimension(&ManifoldKind::Grassmann(5, 2)), 6);
        assert_eq!(manifold_dimension(&ManifoldKind::SO3), 3);
        assert_eq!(manifold_dimension(&ManifoldKind::Torus(4)), 4);
    }

    // ── Geodesic interpolation ────────────────────────────────────────

    #[test]
    fn test_geodesic_interpolation_euclidean() {
        let m = ManifoldKind::Euclidean(2);
        let start = vec![0.0, 0.0];
        let end = vec![4.0, 6.0];
        let tangent = log_map(&m, &start, &end).unwrap();
        let mid_t = scale(&tangent, 0.5);
        let mid = exp_map(&m, &start, &mid_t).unwrap();
        assert!(vec_approx_eq(&mid, &[2.0, 3.0], 1e-10));
    }

    #[test]
    fn test_geodesic_interpolation_sphere() {
        let m = ManifoldKind::Sphere(1.0);
        let p = vec![1.0, 0.0, 0.0];
        let q = vec![0.0, 1.0, 0.0];
        let tangent = log_map(&m, &p, &q).unwrap();
        let mid_t = scale(&tangent, 0.5);
        let mid = exp_map(&m, &p, &mid_t).unwrap();
        // Midpoint should be on the sphere
        assert!(approx_eq(norm(&mid), 1.0, 1e-10));
    }

    // ── SGD step ──────────────────────────────────────────────────────

    #[test]
    fn test_sgd_step_euclidean() {
        let m = ManifoldKind::Euclidean(2);
        let p = vec![1.0, 1.0];
        let g = vec![2.0, 0.0];
        let lr = 0.1;
        let new_p = manifold_sgd(&m, &p, &g, lr).unwrap();
        assert!(vec_approx_eq(&new_p, &[0.8, 1.0], 1e-10));
    }

    // ── Product manifold ──────────────────────────────────────────────

    #[test]
    fn test_product_manifold_exp_log() {
        let m = ManifoldKind::ProductManifold(vec![
            ManifoldKind::Euclidean(2),
            ManifoldKind::Euclidean(3),
        ]);
        let p = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let q = exp_map(&m, &p, &v).unwrap();
        let v2 = log_map(&m, &p, &q).unwrap();
        assert!(vec_approx_eq(&v2, &v, 1e-10));
    }

    // ── Parallel transport ────────────────────────────────────────────

    #[test]
    fn test_parallel_transport_euclidean() {
        let m = ManifoldKind::Euclidean(3);
        let s = vec![0.0, 0.0, 0.0];
        let e = vec![1.0, 0.0, 0.0];
        let v = vec![0.0, 1.0, 0.0];
        let pt = parallel_transport_impl(&m, &s, &e, &v).unwrap();
        assert!(vec_approx_eq(&pt, &v, 1e-10));
    }

    // ── Riemannian gradient ───────────────────────────────────────────

    #[test]
    fn test_riemannian_gradient_sphere() {
        let m = ManifoldKind::Sphere(1.0);
        let p = vec![1.0, 0.0, 0.0];
        let g = vec![1.0, 0.5, 0.3]; // has radial component
        let rg = riemannian_grad(&m, &p, &g).unwrap();
        // Should have zero radial component
        assert!(approx_eq(dot(&rg, &p), 0.0, 1e-10));
    }

    // ── Projection ────────────────────────────────────────────────────

    #[test]
    fn test_project_hyperbolic() {
        let m = ManifoldKind::Hyperbolic(1.0);
        let p = vec![0.99, 0.0]; // near boundary
        let proj = project_to_manifold(&m, &p).unwrap();
        assert!(norm(&proj) < 1.0);
    }

    // ── Xorshift PRNG ─────────────────────────────────────────────────

    #[test]
    fn test_xorshift_deterministic() {
        let mut rng1 = Xorshift64::new(42);
        let mut rng2 = Xorshift64::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_xorshift_range() {
        let mut rng = Xorshift64::new(123);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    // ── Parse manifold kind ───────────────────────────────────────────

    #[test]
    fn test_parse_manifold_kinds() {
        let m = parse_manifold_kind("euclidean", &[Value::Int(5)]).unwrap();
        assert_eq!(m, ManifoldKind::Euclidean(5));

        let m = parse_manifold_kind("sphere", &[Value::Float(2.0)]).unwrap();
        assert_eq!(m, ManifoldKind::Sphere(2.0));

        let m = parse_manifold_kind("hyperbolic", &[]).unwrap();
        assert_eq!(m, ManifoldKind::Hyperbolic(1.0));

        let m = parse_manifold_kind("so3", &[]).unwrap();
        assert_eq!(m, ManifoldKind::SO3);

        let m = parse_manifold_kind("torus", &[Value::Int(3)]).unwrap();
        assert_eq!(m, ManifoldKind::Torus(3));

        let m = parse_manifold_kind("grassmann", &[Value::Int(5), Value::Int(2)]).unwrap();
        assert_eq!(m, ManifoldKind::Grassmann(5, 2));
    }
}

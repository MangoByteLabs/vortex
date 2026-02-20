use crate::crypto::{BigUint256, ECPoint, point_add, point_double, scalar_mul};

/// Naive MSM: sum of scalar_i * point_i
pub fn msm_naive(scalars: &[BigUint256], points: &[ECPoint]) -> ECPoint {
    assert_eq!(scalars.len(), points.len());
    let mut result = ECPoint::identity();
    for (s, p) in scalars.iter().zip(points.iter()) {
        let sp = scalar_mul(s, p);
        result = point_add(&result, &sp);
    }
    result
}

/// Optimal window size for Pippenger
fn optimal_window(num_points: usize) -> u32 {
    if num_points < 32 {
        return 1;
    }
    let log_n = (num_points as f64).log2() as u32;
    // Heuristic: w ≈ log2(n) - 2, clamped to [2, 16]
    log_n.saturating_sub(2).max(2).min(16)
}

/// Pippenger's bucket method for MSM
///
/// Decomposes each scalar into w-bit windows, accumulates points into buckets,
/// then reduces buckets with running sum and combines windows with Horner's method.
pub fn msm_pippenger(scalars: &[BigUint256], points: &[ECPoint]) -> ECPoint {
    assert_eq!(scalars.len(), points.len());
    let n = scalars.len();
    if n == 0 {
        return ECPoint::identity();
    }
    if n == 1 {
        return scalar_mul(&scalars[0], &points[0]);
    }

    let w = optimal_window(n) as usize;
    let num_buckets = (1 << w) - 1; // buckets 1..2^w - 1

    // Find max scalar bits
    let max_bits = scalars.iter().map(|s| s.bits()).max().unwrap_or(0) as usize;
    let num_windows = (max_bits + w - 1) / w;

    let mut total = ECPoint::identity();

    // Process windows from most significant to least significant
    for window_idx in (0..num_windows).rev() {
        // Double w times for Horner's method (skip first window)
        if window_idx < num_windows - 1 {
            for _ in 0..w {
                total = point_double(&total);
            }
        }

        // Initialize buckets
        let mut buckets: Vec<ECPoint> =
            (0..num_buckets).map(|_| ECPoint::identity()).collect();

        // Scatter: put each point into its bucket
        for (scalar, point) in scalars.iter().zip(points.iter()) {
            // Extract w-bit window from scalar
            let bit_offset = window_idx * w;
            let mut bucket_idx: usize = 0;
            for b in 0..w {
                let bit_pos = bit_offset + b;
                if bit_pos < 256 && scalar.bit(bit_pos as u32) {
                    bucket_idx |= 1 << b;
                }
            }
            if bucket_idx > 0 {
                buckets[bucket_idx - 1] = point_add(&buckets[bucket_idx - 1], point);
            }
        }

        // Reduce buckets with running sum
        // window_sum = sum_{i=1}^{num_buckets} i * buckets[i-1]
        // = buckets[num_buckets-1] + (buckets[num_buckets-1] + buckets[num_buckets-2]) + ...
        let mut running_sum = ECPoint::identity();
        let mut window_sum = ECPoint::identity();
        for i in (0..num_buckets).rev() {
            running_sum = point_add(&running_sum, &buckets[i]);
            window_sum = point_add(&window_sum, &running_sum);
        }

        total = point_add(&total, &window_sum);
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::{point_to_affine, scalar_mul, secp256k1_generator, BigUint256};

    #[test]
    fn test_msm_naive_single() {
        let g = secp256k1_generator();
        let s = BigUint256::from_u64(42);
        let expected = scalar_mul(&s, &g);
        let result = msm_naive(&[s], &[g]);
        let (ex, ey) = point_to_affine(&expected);
        let (rx, ry) = point_to_affine(&result);
        assert_eq!(ex, rx);
        assert_eq!(ey, ry);
    }

    #[test]
    fn test_msm_naive_two() {
        let g = secp256k1_generator();
        let s1 = BigUint256::from_u64(3);
        let s2 = BigUint256::from_u64(5);
        let p1 = g.clone();
        let p2 = scalar_mul(&BigUint256::from_u64(7), &g);

        // 3*G + 5*7G = 3G + 35G = 38G
        let expected = scalar_mul(&BigUint256::from_u64(38), &g);
        let result = msm_naive(&[s1, s2], &[p1, p2]);
        let (ex, ey) = point_to_affine(&expected);
        let (rx, ry) = point_to_affine(&result);
        assert_eq!(ex, rx);
        assert_eq!(ey, ry);
    }

    #[test]
    fn test_msm_pippenger_matches_naive() {
        let g = secp256k1_generator();
        // Create 8 points and scalars
        let mut points = Vec::new();
        let mut scalars = Vec::new();
        for i in 1..=8u64 {
            points.push(scalar_mul(&BigUint256::from_u64(i * 7), &g));
            scalars.push(BigUint256::from_u64(i * 3 + 1));
        }

        let naive_result = msm_naive(&scalars, &points);
        let pippenger_result = msm_pippenger(&scalars, &points);

        let (nx, ny) = point_to_affine(&naive_result);
        let (px, py) = point_to_affine(&pippenger_result);
        assert_eq!(nx, px);
        assert_eq!(ny, py);
    }

    #[test]
    fn test_msm_pippenger_empty() {
        let result = msm_pippenger(&[], &[]);
        assert!(result.is_identity());
    }

    #[test]
    fn test_msm_pippenger_single() {
        let g = secp256k1_generator();
        let s = BigUint256::from_u64(100);
        let expected = scalar_mul(&s, &g);
        let result = msm_pippenger(&[s], &[g]);
        let (ex, ey) = point_to_affine(&expected);
        let (rx, ry) = point_to_affine(&result);
        assert_eq!(ex, rx);
        assert_eq!(ey, ry);
    }
}

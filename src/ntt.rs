use crate::crypto::{BigUint256, mod_add, mod_sub, mod_mul, mod_pow, mod_inv};

/// NTT domain parameters
#[derive(Clone, Debug)]
pub struct NTTDomain {
    pub modulus: BigUint256,
    pub root_of_unity: BigUint256,     // primitive 2^k-th root of unity
    pub two_adicity: u32,              // k such that 2^k | (p-1)
    pub size: usize,                   // current domain size (must be power of 2, <= 2^k)
    pub twiddles: Vec<BigUint256>,     // precomputed twiddle factors [ω^0, ω^1, ..., ω^(n/2-1)]
    pub inv_twiddles: Vec<BigUint256>, // precomputed inverse twiddle factors
    pub size_inv: BigUint256,          // n^(-1) mod p
}

/// BN254 Fr scalar field modulus:
/// p = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
fn bn254_fr_modulus() -> BigUint256 {
    BigUint256::from_hex("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001").unwrap()
}

/// Goldilocks field modulus: p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
fn goldilocks_modulus() -> BigUint256 {
    BigUint256::from_hex("FFFFFFFF00000001").unwrap()
}

/// Create an NTT domain for BN254 Fr field.
///
/// BN254 Fr has 2-adicity = 28, meaning p-1 = 2^28 * t where t is odd.
/// We use g = 5 as a generator of Fr*, and compute ω = g^((p-1)/2^28) as
/// the primitive 2^28-th root of unity.
pub fn bn254_fr_domain(log_size: u32) -> NTTDomain {
    assert!(log_size <= 28, "BN254 Fr 2-adicity is 28, log_size must be <= 28");
    let modulus = bn254_fr_modulus();
    let n = 1usize << log_size;

    // Compute primitive 2^28-th root of unity: ω_max = g^((p-1)/2^28)
    // where g = 5 is a generator of Fr*
    let g = BigUint256::from_u64(5);
    let p_minus_1 = mod_sub(&modulus, &BigUint256::ONE, &modulus);
    // (p-1) / 2^28: shift right 28 bits
    let exp_max = shr_n(&p_minus_1, 28);
    let omega_max = mod_pow(&g, &exp_max, &modulus);

    // The n-th root of unity: ω_n = ω_max ^ (2^(28 - log_size))
    let exp_adjust = BigUint256::from_u64(1u64 << (28 - log_size));
    let omega_n = mod_pow(&omega_max, &exp_adjust, &modulus);

    // Verify: ω_n^n = 1
    debug_assert_eq!(
        mod_pow(&omega_n, &BigUint256::from_u64(n as u64), &modulus),
        BigUint256::ONE,
        "ω^n must equal 1"
    );

    // Verify: ω_n^(n/2) != 1 (primitive)
    if log_size > 0 {
        debug_assert_ne!(
            mod_pow(&omega_n, &BigUint256::from_u64((n / 2) as u64), &modulus),
            BigUint256::ONE,
            "ω^(n/2) must not equal 1 (primitive root check)"
        );
    }

    let twiddles = precompute_twiddles(&omega_n, n, &modulus);
    let omega_inv = mod_inv(&omega_n, &modulus);
    let inv_twiddles = precompute_twiddles(&omega_inv, n, &modulus);
    let size_inv = mod_inv(&BigUint256::from_u64(n as u64), &modulus);

    NTTDomain {
        modulus,
        root_of_unity: omega_n,
        two_adicity: 28,
        size: n,
        twiddles,
        inv_twiddles,
        size_inv,
    }
}

/// Create an NTT domain for the Goldilocks field.
///
/// Goldilocks: p = 2^64 - 2^32 + 1, 2-adicity = 32.
/// Generator g = 7 is a known generator of the multiplicative group.
pub fn goldilocks_domain(log_size: u32) -> NTTDomain {
    assert!(log_size <= 32, "Goldilocks 2-adicity is 32, log_size must be <= 32");
    let modulus = goldilocks_modulus();
    let n = 1usize << log_size;

    let g = BigUint256::from_u64(7);
    let p_minus_1 = mod_sub(&modulus, &BigUint256::ONE, &modulus);
    let exp_max = shr_n(&p_minus_1, 32);
    let omega_max = mod_pow(&g, &exp_max, &modulus);

    let exp_adjust = if 32 - log_size < 64 {
        BigUint256::from_u64(1u64 << (32 - log_size))
    } else {
        // For log_size = 0, we need 2^32 which fits in u64
        BigUint256::from_u64(1u64 << 32)
    };
    let omega_n = mod_pow(&omega_max, &exp_adjust, &modulus);

    let twiddles = precompute_twiddles(&omega_n, n, &modulus);
    let omega_inv = mod_inv(&omega_n, &modulus);
    let inv_twiddles = precompute_twiddles(&omega_inv, n, &modulus);
    let size_inv = mod_inv(&BigUint256::from_u64(n as u64), &modulus);

    NTTDomain {
        modulus,
        root_of_unity: omega_n,
        two_adicity: 32,
        size: n,
        twiddles,
        inv_twiddles,
        size_inv,
    }
}

/// Shift a BigUint256 right by `n` bits.
fn shr_n(val: &BigUint256, n: u32) -> BigUint256 {
    let mut result = val.clone();
    for _ in 0..n {
        result = result.shr1();
    }
    result
}

/// Precompute twiddle factors: [ω^0, ω^1, ω^2, ..., ω^(n/2-1)]
pub fn precompute_twiddles(root: &BigUint256, n: usize, modulus: &BigUint256) -> Vec<BigUint256> {
    let half = n / 2;
    if half == 0 {
        return vec![BigUint256::ONE];
    }
    let mut twiddles = Vec::with_capacity(half);
    let mut w = BigUint256::ONE;
    for _ in 0..half {
        twiddles.push(w.clone());
        w = mod_mul(&w, root, modulus);
    }
    twiddles
}

/// In-place bit-reversal permutation on `data`.
pub fn bit_reverse(data: &mut [BigUint256], log_n: u32) {
    let n = data.len();
    assert_eq!(n, 1 << log_n, "data length must be 2^log_n");
    for i in 0..n {
        let j = reverse_bits(i as u32, log_n) as usize;
        if i < j {
            data.swap(i, j);
        }
    }
}

/// Reverse the lower `bits` bits of `x`.
fn reverse_bits(x: u32, bits: u32) -> u32 {
    let mut result = 0u32;
    let mut x = x;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Forward NTT (Cooley-Tukey decimation-in-time).
///
/// Transforms `data` in place from coefficient form to evaluation form.
/// `data` length must equal `domain.size`.
pub fn ntt(data: &mut [BigUint256], domain: &NTTDomain) {
    let n = domain.size;
    assert_eq!(data.len(), n, "data length must equal domain size");
    if n == 1 {
        return;
    }

    let log_n = n.trailing_zeros();
    bit_reverse(data, log_n);

    let modulus = &domain.modulus;
    let twiddles = &domain.twiddles;
    let mut half = 1usize;

    while half < n {
        let stride = n / (2 * half);
        for i in (0..n).step_by(2 * half) {
            for j in 0..half {
                let w = &twiddles[j * stride];
                let u = data[i + j].clone();
                let v = mod_mul(&data[i + j + half], w, modulus);
                data[i + j] = mod_add(&u, &v, modulus);
                data[i + j + half] = mod_sub(&u, &v, modulus);
            }
        }
        half *= 2;
    }
}

/// Inverse NTT.
///
/// Transforms `data` in place from evaluation form back to coefficient form.
/// Uses inverse twiddle factors and scales by n^(-1) at the end.
pub fn intt(data: &mut [BigUint256], domain: &NTTDomain) {
    let n = domain.size;
    assert_eq!(data.len(), n, "data length must equal domain size");
    if n == 1 {
        return;
    }

    let log_n = n.trailing_zeros();
    bit_reverse(data, log_n);

    let modulus = &domain.modulus;
    let inv_twiddles = &domain.inv_twiddles;
    let mut half = 1usize;

    while half < n {
        let stride = n / (2 * half);
        for i in (0..n).step_by(2 * half) {
            for j in 0..half {
                let w = &inv_twiddles[j * stride];
                let u = data[i + j].clone();
                let v = mod_mul(&data[i + j + half], w, modulus);
                data[i + j] = mod_add(&u, &v, modulus);
                data[i + j + half] = mod_sub(&u, &v, modulus);
            }
        }
        half *= 2;
    }

    // Scale by n^(-1)
    for elem in data.iter_mut() {
        *elem = mod_mul(elem, &domain.size_inv, modulus);
    }
}

/// Polynomial multiplication via NTT.
///
/// Computes the product of polynomials `a` and `b` (in coefficient form).
/// The domain size must be at least `a.len() + b.len() - 1`, rounded up to a power of 2.
pub fn poly_mul_ntt(a: &[BigUint256], b: &[BigUint256], domain: &NTTDomain) -> Vec<BigUint256> {
    let result_len = a.len() + b.len() - 1;
    let n = domain.size;
    assert!(n >= result_len, "domain size must be >= a.len() + b.len() - 1");

    let modulus = &domain.modulus;

    // Pad a and b to domain size
    let mut a_padded = Vec::with_capacity(n);
    a_padded.extend_from_slice(a);
    a_padded.resize(n, BigUint256::ZERO);

    let mut b_padded = Vec::with_capacity(n);
    b_padded.extend_from_slice(b);
    b_padded.resize(n, BigUint256::ZERO);

    // Forward NTT
    ntt(&mut a_padded, domain);
    ntt(&mut b_padded, domain);

    // Pointwise multiply
    let mut c = Vec::with_capacity(n);
    for i in 0..n {
        c.push(mod_mul(&a_padded[i], &b_padded[i], modulus));
    }

    // Inverse NTT
    intt(&mut c, domain);

    // Truncate to actual result length
    c.truncate(result_len);
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntt_intt_roundtrip() {
        let domain = bn254_fr_domain(4); // size 16
        let mut data: Vec<BigUint256> = (0..16).map(|i| BigUint256::from_u64(i + 1)).collect();
        let original = data.clone();
        ntt(&mut data, &domain);
        // After NTT, data should be different
        assert_ne!(data, original);
        intt(&mut data, &domain);
        // After INTT, data should match original
        assert_eq!(data, original);
    }

    #[test]
    fn test_poly_mul_ntt() {
        // Multiply (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
        let domain = bn254_fr_domain(2); // size 4
        let a = vec![BigUint256::from_u64(1), BigUint256::from_u64(2)];
        let b = vec![BigUint256::from_u64(3), BigUint256::from_u64(4)];
        let result = poly_mul_ntt(&a, &b, &domain);
        assert_eq!(result[0], BigUint256::from_u64(3));
        assert_eq!(result[1], BigUint256::from_u64(10));
        assert_eq!(result[2], BigUint256::from_u64(8));
    }

    #[test]
    fn test_ntt_size_8() {
        let domain = bn254_fr_domain(3); // size 8
        let mut data: Vec<BigUint256> = (0..8).map(|i| BigUint256::from_u64(i)).collect();
        let original = data.clone();
        ntt(&mut data, &domain);
        intt(&mut data, &domain);
        assert_eq!(data, original);
    }

    #[test]
    fn test_ntt_size_2() {
        let domain = bn254_fr_domain(1); // size 2
        let mut data = vec![BigUint256::from_u64(3), BigUint256::from_u64(7)];
        let original = data.clone();
        ntt(&mut data, &domain);
        assert_ne!(data, original);
        intt(&mut data, &domain);
        assert_eq!(data, original);
    }

    #[test]
    fn test_goldilocks_ntt_roundtrip() {
        let domain = goldilocks_domain(3); // size 8
        let mut data: Vec<BigUint256> = (0..8).map(|i| BigUint256::from_u64(i + 100)).collect();
        let original = data.clone();
        ntt(&mut data, &domain);
        intt(&mut data, &domain);
        assert_eq!(data, original);
    }

    #[test]
    fn test_bit_reverse_basic() {
        // For n=8, bit reversal of indices [0,1,2,3,4,5,6,7] -> [0,4,2,6,1,5,3,7]
        let mut data: Vec<BigUint256> = (0..8).map(|i| BigUint256::from_u64(i)).collect();
        bit_reverse(&mut data, 3);
        assert_eq!(data[0], BigUint256::from_u64(0));
        assert_eq!(data[1], BigUint256::from_u64(4));
        assert_eq!(data[2], BigUint256::from_u64(2));
        assert_eq!(data[3], BigUint256::from_u64(6));
        assert_eq!(data[4], BigUint256::from_u64(1));
        assert_eq!(data[5], BigUint256::from_u64(5));
        assert_eq!(data[6], BigUint256::from_u64(3));
        assert_eq!(data[7], BigUint256::from_u64(7));
    }

    #[test]
    fn test_root_of_unity_order() {
        // Verify that ω^n = 1 for BN254 domain of size 16
        let domain = bn254_fr_domain(4);
        let result = mod_pow(
            &domain.root_of_unity,
            &BigUint256::from_u64(16),
            &domain.modulus,
        );
        assert_eq!(result, BigUint256::ONE);
    }

    #[test]
    fn test_poly_mul_identity() {
        // Multiply (5) * (1) = (5) — constant polynomials
        let domain = bn254_fr_domain(1); // size 2
        let a = vec![BigUint256::from_u64(5)];
        let b = vec![BigUint256::from_u64(1)];
        let result = poly_mul_ntt(&a, &b, &domain);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], BigUint256::from_u64(5));
    }
}

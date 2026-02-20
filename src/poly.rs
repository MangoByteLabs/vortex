use crate::crypto::{BigUint256, mod_add, mod_sub, mod_mul, mod_inv};
use crate::ntt::{NTTDomain, ntt, intt, bn254_fr_domain};

/// Polynomial in coefficient form: coeffs[i] is the coefficient of x^i
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Polynomial {
    pub coeffs: Vec<BigUint256>,
    pub modulus: BigUint256, // the field modulus
}

impl Polynomial {
    pub fn new(coeffs: Vec<BigUint256>, modulus: BigUint256) -> Self {
        Self { coeffs, modulus }
    }

    pub fn zero(modulus: BigUint256) -> Self {
        Self { coeffs: vec![BigUint256::ZERO], modulus }
    }

    pub fn degree(&self) -> usize {
        // Find highest non-zero coefficient
        for i in (0..self.coeffs.len()).rev() {
            if !self.coeffs[i].is_zero() {
                return i;
            }
        }
        0
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| c.is_zero())
    }

    /// Evaluate polynomial at a point using Horner's method
    pub fn eval(&self, x: &BigUint256) -> BigUint256 {
        let mut result = BigUint256::ZERO;
        for i in (0..self.coeffs.len()).rev() {
            result = mod_mul(&result, x, &self.modulus);
            result = mod_add(&result, &self.coeffs[i], &self.modulus);
        }
        result
    }

    /// Add two polynomials
    pub fn add(&self, other: &Polynomial) -> Polynomial {
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut result = vec![BigUint256::ZERO; len];
        for i in 0..len {
            let a = if i < self.coeffs.len() { &self.coeffs[i] } else { &BigUint256::ZERO };
            let b = if i < other.coeffs.len() { &other.coeffs[i] } else { &BigUint256::ZERO };
            result[i] = mod_add(a, b, &self.modulus);
        }
        Polynomial::new(result, self.modulus.clone())
    }

    /// Subtract two polynomials
    pub fn sub(&self, other: &Polynomial) -> Polynomial {
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut result = vec![BigUint256::ZERO; len];
        for i in 0..len {
            let a = if i < self.coeffs.len() { &self.coeffs[i] } else { &BigUint256::ZERO };
            let b = if i < other.coeffs.len() { &other.coeffs[i] } else { &BigUint256::ZERO };
            result[i] = mod_sub(a, b, &self.modulus);
        }
        Polynomial::new(result, self.modulus.clone())
    }

    /// Multiply two polynomials using NTT
    pub fn mul_ntt(&self, other: &Polynomial, _domain: &NTTDomain) -> Polynomial {
        let result_len = self.coeffs.len() + other.coeffs.len() - 1;
        let n = result_len.next_power_of_two();
        let log_n = n.trailing_zeros();

        // Create a domain of appropriate size
        let domain = bn254_fr_domain(log_n);

        // Pad both polynomials to size n
        let mut a = self.coeffs.clone();
        a.resize(n, BigUint256::ZERO);
        let mut b = other.coeffs.clone();
        b.resize(n, BigUint256::ZERO);

        // Forward NTT
        ntt(&mut a, &domain);
        ntt(&mut b, &domain);

        // Pointwise multiply
        let mut c = vec![BigUint256::ZERO; n];
        for i in 0..n {
            c[i] = mod_mul(&a[i], &b[i], &self.modulus);
        }

        // Inverse NTT
        intt(&mut c, &domain);

        // Trim to result length
        c.truncate(result_len);

        Polynomial::new(c, self.modulus.clone())
    }

    /// Multiply two polynomials (schoolbook, for small polynomials)
    pub fn mul_schoolbook(&self, other: &Polynomial) -> Polynomial {
        if self.is_zero() || other.is_zero() {
            return Polynomial::zero(self.modulus.clone());
        }
        let result_len = self.coeffs.len() + other.coeffs.len() - 1;
        let mut result = vec![BigUint256::ZERO; result_len];
        for i in 0..self.coeffs.len() {
            for j in 0..other.coeffs.len() {
                let prod = mod_mul(&self.coeffs[i], &other.coeffs[j], &self.modulus);
                result[i + j] = mod_add(&result[i + j], &prod, &self.modulus);
            }
        }
        Polynomial::new(result, self.modulus.clone())
    }

    /// Scale polynomial by a constant
    pub fn scale(&self, c: &BigUint256) -> Polynomial {
        let coeffs: Vec<_> = self.coeffs.iter()
            .map(|a| mod_mul(a, c, &self.modulus))
            .collect();
        Polynomial::new(coeffs, self.modulus.clone())
    }

    /// Polynomial division: returns (quotient, remainder) such that self = quotient * divisor + remainder
    pub fn div(&self, divisor: &Polynomial) -> (Polynomial, Polynomial) {
        if divisor.is_zero() {
            panic!("polynomial division by zero");
        }

        let mut remainder = self.coeffs.clone();
        let d_deg = divisor.degree();
        let n_deg = self.degree();

        if n_deg < d_deg {
            return (Polynomial::zero(self.modulus.clone()), self.clone());
        }

        let q_len = n_deg - d_deg + 1;
        let mut quotient = vec![BigUint256::ZERO; q_len];

        let lead_inv = mod_inv(&divisor.coeffs[d_deg], &self.modulus);

        for i in (0..q_len).rev() {
            let idx = i + d_deg;
            if idx >= remainder.len() { continue; }
            let coeff = mod_mul(&remainder[idx], &lead_inv, &self.modulus);
            quotient[i] = coeff.clone();
            for j in 0..=d_deg {
                let sub = mod_mul(&coeff, &divisor.coeffs[j], &self.modulus);
                remainder[i + j] = mod_sub(&remainder[i + j], &sub, &self.modulus);
            }
        }

        // Trim remainder
        while remainder.len() > 1 && remainder.last().map_or(false, |c| c.is_zero()) {
            remainder.pop();
        }

        (
            Polynomial::new(quotient, self.modulus.clone()),
            Polynomial::new(remainder, self.modulus.clone()),
        )
    }

    /// Evaluate polynomial at all points in a domain (returns evaluations)
    pub fn eval_domain(&self, domain: &NTTDomain) -> Vec<BigUint256> {
        let n = domain.size;
        let mut evals = self.coeffs.clone();
        evals.resize(n, BigUint256::ZERO);
        ntt(&mut evals, domain);
        evals
    }

    /// Interpolate polynomial from evaluations on a domain
    pub fn interpolate(evals: &[BigUint256], domain: &NTTDomain, modulus: &BigUint256) -> Polynomial {
        let mut coeffs = evals.to_vec();
        coeffs.resize(domain.size, BigUint256::ZERO);
        intt(&mut coeffs, domain);
        // Trim trailing zeros
        while coeffs.len() > 1 && coeffs.last().map_or(false, |c| c.is_zero()) {
            coeffs.pop();
        }
        Polynomial::new(coeffs, modulus.clone())
    }

    /// Placeholder for Kate/KZG polynomial commitment (returns the polynomial itself for now)
    /// In a real implementation, this would compute MSM of coefficients with SRS points
    pub fn commit(&self) -> Vec<BigUint256> {
        // Placeholder: returns coefficients as the "commitment"
        // Real implementation: [f(s)]_1 = sum(coeffs[i] * [s^i]_1) via MSM
        self.coeffs.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::bn254_scalar_prime;

    fn test_modulus() -> BigUint256 {
        // Use BN254 scalar field
        bn254_scalar_prime()
    }

    #[test]
    fn test_poly_eval() {
        let m = test_modulus();
        // p(x) = 1 + 2x + 3x^2
        let p = Polynomial::new(
            vec![BigUint256::from_u64(1), BigUint256::from_u64(2), BigUint256::from_u64(3)],
            m.clone(),
        );
        // p(0) = 1
        assert_eq!(p.eval(&BigUint256::ZERO), BigUint256::from_u64(1));
        // p(1) = 1 + 2 + 3 = 6
        assert_eq!(p.eval(&BigUint256::ONE), BigUint256::from_u64(6));
        // p(2) = 1 + 4 + 12 = 17
        assert_eq!(p.eval(&BigUint256::from_u64(2)), BigUint256::from_u64(17));
    }

    #[test]
    fn test_poly_add() {
        let m = test_modulus();
        let a = Polynomial::new(vec![BigUint256::from_u64(1), BigUint256::from_u64(2)], m.clone());
        let b = Polynomial::new(vec![BigUint256::from_u64(3), BigUint256::from_u64(4)], m.clone());
        let c = a.add(&b);
        assert_eq!(c.coeffs[0], BigUint256::from_u64(4));
        assert_eq!(c.coeffs[1], BigUint256::from_u64(6));
    }

    #[test]
    fn test_poly_mul_schoolbook() {
        let m = test_modulus();
        // (1 + 2x)(3 + 4x) = 3 + 10x + 8x^2
        let a = Polynomial::new(vec![BigUint256::from_u64(1), BigUint256::from_u64(2)], m.clone());
        let b = Polynomial::new(vec![BigUint256::from_u64(3), BigUint256::from_u64(4)], m.clone());
        let c = a.mul_schoolbook(&b);
        assert_eq!(c.coeffs[0], BigUint256::from_u64(3));
        assert_eq!(c.coeffs[1], BigUint256::from_u64(10));
        assert_eq!(c.coeffs[2], BigUint256::from_u64(8));
    }

    #[test]
    fn test_poly_div() {
        let m = test_modulus();
        // (3 + 10x + 8x^2) / (1 + 2x) = (3 + 4x) remainder 0
        let dividend = Polynomial::new(
            vec![BigUint256::from_u64(3), BigUint256::from_u64(10), BigUint256::from_u64(8)],
            m.clone(),
        );
        let divisor = Polynomial::new(
            vec![BigUint256::from_u64(1), BigUint256::from_u64(2)],
            m.clone(),
        );
        let (q, r) = dividend.div(&divisor);
        // Verify by multiplication: q * divisor + r should equal dividend
        let reconstructed = q.mul_schoolbook(&divisor).add(&r);
        for i in 0..dividend.coeffs.len() {
            assert_eq!(
                reconstructed.coeffs.get(i).unwrap_or(&BigUint256::ZERO),
                &dividend.coeffs[i]
            );
        }
    }

    #[test]
    fn test_poly_degree() {
        let m = test_modulus();
        let p = Polynomial::new(
            vec![BigUint256::from_u64(1), BigUint256::ZERO, BigUint256::from_u64(3)],
            m,
        );
        assert_eq!(p.degree(), 2);
    }
}

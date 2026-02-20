use crate::crypto::{BigUint256, mod_add, mod_sub, mod_mul, mod_inv};

/// BN254 base field prime
pub fn bn254_field_prime() -> BigUint256 {
    BigUint256::from_hex("30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47")
        .unwrap()
}

/// BN254 scalar field (Fr) prime
pub fn bn254_scalar_prime() -> BigUint256 {
    BigUint256::from_hex("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001")
        .unwrap()
}

// ---------------------------------------------------------------------------
// Fp2 — Quadratic Extension (F_p^2 = F_p[u] / (u^2 + 1))
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fp2 {
    pub c0: BigUint256, // real part
    pub c1: BigUint256, // imaginary part (coefficient of u)
    pub p: BigUint256,  // the base field prime
}

impl Fp2 {
    pub fn new(c0: BigUint256, c1: BigUint256, p: BigUint256) -> Self {
        Self { c0, c1, p }
    }

    pub fn zero(p: BigUint256) -> Self {
        Self {
            c0: BigUint256::ZERO,
            c1: BigUint256::ZERO,
            p,
        }
    }

    pub fn one(p: BigUint256) -> Self {
        Self {
            c0: BigUint256::ONE,
            c1: BigUint256::ZERO,
            p,
        }
    }

    pub fn add(&self, other: &Fp2) -> Fp2 {
        Fp2 {
            c0: mod_add(&self.c0, &other.c0, &self.p),
            c1: mod_add(&self.c1, &other.c1, &self.p),
            p: self.p.clone(),
        }
    }

    pub fn sub(&self, other: &Fp2) -> Fp2 {
        Fp2 {
            c0: mod_sub(&self.c0, &other.c0, &self.p),
            c1: mod_sub(&self.c1, &other.c1, &self.p),
            p: self.p.clone(),
        }
    }

    pub fn neg(&self) -> Fp2 {
        Fp2 {
            c0: if self.c0.is_zero() {
                BigUint256::ZERO
            } else {
                mod_sub(&self.p, &self.c0, &self.p)
            },
            c1: if self.c1.is_zero() {
                BigUint256::ZERO
            } else {
                mod_sub(&self.p, &self.c1, &self.p)
            },
            p: self.p.clone(),
        }
    }

    /// Karatsuba multiplication: (a+bi)(c+di) with β = -1
    /// v0 = a*c, v1 = b*d
    /// real = v0 - v1, imag = (a+b)(c+d) - v0 - v1
    pub fn mul(&self, other: &Fp2) -> Fp2 {
        let v0 = mod_mul(&self.c0, &other.c0, &self.p);
        let v1 = mod_mul(&self.c1, &other.c1, &self.p);
        let c0_new = mod_sub(&v0, &v1, &self.p);
        let s1 = mod_add(&self.c0, &self.c1, &self.p);
        let s2 = mod_add(&other.c0, &other.c1, &self.p);
        let c1_new = mod_sub(&mod_sub(&mod_mul(&s1, &s2, &self.p), &v0, &self.p), &v1, &self.p);
        Fp2 {
            c0: c0_new,
            c1: c1_new,
            p: self.p.clone(),
        }
    }

    /// Optimized squaring: real = (a+b)(a-b), imag = 2ab
    pub fn square(&self) -> Fp2 {
        let ab = mod_mul(&self.c0, &self.c1, &self.p);
        let a_plus_b = mod_add(&self.c0, &self.c1, &self.p);
        let a_minus_b = mod_sub(&self.c0, &self.c1, &self.p);
        Fp2 {
            c0: mod_mul(&a_plus_b, &a_minus_b, &self.p),
            c1: mod_add(&ab, &ab, &self.p),
            p: self.p.clone(),
        }
    }

    /// Inverse: (a - bi) / (a^2 + b^2)
    pub fn inv(&self) -> Fp2 {
        let a2 = mod_mul(&self.c0, &self.c0, &self.p);
        let b2 = mod_mul(&self.c1, &self.c1, &self.p);
        let t = mod_add(&a2, &b2, &self.p);
        let t_inv = mod_inv(&t, &self.p);
        let c0_new = mod_mul(&self.c0, &t_inv, &self.p);
        let c1_neg = mod_mul(&self.c1, &t_inv, &self.p);
        let c1_new = if c1_neg.is_zero() {
            BigUint256::ZERO
        } else {
            mod_sub(&self.p, &c1_neg, &self.p)
        };
        Fp2 {
            c0: c0_new,
            c1: c1_new,
            p: self.p.clone(),
        }
    }

    pub fn conjugate(&self) -> Fp2 {
        Fp2 {
            c0: self.c0.clone(),
            c1: if self.c1.is_zero() {
                BigUint256::ZERO
            } else {
                mod_sub(&self.p, &self.c1, &self.p)
            },
            p: self.p.clone(),
        }
    }

    /// Multiply by non-residue (9 + u) for BN254 tower.
    /// (9+u)(a+bu) = (9a - b) + (a + 9b)u
    pub fn mul_by_nonresidue(&self) -> Fp2 {
        let nine = BigUint256::from_u64(9);
        let nine_a = mod_mul(&nine, &self.c0, &self.p);
        let nine_b = mod_mul(&nine, &self.c1, &self.p);
        Fp2 {
            c0: mod_sub(&nine_a, &self.c1, &self.p),
            c1: mod_add(&self.c0, &nine_b, &self.p),
            p: self.p.clone(),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }

    pub fn scale(&self, scalar: &BigUint256) -> Fp2 {
        Fp2 {
            c0: mod_mul(&self.c0, scalar, &self.p),
            c1: mod_mul(&self.c1, scalar, &self.p),
            p: self.p.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Fp6 — Cubic Extension over Fp2 (F_p^6 = F_p^2[v] / (v^3 - ξ))
// For BN254, ξ = 9 + u
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fp6 {
    pub c0: Fp2,
    pub c1: Fp2,
    pub c2: Fp2,
}

impl Fp6 {
    pub fn new(c0: Fp2, c1: Fp2, c2: Fp2) -> Self {
        Self { c0, c1, c2 }
    }

    pub fn zero(p: BigUint256) -> Self {
        Self {
            c0: Fp2::zero(p.clone()),
            c1: Fp2::zero(p.clone()),
            c2: Fp2::zero(p),
        }
    }

    pub fn one(p: BigUint256) -> Self {
        Self {
            c0: Fp2::one(p.clone()),
            c1: Fp2::zero(p.clone()),
            c2: Fp2::zero(p),
        }
    }

    fn p(&self) -> BigUint256 {
        self.c0.p.clone()
    }

    pub fn add(&self, other: &Fp6) -> Fp6 {
        Fp6 {
            c0: self.c0.add(&other.c0),
            c1: self.c1.add(&other.c1),
            c2: self.c2.add(&other.c2),
        }
    }

    pub fn sub(&self, other: &Fp6) -> Fp6 {
        Fp6 {
            c0: self.c0.sub(&other.c0),
            c1: self.c1.sub(&other.c1),
            c2: self.c2.sub(&other.c2),
        }
    }

    pub fn neg(&self) -> Fp6 {
        Fp6 {
            c0: self.c0.neg(),
            c1: self.c1.neg(),
            c2: self.c2.neg(),
        }
    }

    /// Schoolbook multiplication with non-residue reduction.
    pub fn mul(&self, other: &Fp6) -> Fp6 {
        let v0 = self.c0.mul(&other.c0);
        let v1 = self.c1.mul(&other.c1);
        let v2 = self.c2.mul(&other.c2);

        // c0 = v0 + ((c1+c2)(o1+o2) - v1 - v2) * ξ
        let t0 = self.c1.add(&self.c2).mul(&other.c1.add(&other.c2));
        let c0 = v0.add(&t0.sub(&v1).sub(&v2).mul_by_nonresidue());

        // c1 = (c0+c1)(o0+o1) - v0 - v1 + v2*ξ
        let t1 = self.c0.add(&self.c1).mul(&other.c0.add(&other.c1));
        let c1 = t1.sub(&v0).sub(&v1).add(&v2.mul_by_nonresidue());

        // c2 = (c0+c2)(o0+o2) - v0 + v1 - v2
        let t2 = self.c0.add(&self.c2).mul(&other.c0.add(&other.c2));
        let c2 = t2.sub(&v0).add(&v1).sub(&v2);

        Fp6 { c0, c1, c2 }
    }

    /// Squaring via schoolbook (same structure as mul with self).
    pub fn square(&self) -> Fp6 {
        self.mul(self)
    }

    /// Inverse of cubic extension element.
    /// Uses: a^-1 = adj(a) / det(a) where
    ///   A = c0^2 - c1*c2*ξ
    ///   B = c2^2*ξ - c0*c1
    ///   C = c1^2 - c0*c2
    ///   det = c0*A + c2*B*ξ + c1*C*ξ  ... actually:
    ///   det = c0*A + c1*(c2^2*ξ - c0*c1)*...
    /// Standard formula: det = c0^3 + c1^3*ξ + c2^3*ξ^2 - 3*c0*c1*c2*ξ
    /// Simpler: use cofactor approach.
    pub fn inv(&self) -> Fp6 {
        // Cofactors:
        //   A = c0^2 - c1*c2*ξ  ... no, standard cubic inverse:
        // For x = a + bv + cv^2 in Fp2[v]/(v^3 - ξ):
        //   t0 = a^2 - b*c*ξ  ... this is wrong for general cubic.
        // Correct cofactor inverse for Fp6 = Fp2[v]/(v^3 - ξ):
        //   s0 = c0^2 - c1*c2.mul_by_nonresidue()
        //   s1 = c2^2.mul_by_nonresidue() - c0*c1
        //   s2 = c1^2 - c0*c2
        //   det = c0*s0 + c1*s2.mul_by_nonresidue() + c2*s1.mul_by_nonresidue()
        //   ... actually the standard formula:
        //   det = c0*s0 + (c2*s1 + c1*s2).mul_by_nonresidue()

        let c0s = self.c0.square();
        let c1s = self.c1.square();
        let c2s = self.c2.square();
        let c01 = self.c0.mul(&self.c1);
        let c02 = self.c0.mul(&self.c2);
        let c12 = self.c1.mul(&self.c2);

        let s0 = c0s.sub(&c12.mul_by_nonresidue());
        let s1 = c2s.mul_by_nonresidue().sub(&c01);
        let s2 = c1s.sub(&c02);

        // det = c0*s0 + (c2*s1 + c1*s2).mul_by_nonresidue()
        let det = self.c0.mul(&s0).add(
            &self.c2.mul(&s1).add(&self.c1.mul(&s2)).mul_by_nonresidue(),
        );
        let det_inv = det.inv();

        Fp6 {
            c0: s0.mul(&det_inv),
            c1: s1.mul(&det_inv),
            c2: s2.mul(&det_inv),
        }
    }

    /// Shift: (c0, c1, c2) -> (c2*ξ, c0, c1)
    pub fn mul_by_nonresidue(&self) -> Fp6 {
        Fp6 {
            c0: self.c2.mul_by_nonresidue(),
            c1: self.c0.clone(),
            c2: self.c1.clone(),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero() && self.c2.is_zero()
    }
}

// ---------------------------------------------------------------------------
// Fp12 — Quadratic Extension over Fp6 (F_p^12 = F_p^6[w] / (w^2 - v))
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fp12 {
    pub c0: Fp6,
    pub c1: Fp6,
}

impl Fp12 {
    pub fn new(c0: Fp6, c1: Fp6) -> Self {
        Self { c0, c1 }
    }

    pub fn zero(p: BigUint256) -> Self {
        Self {
            c0: Fp6::zero(p.clone()),
            c1: Fp6::zero(p),
        }
    }

    pub fn one(p: BigUint256) -> Self {
        Self {
            c0: Fp6::one(p.clone()),
            c1: Fp6::zero(p),
        }
    }

    fn p(&self) -> BigUint256 {
        self.c0.p()
    }

    pub fn add(&self, other: &Fp12) -> Fp12 {
        Fp12 {
            c0: self.c0.add(&other.c0),
            c1: self.c1.add(&other.c1),
        }
    }

    pub fn sub(&self, other: &Fp12) -> Fp12 {
        Fp12 {
            c0: self.c0.sub(&other.c0),
            c1: self.c1.sub(&other.c1),
        }
    }

    pub fn neg(&self) -> Fp12 {
        Fp12 {
            c0: self.c0.neg(),
            c1: self.c1.neg(),
        }
    }

    /// Karatsuba multiplication over Fp6.
    /// w^2 = v, so v1 * w^2 = v1 shifted via mul_by_nonresidue on Fp6.
    pub fn mul(&self, other: &Fp12) -> Fp12 {
        let v0 = self.c0.mul(&other.c0);
        let v1 = self.c1.mul(&other.c1);
        let c0_new = v0.add(&v1.mul_by_nonresidue());
        let c1_new = self
            .c0
            .add(&self.c1)
            .mul(&other.c0.add(&other.c1))
            .sub(&v0)
            .sub(&v1);
        Fp12 {
            c0: c0_new,
            c1: c1_new,
        }
    }

    /// Complex squaring: c0' = c0^2 + c1^2*nonresidue, c1' = 2*c0*c1
    pub fn square(&self) -> Fp12 {
        let ab = self.c0.mul(&self.c1);
        let a_plus_b = self.c0.add(&self.c1);
        let a_plus_b_nr = self.c0.add(&self.c1.mul_by_nonresidue());
        // c0 = (c0 + c1)(c0 + c1*nr) - ab - ab*nr
        let c0_new = a_plus_b_nr
            .mul(&a_plus_b)
            .sub(&ab)
            .sub(&ab.mul_by_nonresidue());
        // c1 = 2*ab
        let c1_new = ab.add(&ab);
        Fp12 {
            c0: c0_new,
            c1: c1_new,
        }
    }

    /// Inverse: 1/(c0 + c1*w) = (c0 - c1*w) / (c0^2 - c1^2 * v)
    pub fn inv(&self) -> Fp12 {
        let c0s = self.c0.square();
        let c1s = self.c1.square();
        let det = c0s.sub(&c1s.mul_by_nonresidue());
        let det_inv = det.inv();
        Fp12 {
            c0: self.c0.mul(&det_inv),
            c1: self.c1.neg().mul(&det_inv),
        }
    }

    pub fn conjugate(&self) -> Fp12 {
        Fp12 {
            c0: self.c0.clone(),
            c1: self.c1.neg(),
        }
    }

    /// Cyclotomic squaring (placeholder — uses regular square).
    pub fn cyclotomic_square(&self) -> Fp12 {
        self.square()
    }

    /// Frobenius map p^power (placeholder — returns self).
    pub fn frobenius_map(&self, _power: usize) -> Fp12 {
        self.clone()
    }

    pub fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }

    pub fn is_one(&self) -> bool {
        let p = self.p();
        self.c0 == Fp6::one(p.clone()) && self.c1 == Fp6::zero(p)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> BigUint256 {
        bn254_field_prime()
    }

    #[test]
    fn test_fp2_mul() {
        // (1 + 2i)(3 + 4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
        let a = Fp2::new(BigUint256::from_u64(1), BigUint256::from_u64(2), p());
        let b = Fp2::new(BigUint256::from_u64(3), BigUint256::from_u64(4), p());
        let c = a.mul(&b);
        // -5 mod p = p - 5
        let expected_c0 = mod_sub(&p(), &BigUint256::from_u64(5), &p());
        let expected_c1 = BigUint256::from_u64(10);
        assert_eq!(c.c0, expected_c0);
        assert_eq!(c.c1, expected_c1);
    }

    #[test]
    fn test_fp2_square() {
        let a = Fp2::new(BigUint256::from_u64(3), BigUint256::from_u64(7), p());
        let sq = a.square();
        let mul = a.mul(&a);
        assert_eq!(sq, mul);
    }

    #[test]
    fn test_fp2_inv() {
        let a = Fp2::new(BigUint256::from_u64(5), BigUint256::from_u64(12), p());
        let a_inv = a.inv();
        let product = a.mul(&a_inv);
        assert_eq!(product, Fp2::one(p()));
    }

    #[test]
    fn test_fp2_conjugate() {
        let a = Fp2::new(BigUint256::from_u64(3), BigUint256::from_u64(7), p());
        let conj = a.conjugate();
        let product = a.mul(&conj);
        // a * conj(a) = (a^2 + b^2, 0) — purely real
        assert_eq!(product.c1, BigUint256::ZERO);
    }

    #[test]
    fn test_fp6_mul() {
        let pp = p();
        let a = Fp6::new(
            Fp2::new(BigUint256::from_u64(1), BigUint256::from_u64(2), pp.clone()),
            Fp2::new(BigUint256::from_u64(3), BigUint256::from_u64(4), pp.clone()),
            Fp2::new(BigUint256::from_u64(5), BigUint256::from_u64(6), pp.clone()),
        );
        let one = Fp6::one(pp.clone());
        let result = a.mul(&one);
        assert_eq!(result, a);
    }

    #[test]
    fn test_fp6_inv() {
        let pp = p();
        let a = Fp6::new(
            Fp2::new(BigUint256::from_u64(1), BigUint256::from_u64(2), pp.clone()),
            Fp2::new(BigUint256::from_u64(3), BigUint256::from_u64(4), pp.clone()),
            Fp2::new(BigUint256::from_u64(5), BigUint256::from_u64(6), pp.clone()),
        );
        let a_inv = a.inv();
        let product = a.mul(&a_inv);
        assert_eq!(product, Fp6::one(pp));
    }

    #[test]
    fn test_fp12_mul() {
        let pp = p();
        let a = Fp12::new(
            Fp6::new(
                Fp2::new(BigUint256::from_u64(1), BigUint256::from_u64(2), pp.clone()),
                Fp2::new(BigUint256::from_u64(3), BigUint256::from_u64(4), pp.clone()),
                Fp2::new(BigUint256::from_u64(5), BigUint256::from_u64(6), pp.clone()),
            ),
            Fp6::new(
                Fp2::new(BigUint256::from_u64(7), BigUint256::from_u64(8), pp.clone()),
                Fp2::new(BigUint256::from_u64(9), BigUint256::from_u64(10), pp.clone()),
                Fp2::new(BigUint256::from_u64(11), BigUint256::from_u64(12), pp.clone()),
            ),
        );
        let one = Fp12::one(pp.clone());
        let result = a.mul(&one);
        assert_eq!(result, a);
    }

    #[test]
    fn test_fp12_inv() {
        let pp = p();
        let a = Fp12::new(
            Fp6::new(
                Fp2::new(BigUint256::from_u64(1), BigUint256::from_u64(2), pp.clone()),
                Fp2::new(BigUint256::from_u64(3), BigUint256::from_u64(4), pp.clone()),
                Fp2::new(BigUint256::from_u64(5), BigUint256::from_u64(6), pp.clone()),
            ),
            Fp6::new(
                Fp2::new(BigUint256::from_u64(7), BigUint256::from_u64(8), pp.clone()),
                Fp2::new(BigUint256::from_u64(9), BigUint256::from_u64(10), pp.clone()),
                Fp2::new(BigUint256::from_u64(11), BigUint256::from_u64(12), pp.clone()),
            ),
        );
        let a_inv = a.inv();
        let product = a.mul(&a_inv);
        assert_eq!(product, Fp12::one(pp));
    }
}

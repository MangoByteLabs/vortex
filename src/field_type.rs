//! Field<P> type for prime field arithmetic.
//! Represents elements of Z/pZ with compile-time prime modulus.

/// A prime field element
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldElement {
    pub value: u64,
    pub modulus: u64,
}

impl FieldElement {
    pub fn new(value: u64, modulus: u64) -> Self {
        Self { value: value % modulus, modulus }
    }

    pub fn add(self, other: Self) -> Self {
        assert_eq!(self.modulus, other.modulus, "Field modulus mismatch");
        Self::new((self.value + other.value) % self.modulus, self.modulus)
    }

    pub fn sub(self, other: Self) -> Self {
        assert_eq!(self.modulus, other.modulus);
        let v = if self.value >= other.value {
            self.value - other.value
        } else {
            self.modulus - (other.value - self.value)
        };
        Self::new(v, self.modulus)
    }

    pub fn mul(self, other: Self) -> Self {
        assert_eq!(self.modulus, other.modulus);
        // Use u128 to avoid overflow
        let v = ((self.value as u128 * other.value as u128) % self.modulus as u128) as u64;
        Self::new(v, self.modulus)
    }

    /// Modular inverse via Fermat's little theorem (p must be prime)
    pub fn inv(self) -> Option<Self> {
        if self.value == 0 { return None; }
        Some(Self::new(self.pow(self.modulus - 2), self.modulus))
    }

    pub fn div(self, other: Self) -> Option<Self> {
        other.inv().map(|inv| self.mul(inv))
    }

    fn pow(self, mut exp: u64) -> u64 {
        let mut base = self.value;
        let mut result = 1u64;
        let m = self.modulus;
        while exp > 0 {
            if exp & 1 == 1 {
                result = ((result as u128 * base as u128) % m as u128) as u64;
            }
            base = ((base as u128 * base as u128) % m as u128) as u64;
            exp >>= 1;
        }
        result
    }

    pub fn zero(modulus: u64) -> Self { Self::new(0, modulus) }
    pub fn one(modulus: u64) -> Self { Self::new(1, modulus) }
}

impl std::fmt::Display for FieldElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}_{}", self.value, self.modulus)
    }
}

/// Well-known prime field moduli
pub mod primes {
    /// BN254 scalar field prime (254-bit; too large for u64/u128 — stored as string)
    pub const BN254_FR_STR: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495617";
    pub const GOLDILOCKS: u64 = 18446744069414584321; // 2^64 - 2^32 + 1
    pub const MERSENNE31: u64 = 2147483647; // 2^31 - 1
    pub const BABY_BEAR: u64 = 2013265921; // 2^31 - 2^27 + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_add() {
        let p = primes::MERSENNE31;
        let a = FieldElement::new(10, p);
        let b = FieldElement::new(20, p);
        assert_eq!(a.add(b).value, 30);
    }

    #[test]
    fn test_field_add_wrap() {
        let p = 7u64;
        let a = FieldElement::new(5, p);
        let b = FieldElement::new(4, p);
        assert_eq!(a.add(b).value, 2); // (5+4) % 7 = 2
    }

    #[test]
    fn test_field_mul() {
        let p = 7u64;
        let a = FieldElement::new(3, p);
        let b = FieldElement::new(4, p);
        assert_eq!(a.mul(b).value, 5); // 12 % 7 = 5
    }

    #[test]
    fn test_field_inv() {
        let p = 7u64;
        let a = FieldElement::new(3, p);
        let inv = a.inv().unwrap();
        assert_eq!(a.mul(inv).value, 1);
    }

    #[test]
    fn test_field_inv_zero() {
        let a = FieldElement::new(0, 7);
        assert!(a.inv().is_none());
    }

    #[test]
    fn test_field_sub_wrap() {
        let p = 7u64;
        let a = FieldElement::new(2, p);
        let b = FieldElement::new(5, p);
        assert_eq!(a.sub(b).value, 4); // (2-5) mod 7 = 4
    }

    #[test]
    fn test_field_zero_one() {
        let p = 7u64;
        assert_eq!(FieldElement::zero(p).value, 0);
        assert_eq!(FieldElement::one(p).value, 1);
    }

    #[test]
    fn test_field_div() {
        let p = 7u64;
        let a = FieldElement::new(6, p);
        let b = FieldElement::new(2, p);
        let result = a.div(b).unwrap();
        assert_eq!(result.value, 3); // 6/2 = 3
    }

    #[test]
    fn test_field_display() {
        let a = FieldElement::new(5, 7);
        assert_eq!(format!("{}", a), "5_7");
    }
}

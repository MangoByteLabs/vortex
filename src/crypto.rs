use std::fmt;

/// 256-bit unsigned integer, stored as 8 x 32-bit limbs (little-endian)
#[derive(Clone, PartialEq, Eq)]
pub struct BigUint256 {
    pub limbs: [u32; 8],
}

impl BigUint256 {
    pub const ZERO: BigUint256 = BigUint256 { limbs: [0; 8] };
    pub const ONE: BigUint256 = BigUint256 { limbs: [1, 0, 0, 0, 0, 0, 0, 0] };

    pub fn new(limbs: [u32; 8]) -> Self {
        Self { limbs }
    }

    pub fn from_u64(val: u64) -> Self {
        let mut limbs = [0u32; 8];
        limbs[0] = val as u32;
        limbs[1] = (val >> 32) as u32;
        Self { limbs }
    }

    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.strip_prefix("0x").or_else(|| hex.strip_prefix("0X")).unwrap_or(hex);
        let hex = hex.trim_start_matches('0');
        if hex.is_empty() {
            return Some(Self::ZERO);
        }
        if hex.len() > 64 {
            return None; // Too large for 256 bits
        }
        // Pad to 64 hex chars
        let padded = format!("{:0>64}", hex);
        let mut limbs = [0u32; 8];
        for i in 0..8 {
            let start = 64 - (i + 1) * 8;
            let end = start + 8;
            limbs[i] = u32::from_str_radix(&padded[start..end], 16).ok()?;
        }
        Some(Self { limbs })
    }

    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&l| l == 0)
    }

    pub fn bit(&self, n: u32) -> bool {
        if n >= 256 {
            return false;
        }
        let limb_idx = (n / 32) as usize;
        let bit_idx = n % 32;
        (self.limbs[limb_idx] >> bit_idx) & 1 == 1
    }

    pub fn bits(&self) -> u32 {
        for i in (0..8).rev() {
            if self.limbs[i] != 0 {
                return i as u32 * 32 + (32 - self.limbs[i].leading_zeros());
            }
        }
        0
    }

    /// Add two 256-bit numbers, returning (result, carry)
    pub fn add_with_carry(&self, other: &BigUint256) -> (BigUint256, bool) {
        let mut result = [0u32; 8];
        let mut carry: u64 = 0;
        for i in 0..8 {
            let sum = self.limbs[i] as u64 + other.limbs[i] as u64 + carry;
            result[i] = sum as u32;
            carry = sum >> 32;
        }
        (BigUint256 { limbs: result }, carry != 0)
    }

    /// Subtract: self - other, returning (result, borrow)
    pub fn sub_with_borrow(&self, other: &BigUint256) -> (BigUint256, bool) {
        let mut result = [0u32; 8];
        let mut borrow: i64 = 0;
        for i in 0..8 {
            let diff = self.limbs[i] as i64 - other.limbs[i] as i64 - borrow;
            if diff < 0 {
                result[i] = (diff + (1i64 << 32)) as u32;
                borrow = 1;
            } else {
                result[i] = diff as u32;
                borrow = 0;
            }
        }
        (BigUint256 { limbs: result }, borrow != 0)
    }

    /// Full 512-bit multiplication, returns (low 256 bits, high 256 bits)
    pub fn mul_wide(&self, other: &BigUint256) -> (BigUint256, BigUint256) {
        let mut result = [0u64; 16];
        for i in 0..8 {
            let mut carry: u64 = 0;
            for j in 0..8 {
                let prod = self.limbs[i] as u64 * other.limbs[j] as u64 + result[i + j] + carry;
                result[i + j] = prod & 0xFFFFFFFF;
                carry = prod >> 32;
            }
            result[i + 8] += carry;
        }
        let mut lo = [0u32; 8];
        let mut hi = [0u32; 8];
        for i in 0..8 {
            lo[i] = result[i] as u32;
            hi[i] = result[i + 8] as u32;
        }
        (BigUint256 { limbs: lo }, BigUint256 { limbs: hi })
    }

    /// Compare: returns -1, 0, or 1
    pub fn cmp(&self, other: &BigUint256) -> std::cmp::Ordering {
        for i in (0..8).rev() {
            match self.limbs[i].cmp(&other.limbs[i]) {
                std::cmp::Ordering::Equal => continue,
                ord => return ord,
            }
        }
        std::cmp::Ordering::Equal
    }

    pub fn is_even(&self) -> bool {
        self.limbs[0] & 1 == 0
    }

    /// Shift right by 1 bit
    pub fn shr1(&self) -> BigUint256 {
        let mut result = [0u32; 8];
        for i in 0..7 {
            result[i] = (self.limbs[i] >> 1) | (self.limbs[i + 1] << 31);
        }
        result[7] = self.limbs[7] >> 1;
        BigUint256 { limbs: result }
    }
}

impl fmt::Display for BigUint256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print as hex without leading zeros
        let mut started = false;
        for i in (0..8).rev() {
            if self.limbs[i] != 0 || started || i == 0 {
                if started {
                    write!(f, "{:08x}", self.limbs[i])?;
                } else {
                    write!(f, "{:x}", self.limbs[i])?;
                    started = true;
                }
            }
        }
        if !started {
            write!(f, "0")?;
        }
        Ok(())
    }
}

impl fmt::Debug for BigUint256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BigUint256(0x{})", self)
    }
}

impl PartialOrd for BigUint256 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigUint256 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        BigUint256::cmp(self, other)
    }
}

// --- Modular arithmetic ---

/// Modular addition: (a + b) mod p
pub fn mod_add(a: &BigUint256, b: &BigUint256, p: &BigUint256) -> BigUint256 {
    let (sum, carry) = a.add_with_carry(b);
    if carry || sum.cmp(p) != std::cmp::Ordering::Less {
        let (result, _) = sum.sub_with_borrow(p);
        result
    } else {
        sum
    }
}

/// Modular subtraction: (a - b) mod p
pub fn mod_sub(a: &BigUint256, b: &BigUint256, p: &BigUint256) -> BigUint256 {
    let (diff, borrow) = a.sub_with_borrow(b);
    if borrow {
        let (result, _) = diff.add_with_carry(p);
        result
    } else {
        diff
    }
}

/// Modular multiplication: (a * b) mod p using Barrett-like reduction
pub fn mod_mul(a: &BigUint256, b: &BigUint256, p: &BigUint256) -> BigUint256 {
    let (lo, hi) = a.mul_wide(b);
    // Use schoolbook reduction: divide 512-bit product by p
    mod_reduce_512(&lo, &hi, p)
}

/// Reduce a 512-bit number (lo, hi) mod p
fn mod_reduce_512(lo: &BigUint256, hi: &BigUint256, p: &BigUint256) -> BigUint256 {
    // Simple repeated subtraction for correctness (not fast, but correct for Phase 0)
    // We'll implement proper Montgomery multiplication later for GPU
    let mut result = [0u32; 16];
    for i in 0..8 {
        result[i] = lo.limbs[i];
        result[i + 8] = hi.limbs[i];
    }

    // Long division: shift and subtract
    let p_bits = p.bits() as usize;
    if p_bits == 0 {
        return BigUint256::ZERO;
    }

    // Find highest set bit in the 512-bit number
    let mut total_bits = 0;
    for i in (0..16).rev() {
        if result[i] != 0 {
            total_bits = i * 32 + (32 - result[i].leading_zeros() as usize);
            break;
        }
    }

    if total_bits == 0 {
        return BigUint256::ZERO;
    }

    // Extended p to 512 bits
    let mut p_ext = [0u32; 16];
    for i in 0..8 {
        p_ext[i] = p.limbs[i];
    }

    // Shift p left so its MSB aligns with result's MSB
    let shift = if total_bits > p_bits { total_bits - p_bits } else { 0 };

    // Shift p_ext left by `shift` bits
    let mut shifted_p = [0u32; 16];
    let word_shift = shift / 32;
    let bit_shift = shift % 32;

    for i in 0..16 {
        if i >= word_shift {
            let src = i - word_shift;
            if src < 16 {
                shifted_p[i] |= p_ext[src] << bit_shift;
            }
            if bit_shift > 0 && src > 0 && (src - 1) < 16 {
                shifted_p[i] |= p_ext[src - 1] >> (32 - bit_shift);
            }
        }
    }

    // Repeatedly subtract shifted_p from result
    for _ in 0..=shift {
        // Compare result >= shifted_p
        let mut ge = true;
        for i in (0..16).rev() {
            if result[i] > shifted_p[i] {
                break;
            } else if result[i] < shifted_p[i] {
                ge = false;
                break;
            }
        }

        if ge {
            let mut borrow: i64 = 0;
            for i in 0..16 {
                let diff = result[i] as i64 - shifted_p[i] as i64 - borrow;
                if diff < 0 {
                    result[i] = (diff + (1i64 << 32)) as u32;
                    borrow = 1;
                } else {
                    result[i] = diff as u32;
                    borrow = 0;
                }
            }
        }

        // Shift shifted_p right by 1
        for i in 0..15 {
            shifted_p[i] = (shifted_p[i] >> 1) | (shifted_p[i + 1] << 31);
        }
        shifted_p[15] >>= 1;
    }

    let mut out = [0u32; 8];
    for i in 0..8 {
        out[i] = result[i];
    }
    BigUint256 { limbs: out }
}

/// Modular exponentiation: base^exp mod p (binary method)
pub fn mod_pow(base: &BigUint256, exp: &BigUint256, p: &BigUint256) -> BigUint256 {
    if exp.is_zero() {
        return BigUint256::ONE;
    }

    let mut result = BigUint256::ONE;
    let mut base = mod_mul(base, &BigUint256::ONE, p); // Reduce base mod p

    let bits = exp.bits();
    for i in 0..bits {
        if exp.bit(i) {
            result = mod_mul(&result, &base, p);
        }
        base = mod_mul(&base, &base, p);
    }
    result
}

/// Modular inverse: a^(-1) mod p using Fermat's little theorem: a^(p-2) mod p
pub fn mod_inv(a: &BigUint256, p: &BigUint256) -> BigUint256 {
    let (p_minus_2, _) = p.sub_with_borrow(&BigUint256::from_u64(2));
    mod_pow(a, &p_minus_2, p)
}

// --- Field Element ---

#[derive(Clone, PartialEq, Eq)]
pub struct FieldElement {
    pub value: BigUint256,
    pub modulus: BigUint256,
}

impl FieldElement {
    pub fn new(value: BigUint256, modulus: BigUint256) -> Self {
        let value = mod_mul(&value, &BigUint256::ONE, &modulus);
        Self { value, modulus }
    }

    pub fn zero(modulus: BigUint256) -> Self {
        Self { value: BigUint256::ZERO, modulus }
    }

    pub fn one(modulus: BigUint256) -> Self {
        Self { value: BigUint256::ONE, modulus }
    }

    pub fn add(&self, other: &FieldElement) -> FieldElement {
        FieldElement {
            value: mod_add(&self.value, &other.value, &self.modulus),
            modulus: self.modulus.clone(),
        }
    }

    pub fn sub(&self, other: &FieldElement) -> FieldElement {
        FieldElement {
            value: mod_sub(&self.value, &other.value, &self.modulus),
            modulus: self.modulus.clone(),
        }
    }

    pub fn mul(&self, other: &FieldElement) -> FieldElement {
        FieldElement {
            value: mod_mul(&self.value, &other.value, &self.modulus),
            modulus: self.modulus.clone(),
        }
    }

    pub fn inv(&self) -> FieldElement {
        FieldElement {
            value: mod_inv(&self.value, &self.modulus),
            modulus: self.modulus.clone(),
        }
    }

    pub fn pow(&self, exp: &BigUint256) -> FieldElement {
        FieldElement {
            value: mod_pow(&self.value, exp, &self.modulus),
            modulus: self.modulus.clone(),
        }
    }

    pub fn neg(&self) -> FieldElement {
        FieldElement {
            value: mod_sub(&BigUint256::ZERO, &self.value, &self.modulus),
            modulus: self.modulus.clone(),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
}

impl fmt::Display for FieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{}", self.value)
    }
}

impl fmt::Debug for FieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FieldElement(0x{})", self.value)
    }
}

// --- secp256k1 curve ---

/// secp256k1 field prime: p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
pub fn secp256k1_field_prime() -> BigUint256 {
    BigUint256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F").unwrap()
}

/// secp256k1 curve order: n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
pub fn secp256k1_order() -> BigUint256 {
    BigUint256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141").unwrap()
}

/// EC point in Jacobian coordinates (X, Y, Z) where the affine point is (X/Z^2, Y/Z^3)
/// Identity (point at infinity) is represented as Z = 0
#[derive(Clone, PartialEq, Eq)]
pub struct ECPoint {
    pub x: BigUint256,
    pub y: BigUint256,
    pub z: BigUint256,
}

impl ECPoint {
    pub fn identity() -> Self {
        ECPoint {
            x: BigUint256::ONE,
            y: BigUint256::ONE,
            z: BigUint256::ZERO,
        }
    }

    pub fn from_affine(x: BigUint256, y: BigUint256) -> Self {
        ECPoint {
            x,
            y,
            z: BigUint256::ONE,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.z.is_zero()
    }
}

impl fmt::Display for ECPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identity() {
            write!(f, "ECPoint(infinity)")
        } else {
            let (ax, ay) = point_to_affine_internal(self);
            write!(f, "ECPoint(0x{}, 0x{})", ax, ay)
        }
    }
}

impl fmt::Debug for ECPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

/// secp256k1 generator point G
pub fn secp256k1_generator() -> ECPoint {
    let gx = BigUint256::from_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798").unwrap();
    let gy = BigUint256::from_hex("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8").unwrap();
    ECPoint::from_affine(gx, gy)
}

/// Point doubling on secp256k1: 2P in Jacobian coordinates
/// secp256k1: y^2 = x^3 + 7, so a = 0
pub fn point_double(p: &ECPoint) -> ECPoint {
    let prime = secp256k1_field_prime();

    if p.is_identity() || p.y.is_zero() {
        return ECPoint::identity();
    }

    // a = 0 for secp256k1, simplified doubling formulas
    let ysq = mod_mul(&p.y, &p.y, &prime);
    let s = mod_mul(&BigUint256::from_u64(4), &mod_mul(&p.x, &ysq, &prime), &prime);
    let xsq = mod_mul(&p.x, &p.x, &prime);
    let m = mod_mul(&BigUint256::from_u64(3), &xsq, &prime);
    // (a=0, so no a*z^4 term)

    let msq = mod_mul(&m, &m, &prime);
    let x3 = mod_sub(&msq, &mod_mul(&BigUint256::from_u64(2), &s, &prime), &prime);

    let s_minus_x3 = mod_sub(&s, &x3, &prime);
    let ysq_sq = mod_mul(&ysq, &ysq, &prime);
    let y3 = mod_sub(
        &mod_mul(&m, &s_minus_x3, &prime),
        &mod_mul(&BigUint256::from_u64(8), &ysq_sq, &prime),
        &prime,
    );

    let z3 = mod_mul(&BigUint256::from_u64(2), &mod_mul(&p.y, &p.z, &prime), &prime);

    ECPoint { x: x3, y: y3, z: z3 }
}

/// Point addition: P + Q in Jacobian coordinates
pub fn point_add(p: &ECPoint, q: &ECPoint) -> ECPoint {
    let prime = secp256k1_field_prime();

    if p.is_identity() {
        return q.clone();
    }
    if q.is_identity() {
        return p.clone();
    }

    let z1sq = mod_mul(&p.z, &p.z, &prime);
    let z2sq = mod_mul(&q.z, &q.z, &prime);
    let u1 = mod_mul(&p.x, &z2sq, &prime);
    let u2 = mod_mul(&q.x, &z1sq, &prime);
    let s1 = mod_mul(&p.y, &mod_mul(&q.z, &z2sq, &prime), &prime);
    let s2 = mod_mul(&q.y, &mod_mul(&p.z, &z1sq, &prime), &prime);

    if u1 == u2 {
        if s1 == s2 {
            return point_double(p);
        } else {
            return ECPoint::identity();
        }
    }

    let h = mod_sub(&u2, &u1, &prime);
    let i = mod_mul(&BigUint256::from_u64(2), &h, &prime);
    let i = mod_mul(&i, &i, &prime);
    let j = mod_mul(&h, &i, &prime);
    let r = mod_mul(&BigUint256::from_u64(2), &mod_sub(&s2, &s1, &prime), &prime);
    let v = mod_mul(&u1, &i, &prime);

    let rsq = mod_mul(&r, &r, &prime);
    let x3 = mod_sub(&mod_sub(&rsq, &j, &prime), &mod_mul(&BigUint256::from_u64(2), &v, &prime), &prime);

    let v_minus_x3 = mod_sub(&v, &x3, &prime);
    let y3 = mod_sub(
        &mod_mul(&r, &v_minus_x3, &prime),
        &mod_mul(&BigUint256::from_u64(2), &mod_mul(&s1, &j, &prime), &prime),
        &prime,
    );

    let z1z2 = mod_mul(&p.z, &q.z, &prime);
    let z3 = mod_mul(&mod_mul(&BigUint256::from_u64(2), &z1z2, &prime), &h, &prime);

    ECPoint { x: x3, y: y3, z: z3 }
}

/// Scalar multiplication: k * P using double-and-add
pub fn scalar_mul(k: &BigUint256, p: &ECPoint) -> ECPoint {
    if k.is_zero() || p.is_identity() {
        return ECPoint::identity();
    }

    let mut result = ECPoint::identity();
    let bits = k.bits();

    for i in (0..bits).rev() {
        result = point_double(&result);
        if k.bit(i) {
            result = point_add(&result, p);
        }
    }
    result
}

/// Convert Jacobian to affine coordinates
fn point_to_affine_internal(p: &ECPoint) -> (BigUint256, BigUint256) {
    if p.is_identity() {
        return (BigUint256::ZERO, BigUint256::ZERO);
    }
    let prime = secp256k1_field_prime();
    let z_inv = mod_inv(&p.z, &prime);
    let z_inv2 = mod_mul(&z_inv, &z_inv, &prime);
    let z_inv3 = mod_mul(&z_inv2, &z_inv, &prime);
    let ax = mod_mul(&p.x, &z_inv2, &prime);
    let ay = mod_mul(&p.y, &z_inv3, &prime);
    (ax, ay)
}

pub fn point_to_affine(p: &ECPoint) -> (BigUint256, BigUint256) {
    point_to_affine_internal(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bigint_from_hex() {
        let n = BigUint256::from_hex("FF").unwrap();
        assert_eq!(n.limbs[0], 255);
        assert_eq!(n.limbs[1], 0);

        let n = BigUint256::from_hex("100000000").unwrap();
        assert_eq!(n.limbs[0], 0);
        assert_eq!(n.limbs[1], 1);
    }

    #[test]
    fn test_bigint_display() {
        let n = BigUint256::from_hex("DEADBEEF").unwrap();
        assert_eq!(format!("{}", n), "deadbeef");
    }

    #[test]
    fn test_add() {
        let a = BigUint256::from_u64(100);
        let b = BigUint256::from_u64(200);
        let (sum, carry) = a.add_with_carry(&b);
        assert_eq!(sum, BigUint256::from_u64(300));
        assert!(!carry);
    }

    #[test]
    fn test_sub() {
        let a = BigUint256::from_u64(300);
        let b = BigUint256::from_u64(200);
        let (diff, borrow) = a.sub_with_borrow(&b);
        assert_eq!(diff, BigUint256::from_u64(100));
        assert!(!borrow);
    }

    #[test]
    fn test_mul() {
        let a = BigUint256::from_u64(1000);
        let b = BigUint256::from_u64(2000);
        let (lo, hi) = a.mul_wide(&b);
        assert_eq!(lo, BigUint256::from_u64(2_000_000));
        assert!(hi.is_zero());
    }

    #[test]
    fn test_mod_mul() {
        let a = BigUint256::from_u64(7);
        let b = BigUint256::from_u64(6);
        let p = BigUint256::from_u64(13);
        let result = mod_mul(&a, &b, &p);
        assert_eq!(result, BigUint256::from_u64(3)); // 42 mod 13 = 3
    }

    #[test]
    fn test_mod_pow() {
        let base = BigUint256::from_u64(3);
        let exp = BigUint256::from_u64(4);
        let p = BigUint256::from_u64(13);
        let result = mod_pow(&base, &exp, &p);
        assert_eq!(result, BigUint256::from_u64(3)); // 81 mod 13 = 3
    }

    #[test]
    fn test_mod_inv() {
        let a = BigUint256::from_u64(3);
        let p = BigUint256::from_u64(13);
        let inv = mod_inv(&a, &p);
        let check = mod_mul(&a, &inv, &p);
        assert_eq!(check, BigUint256::ONE); // 3 * 3^(-1) mod 13 = 1
    }

    #[test]
    fn test_secp256k1_generator_on_curve() {
        let g = secp256k1_generator();
        let p = secp256k1_field_prime();
        // Check y^2 = x^3 + 7
        let y2 = mod_mul(&g.y, &g.y, &p);
        let x3 = mod_mul(&g.x, &mod_mul(&g.x, &g.x, &p), &p);
        let rhs = mod_add(&x3, &BigUint256::from_u64(7), &p);
        assert_eq!(y2, rhs, "Generator point must be on the curve");
    }

    #[test]
    fn test_scalar_mul_identity() {
        let g = secp256k1_generator();
        let result = scalar_mul(&BigUint256::ONE, &g);
        let (rx, ry) = point_to_affine(&result);
        let (gx, gy) = (g.x.clone(), g.y.clone());
        assert_eq!(rx, gx);
        assert_eq!(ry, gy);
    }

    #[test]
    fn test_scalar_mul_2g() {
        let g = secp256k1_generator();
        let two_g = scalar_mul(&BigUint256::from_u64(2), &g);
        let (x, _y) = point_to_affine(&two_g);
        // Known 2G x-coordinate for secp256k1
        let expected_x = BigUint256::from_hex(
            "C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5"
        ).unwrap();
        assert_eq!(x, expected_x, "2G x-coordinate must match known value");
    }

    #[test]
    fn test_point_add_equals_double() {
        let g = secp256k1_generator();
        let double_g = point_double(&g);
        let add_g = point_add(&g, &g);
        let (dx, dy) = point_to_affine(&double_g);
        let (ax, ay) = point_to_affine(&add_g);
        assert_eq!(dx, ax);
        assert_eq!(dy, ay);
    }

    #[test]
    fn test_known_privkey_to_pubkey() {
        // Private key = 1 -> Public key = Generator point
        let g = secp256k1_generator();
        let pubkey = scalar_mul(&BigUint256::ONE, &g);
        let (px, py) = point_to_affine(&pubkey);
        assert_eq!(px, g.x);
        assert_eq!(py, g.y);
    }
}

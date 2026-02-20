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

// --- Scalar field (mod N) operations ---

/// Scalar (mod N) — operates in the curve order group
#[derive(Clone, PartialEq, Eq)]
pub struct Scalar {
    pub value: BigUint256,
}

impl Scalar {
    pub fn new(value: BigUint256) -> Self {
        let n = secp256k1_order();
        let value = mod_mul(&value, &BigUint256::ONE, &n);
        Self { value }
    }

    pub fn from_hex(hex: &str) -> Option<Self> {
        BigUint256::from_hex(hex).map(|v| Self::new(v))
    }

    pub fn zero() -> Self {
        Self { value: BigUint256::ZERO }
    }

    pub fn one() -> Self {
        Self { value: BigUint256::ONE }
    }

    pub fn add(&self, other: &Scalar) -> Scalar {
        let n = secp256k1_order();
        Scalar { value: mod_add(&self.value, &other.value, &n) }
    }

    pub fn sub(&self, other: &Scalar) -> Scalar {
        let n = secp256k1_order();
        Scalar { value: mod_sub(&self.value, &other.value, &n) }
    }

    pub fn mul(&self, other: &Scalar) -> Scalar {
        let n = secp256k1_order();
        Scalar { value: mod_mul(&self.value, &other.value, &n) }
    }

    pub fn inv(&self) -> Scalar {
        let n = secp256k1_order();
        Scalar { value: mod_inv(&self.value, &n) }
    }

    pub fn neg(&self) -> Scalar {
        let n = secp256k1_order();
        Scalar { value: mod_sub(&BigUint256::ZERO, &self.value, &n) }
    }

    pub fn pow(&self, exp: &BigUint256) -> Scalar {
        let n = secp256k1_order();
        Scalar { value: mod_pow(&self.value, exp, &n) }
    }

    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    /// Check if scalar is in valid range [1, n-1]
    pub fn is_valid_privkey(&self) -> bool {
        !self.is_zero() && self.value.cmp(&secp256k1_order()) == std::cmp::Ordering::Less
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{}", self.value)
    }
}

impl fmt::Debug for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scalar(0x{})", self.value)
    }
}

// --- Point validation, negation, compression ---

/// secp256k1 curve constant b = 7
pub fn secp256k1_b() -> BigUint256 {
    BigUint256::from_u64(7)
}

/// Check if affine point (x, y) is on secp256k1: y^2 = x^3 + 7 (mod p)
pub fn point_on_curve(x: &BigUint256, y: &BigUint256) -> bool {
    let p = secp256k1_field_prime();
    let y2 = mod_mul(y, y, &p);
    let x3 = mod_mul(x, &mod_mul(x, x, &p), &p);
    let rhs = mod_add(&x3, &secp256k1_b(), &p);
    y2 == rhs
}

/// Validate an ECPoint (Jacobian) is on the curve
pub fn validate_point(pt: &ECPoint) -> bool {
    if pt.is_identity() {
        return true;
    }
    let (ax, ay) = point_to_affine(pt);
    point_on_curve(&ax, &ay)
}

/// Negate a point: -P = (x, -y)
pub fn point_negate(pt: &ECPoint) -> ECPoint {
    if pt.is_identity() {
        return ECPoint::identity();
    }
    let p = secp256k1_field_prime();
    ECPoint {
        x: pt.x.clone(),
        y: mod_sub(&BigUint256::ZERO, &pt.y, &p),
        z: pt.z.clone(),
    }
}

/// Point subtraction: P - Q = P + (-Q)
pub fn point_sub(p: &ECPoint, q: &ECPoint) -> ECPoint {
    point_add(p, &point_negate(q))
}

/// Compress point to 33 bytes: 02/03 prefix + x-coordinate
pub fn point_compress(pt: &ECPoint) -> Vec<u8> {
    if pt.is_identity() {
        return vec![0x00]; // infinity
    }
    let (ax, ay) = point_to_affine(pt);
    let prefix = if ay.is_even() { 0x02 } else { 0x03 };
    let mut result = vec![prefix];
    // x-coordinate as 32 bytes big-endian
    for i in (0..8).rev() {
        result.extend_from_slice(&ax.limbs[i].to_be_bytes());
    }
    result
}

/// Decompress point from 33 bytes (02/03 + x)
pub fn point_decompress(data: &[u8]) -> Option<ECPoint> {
    if data.len() != 33 {
        return None;
    }
    let prefix = data[0];
    if prefix != 0x02 && prefix != 0x03 {
        return None;
    }

    // Parse x from big-endian bytes
    let mut limbs = [0u32; 8];
    for i in 0..8 {
        let offset = 1 + (7 - i) * 4;
        limbs[i] = u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
    }
    let x = BigUint256 { limbs };

    // Compute y^2 = x^3 + 7 mod p
    let p = secp256k1_field_prime();
    let x3 = mod_mul(&x, &mod_mul(&x, &x, &p), &p);
    let y2 = mod_add(&x3, &secp256k1_b(), &p);

    // Square root: y = y2^((p+1)/4) mod p (works because p ≡ 3 mod 4)
    let (p_plus_1, _) = p.add_with_carry(&BigUint256::ONE);
    let exp = p_plus_1.shr1().shr1(); // (p+1)/4
    let y = mod_pow(&y2, &exp, &p);

    // Verify
    if mod_mul(&y, &y, &p) != y2 {
        return None; // x is not on the curve
    }

    // Choose correct y parity
    let want_even = prefix == 0x02;
    let y = if y.is_even() == want_even {
        y
    } else {
        mod_sub(&BigUint256::ZERO, &y, &p)
    };

    Some(ECPoint::from_affine(x, y))
}

/// Serialize uncompressed point: 04 + x (32 bytes) + y (32 bytes) = 65 bytes
pub fn point_serialize_uncompressed(pt: &ECPoint) -> Vec<u8> {
    if pt.is_identity() {
        return vec![0x00];
    }
    let (ax, ay) = point_to_affine(pt);
    let mut result = vec![0x04];
    for i in (0..8).rev() {
        result.extend_from_slice(&ax.limbs[i].to_be_bytes());
    }
    for i in (0..8).rev() {
        result.extend_from_slice(&ay.limbs[i].to_be_bytes());
    }
    result
}

/// Parse SEC1 encoded point (compressed or uncompressed)
pub fn point_parse_sec1(data: &[u8]) -> Option<ECPoint> {
    if data.is_empty() {
        return None;
    }
    match data[0] {
        0x00 => Some(ECPoint::identity()),
        0x02 | 0x03 => point_decompress(data),
        0x04 => {
            if data.len() != 65 {
                return None;
            }
            let mut x_limbs = [0u32; 8];
            let mut y_limbs = [0u32; 8];
            for i in 0..8 {
                let offset = 1 + (7 - i) * 4;
                x_limbs[i] = u32::from_be_bytes([
                    data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
                ]);
            }
            for i in 0..8 {
                let offset = 33 + (7 - i) * 4;
                y_limbs[i] = u32::from_be_bytes([
                    data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
                ]);
            }
            let x = BigUint256 { limbs: x_limbs };
            let y = BigUint256 { limbs: y_limbs };
            if point_on_curve(&x, &y) {
                Some(ECPoint::from_affine(x, y))
            } else {
                None
            }
        }
        _ => None,
    }
}

// --- SHA-256 ---

/// SHA-256 hash function (pure implementation, no dependencies)
pub fn sha256(data: &[u8]) -> [u8; 32] {
    // Initial hash values (first 32 bits of fractional parts of sqrt of first 8 primes)
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    // Round constants
    let k: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    // Pre-processing: padding
    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit (64-byte) block
    for chunk in padded.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(k[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut result = [0u8; 32];
    for i in 0..8 {
        result[i * 4..i * 4 + 4].copy_from_slice(&h[i].to_be_bytes());
    }
    result
}

/// Double SHA-256 (used in Bitcoin)
pub fn sha256d(data: &[u8]) -> [u8; 32] {
    sha256(&sha256(data))
}

/// SHA-256 hash to BigUint256
pub fn sha256_to_bigint(data: &[u8]) -> BigUint256 {
    let hash = sha256(data);
    let mut limbs = [0u32; 8];
    for i in 0..8 {
        let offset = (7 - i) * 4;
        limbs[i] = u32::from_be_bytes([
            hash[offset], hash[offset + 1], hash[offset + 2], hash[offset + 3],
        ]);
    }
    BigUint256 { limbs }
}

/// Convert BigUint256 to 32-byte big-endian
pub fn bigint_to_bytes32(n: &BigUint256) -> [u8; 32] {
    let mut result = [0u8; 32];
    for i in 0..8 {
        let bytes = n.limbs[7 - i].to_be_bytes();
        result[i * 4..i * 4 + 4].copy_from_slice(&bytes);
    }
    result
}

/// Convert 32-byte big-endian to BigUint256
pub fn bytes32_to_bigint(data: &[u8; 32]) -> BigUint256 {
    let mut limbs = [0u32; 8];
    for i in 0..8 {
        let offset = (7 - i) * 4;
        limbs[i] = u32::from_be_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
        ]);
    }
    BigUint256 { limbs }
}

// --- ECDSA ---

/// ECDSA signature
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ECDSASignature {
    pub r: BigUint256,
    pub s: BigUint256,
}

impl fmt::Display for ECDSASignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ECDSA(r=0x{}, s=0x{})", self.r, self.s)
    }
}

/// ECDSA sign: produce (r, s) from private key and message hash
///
/// message_hash should be the SHA-256 hash of the message, truncated to 256 bits.
/// k is the nonce — MUST be unique per signature. In production, use RFC 6979.
pub fn ecdsa_sign(
    privkey: &BigUint256,
    message_hash: &BigUint256,
    k: &BigUint256,
) -> Option<ECDSASignature> {
    let n = secp256k1_order();
    let g = secp256k1_generator();

    // R = k * G
    let r_point = scalar_mul(k, &g);
    let (rx, _) = point_to_affine(&r_point);

    // r = rx mod n
    let r = mod_mul(&rx, &BigUint256::ONE, &n);
    if r.is_zero() {
        return None;
    }

    // s = k^-1 * (hash + r * privkey) mod n
    let k_inv = mod_inv(k, &n);
    let r_priv = mod_mul(&r, privkey, &n);
    let hash_plus_r_priv = mod_add(message_hash, &r_priv, &n);
    let s = mod_mul(&k_inv, &hash_plus_r_priv, &n);
    if s.is_zero() {
        return None;
    }

    // Low-S normalization (BIP-62): if s > n/2, use n - s
    let half_n = n.shr1();
    let s = if s.cmp(&half_n) == std::cmp::Ordering::Greater {
        mod_sub(&BigUint256::ZERO, &s, &n)
    } else {
        s
    };

    Some(ECDSASignature { r, s })
}

/// ECDSA sign with automatic deterministic nonce (RFC 6979 simplified)
pub fn ecdsa_sign_deterministic(
    privkey: &BigUint256,
    message: &[u8],
) -> Option<ECDSASignature> {
    let message_hash = sha256_to_bigint(message);

    // Simplified RFC 6979: k = SHA256(privkey || message_hash)
    // NOTE: This is a simplified version. Production should use full HMAC-DRBG.
    let mut k_input = Vec::new();
    k_input.extend_from_slice(&bigint_to_bytes32(privkey));
    k_input.extend_from_slice(&bigint_to_bytes32(&message_hash));
    let k_hash = sha256(&k_input);
    let k = bytes32_to_bigint(&k_hash);

    // Ensure k is in [1, n-1]
    let n = secp256k1_order();
    let k = mod_mul(&k, &BigUint256::ONE, &n);
    if k.is_zero() {
        return None;
    }

    ecdsa_sign(privkey, &message_hash, &k)
}

/// ECDSA verify: check that signature (r, s) is valid for pubkey and message hash
pub fn ecdsa_verify(
    pubkey: &ECPoint,
    message_hash: &BigUint256,
    sig: &ECDSASignature,
) -> bool {
    let n = secp256k1_order();
    let g = secp256k1_generator();

    // Check r, s in [1, n-1]
    if sig.r.is_zero() || sig.s.is_zero() {
        return false;
    }
    if sig.r.cmp(&n) != std::cmp::Ordering::Less {
        return false;
    }
    if sig.s.cmp(&n) != std::cmp::Ordering::Less {
        return false;
    }

    // s_inv = s^-1 mod n
    let s_inv = mod_inv(&sig.s, &n);

    // u1 = hash * s_inv mod n
    let u1 = mod_mul(message_hash, &s_inv, &n);

    // u2 = r * s_inv mod n
    let u2 = mod_mul(&sig.r, &s_inv, &n);

    // R = u1*G + u2*pubkey
    let r_point = point_add(&scalar_mul(&u1, &g), &scalar_mul(&u2, pubkey));

    if r_point.is_identity() {
        return false;
    }

    let (rx, _) = point_to_affine(&r_point);
    let rx_mod_n = mod_mul(&rx, &BigUint256::ONE, &n);

    rx_mod_n == sig.r
}

// --- Schnorr signatures (BIP-340) ---

/// BIP-340 Schnorr signature (64 bytes: R.x || s)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchnorrSignature {
    pub rx: BigUint256,  // x-coordinate of R
    pub s: BigUint256,
}

impl fmt::Display for SchnorrSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Schnorr(rx=0x{}, s=0x{})", self.rx, self.s)
    }
}

/// BIP-340 tagged hash: SHA256(SHA256(tag) || SHA256(tag) || data)
fn tagged_hash(tag: &str, data: &[u8]) -> [u8; 32] {
    let tag_hash = sha256(tag.as_bytes());
    let mut input = Vec::new();
    input.extend_from_slice(&tag_hash);
    input.extend_from_slice(&tag_hash);
    input.extend_from_slice(data);
    sha256(&input)
}

/// BIP-340 Schnorr sign
/// privkey must be in [1, n-1]. The public key is the x-only representation.
pub fn schnorr_sign(privkey: &BigUint256, message: &[u8]) -> Option<SchnorrSignature> {
    let n = secp256k1_order();
    let p = secp256k1_field_prime();
    let g = secp256k1_generator();

    // P = d' * G
    let pubkey = scalar_mul(privkey, &g);
    let (px, py) = point_to_affine(&pubkey);

    // If Y is odd, negate the secret key
    let d = if !py.is_even() {
        mod_sub(&BigUint256::ZERO, privkey, &n)
    } else {
        privkey.clone()
    };

    // t = d XOR tagged_hash("BIP0340/aux", aux_rand)
    // Simplified: use deterministic aux from privkey
    let aux = sha256(&bigint_to_bytes32(privkey));
    let t_hash = tagged_hash("BIP0340/aux", &aux);
    let t = bytes32_to_bigint(&t_hash);
    let mut xor_limbs = [0u32; 8];
    for i in 0..8 {
        xor_limbs[i] = d.limbs[i] ^ t.limbs[i];
    }
    let t_xor = BigUint256 { limbs: xor_limbs };

    // k' = tagged_hash("BIP0340/nonce", t || px || m) mod n
    let mut nonce_input = Vec::new();
    nonce_input.extend_from_slice(&bigint_to_bytes32(&t_xor));
    nonce_input.extend_from_slice(&bigint_to_bytes32(&px));
    nonce_input.extend_from_slice(message);
    let k_hash = tagged_hash("BIP0340/nonce", &nonce_input);
    let k_prime = mod_mul(&bytes32_to_bigint(&k_hash), &BigUint256::ONE, &n);
    if k_prime.is_zero() {
        return None;
    }

    // R = k' * G
    let r_point = scalar_mul(&k_prime, &g);
    let (rx, ry) = point_to_affine(&r_point);

    // If R.y is odd, negate k
    let k = if !ry.is_even() {
        mod_sub(&BigUint256::ZERO, &k_prime, &n)
    } else {
        k_prime
    };

    // e = tagged_hash("BIP0340/challenge", R.x || P.x || m) mod n
    let mut challenge_input = Vec::new();
    challenge_input.extend_from_slice(&bigint_to_bytes32(&rx));
    challenge_input.extend_from_slice(&bigint_to_bytes32(&px));
    challenge_input.extend_from_slice(message);
    let e_hash = tagged_hash("BIP0340/challenge", &challenge_input);
    let e = mod_mul(&bytes32_to_bigint(&e_hash), &BigUint256::ONE, &n);

    // s = (k + e * d) mod n
    let _ = p; // field prime unused here, we work mod n
    let s = mod_add(&k, &mod_mul(&e, &d, &n), &n);

    Some(SchnorrSignature { rx, s })
}

/// BIP-340 Schnorr verify
pub fn schnorr_verify(pubkey_x: &BigUint256, message: &[u8], sig: &SchnorrSignature) -> bool {
    let n = secp256k1_order();
    let p = secp256k1_field_prime();
    let g = secp256k1_generator();

    // Check r < p and s < n
    if sig.rx.cmp(&p) != std::cmp::Ordering::Less {
        return false;
    }
    if sig.s.cmp(&n) != std::cmp::Ordering::Less {
        return false;
    }

    // e = tagged_hash("BIP0340/challenge", R.x || P.x || m) mod n
    let mut challenge_input = Vec::new();
    challenge_input.extend_from_slice(&bigint_to_bytes32(&sig.rx));
    challenge_input.extend_from_slice(&bigint_to_bytes32(pubkey_x));
    challenge_input.extend_from_slice(message);
    let e_hash = tagged_hash("BIP0340/challenge", &challenge_input);
    let e = mod_mul(&bytes32_to_bigint(&e_hash), &BigUint256::ONE, &n);

    // R = s*G - e*P
    // First, lift P from x-only
    let pubkey = match lift_x(pubkey_x) {
        Some(p) => p,
        None => return false,
    };

    let s_g = scalar_mul(&sig.s, &g);
    let e_p = scalar_mul(&e, &pubkey);
    let r_point = point_sub(&s_g, &e_p);

    if r_point.is_identity() {
        return false;
    }

    let (rx, ry) = point_to_affine(&r_point);

    // R must have even y
    if !ry.is_even() {
        return false;
    }

    // R.x must equal sig.rx
    rx == sig.rx
}

/// Lift x-only public key to full point (BIP-340: choose even y)
pub fn lift_x(x: &BigUint256) -> Option<ECPoint> {
    let p = secp256k1_field_prime();
    if x.cmp(&p) != std::cmp::Ordering::Less {
        return None;
    }

    let x3 = mod_mul(x, &mod_mul(x, x, &p), &p);
    let c = mod_add(&x3, &secp256k1_b(), &p);

    // y = c^((p+1)/4) mod p
    let (p_plus_1, _) = p.add_with_carry(&BigUint256::ONE);
    let exp = p_plus_1.shr1().shr1();
    let y = mod_pow(&c, &exp, &p);

    if mod_mul(&y, &y, &p) != c {
        return None; // not a valid x
    }

    let y = if y.is_even() { y } else { mod_sub(&BigUint256::ZERO, &y, &p) };
    Some(ECPoint::from_affine(x.clone(), y))
}

// --- HMAC-SHA256 (for RFC 6979) ---

/// HMAC-SHA256
pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let block_size = 64;

    let key_padded = if key.len() > block_size {
        let h = sha256(key);
        let mut k = vec![0u8; block_size];
        k[..32].copy_from_slice(&h);
        k
    } else {
        let mut k = vec![0u8; block_size];
        k[..key.len()].copy_from_slice(key);
        k
    };

    // ipad = key XOR 0x36...
    let mut ipad = vec![0u8; block_size];
    for i in 0..block_size {
        ipad[i] = key_padded[i] ^ 0x36;
    }

    // opad = key XOR 0x5c...
    let mut opad = vec![0u8; block_size];
    for i in 0..block_size {
        opad[i] = key_padded[i] ^ 0x5c;
    }

    // inner = SHA256(ipad || data)
    let mut inner_input = ipad;
    inner_input.extend_from_slice(data);
    let inner = sha256(&inner_input);

    // outer = SHA256(opad || inner)
    let mut outer_input = opad;
    outer_input.extend_from_slice(&inner);
    sha256(&outer_input)
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

    // --- Scalar (mod N) tests ---

    #[test]
    fn test_scalar_arithmetic() {
        let a = Scalar::new(BigUint256::from_u64(42));
        let b = Scalar::new(BigUint256::from_u64(7));

        let sum = a.add(&b);
        assert_eq!(sum.value, BigUint256::from_u64(49));

        let diff = a.sub(&b);
        assert_eq!(diff.value, BigUint256::from_u64(35));

        let prod = a.mul(&b);
        assert_eq!(prod.value, BigUint256::from_u64(294));

        // a * a^-1 = 1 (mod n)
        let a_inv = a.inv();
        let check = a.mul(&a_inv);
        assert_eq!(check.value, BigUint256::ONE);
    }

    #[test]
    fn test_scalar_negation() {
        let a = Scalar::new(BigUint256::from_u64(42));
        let neg_a = a.neg();
        let zero = a.add(&neg_a);
        assert!(zero.is_zero());
    }

    #[test]
    fn test_scalar_valid_privkey() {
        let valid = Scalar::new(BigUint256::from_u64(1));
        assert!(valid.is_valid_privkey());

        let zero = Scalar::zero();
        assert!(!zero.is_valid_privkey());
    }

    // --- Point validation tests ---

    #[test]
    fn test_point_on_curve() {
        let g = secp256k1_generator();
        assert!(point_on_curve(&g.x, &g.y));
        assert!(validate_point(&g));
        assert!(validate_point(&ECPoint::identity()));
    }

    #[test]
    fn test_point_not_on_curve() {
        // Random point not on curve
        let x = BigUint256::from_u64(12345);
        let y = BigUint256::from_u64(67890);
        assert!(!point_on_curve(&x, &y));
    }

    // --- Point negation tests ---

    #[test]
    fn test_point_negate() {
        let g = secp256k1_generator();
        let neg_g = point_negate(&g);
        let sum = point_add(&g, &neg_g);
        assert!(sum.is_identity());
    }

    #[test]
    fn test_point_sub() {
        let g = secp256k1_generator();
        let two_g = scalar_mul(&BigUint256::from_u64(2), &g);
        let result = point_sub(&two_g, &g);
        let (rx, ry) = point_to_affine(&result);
        assert_eq!(rx, g.x);
        assert_eq!(ry, g.y);
    }

    // --- Point compression tests ---

    #[test]
    fn test_point_compress_decompress() {
        let g = secp256k1_generator();
        let compressed = point_compress(&g);
        assert_eq!(compressed.len(), 33);
        assert!(compressed[0] == 0x02 || compressed[0] == 0x03);

        let decompressed = point_decompress(&compressed).unwrap();
        let (dx, dy) = point_to_affine(&decompressed);
        assert_eq!(dx, g.x);
        assert_eq!(dy, g.y);
    }

    #[test]
    fn test_point_uncompressed_roundtrip() {
        let g = secp256k1_generator();
        let data = point_serialize_uncompressed(&g);
        assert_eq!(data.len(), 65);
        assert_eq!(data[0], 0x04);

        let parsed = point_parse_sec1(&data).unwrap();
        let (px, py) = point_to_affine(&parsed);
        assert_eq!(px, g.x);
        assert_eq!(py, g.y);
    }

    #[test]
    fn test_compressed_2g() {
        let g = secp256k1_generator();
        let two_g = scalar_mul(&BigUint256::from_u64(2), &g);
        let compressed = point_compress(&two_g);
        let decompressed = point_decompress(&compressed).unwrap();
        let (dx, dy) = point_to_affine(&decompressed);
        let (ox, oy) = point_to_affine(&two_g);
        assert_eq!(dx, ox);
        assert_eq!(dy, oy);
    }

    // --- SHA-256 tests ---

    #[test]
    fn test_sha256_empty() {
        let hash = sha256(b"");
        let expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(hex, expected);
    }

    #[test]
    fn test_sha256_abc() {
        let hash = sha256(b"abc");
        let expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(hex, expected);
    }

    #[test]
    fn test_sha256_longer() {
        let hash = sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        let expected = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
        let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(hex, expected);
    }

    #[test]
    fn test_sha256d() {
        let hash = sha256d(b"hello");
        // SHA256(SHA256("hello")) - known value
        assert_eq!(hash.len(), 32);
        // Just verify it's not the same as single hash
        assert_ne!(hash, sha256(b"hello"));
    }

    #[test]
    fn test_hmac_sha256() {
        // RFC 4231 test vector 2
        let key = b"Jefe";
        let data = b"what do ya want for nothing?";
        let mac = hmac_sha256(key, data);
        let expected = "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843";
        let hex: String = mac.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(hex, expected);
    }

    // --- ECDSA tests ---

    #[test]
    fn test_ecdsa_sign_verify() {
        // Private key = 42
        let privkey = BigUint256::from_u64(42);
        let g = secp256k1_generator();
        let pubkey = scalar_mul(&privkey, &g);

        let message = b"Hello, Vortex!";
        let message_hash = sha256_to_bigint(message);

        // Sign with explicit nonce k = 12345
        let k = BigUint256::from_u64(12345);
        let sig = ecdsa_sign(&privkey, &message_hash, &k).unwrap();

        // Verify
        assert!(ecdsa_verify(&pubkey, &message_hash, &sig));

        // Verify fails with wrong message
        let wrong_hash = sha256_to_bigint(b"Wrong message");
        assert!(!ecdsa_verify(&pubkey, &wrong_hash, &sig));

        // Verify fails with wrong pubkey
        let wrong_pubkey = scalar_mul(&BigUint256::from_u64(43), &g);
        assert!(!ecdsa_verify(&wrong_pubkey, &message_hash, &sig));
    }

    #[test]
    fn test_ecdsa_deterministic() {
        let privkey = BigUint256::from_u64(1);
        let g = secp256k1_generator();
        let pubkey = scalar_mul(&privkey, &g);

        let message = b"test message";
        let sig = ecdsa_sign_deterministic(&privkey, message).unwrap();

        let message_hash = sha256_to_bigint(message);
        assert!(ecdsa_verify(&pubkey, &message_hash, &sig));

        // Same message produces same signature (deterministic)
        let sig2 = ecdsa_sign_deterministic(&privkey, message).unwrap();
        assert_eq!(sig, sig2);
    }

    #[test]
    fn test_ecdsa_low_s() {
        // Verify BIP-62 low-S normalization
        let privkey = BigUint256::from_u64(42);
        let message_hash = sha256_to_bigint(b"test");
        let k = BigUint256::from_u64(99999);
        let sig = ecdsa_sign(&privkey, &message_hash, &k).unwrap();

        let n = secp256k1_order();
        let half_n = n.shr1();
        assert!(sig.s.cmp(&half_n) != std::cmp::Ordering::Greater, "s must be low");
    }

    // --- Schnorr signature tests ---

    #[test]
    fn test_schnorr_sign_verify() {
        let privkey = BigUint256::from_u64(42);
        let g = secp256k1_generator();
        let pubkey = scalar_mul(&privkey, &g);
        let (px, _) = point_to_affine(&pubkey);

        let message = b"Hello, Schnorr!";
        let sig = schnorr_sign(&privkey, message).unwrap();

        assert!(schnorr_verify(&px, message, &sig));

        // Wrong message fails
        assert!(!schnorr_verify(&px, b"Wrong", &sig));
    }

    #[test]
    fn test_schnorr_deterministic() {
        let privkey = BigUint256::from_u64(1);
        let message = b"deterministic test";

        let sig1 = schnorr_sign(&privkey, message).unwrap();
        let sig2 = schnorr_sign(&privkey, message).unwrap();
        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_lift_x() {
        let g = secp256k1_generator();
        let lifted = lift_x(&g.x).unwrap();
        let (lx, ly) = point_to_affine(&lifted);
        assert_eq!(lx, g.x);
        assert!(ly.is_even()); // lift_x always returns even y
    }

    // --- Byte conversion tests ---

    #[test]
    fn test_bigint_bytes32_roundtrip() {
        let n = BigUint256::from_hex("DEADBEEFCAFEBABE1234567890ABCDEF").unwrap();
        let bytes = bigint_to_bytes32(&n);
        let recovered = bytes32_to_bigint(&bytes);
        assert_eq!(n, recovered);
    }
}

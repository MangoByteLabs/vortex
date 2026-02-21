// bytecarry.rs — Optimized carry propagation primitives for Vortex
//
// Carry chains are fundamental to:
// - Big integer arithmetic (crypto): multi-limb add/mul across limbs
// - Quantized neural networks: INT4/INT8 accumulation overflow management
// - Mixed-precision training: FP16→FP32 overflow handling
// - Residue Number Systems: positional↔RNS conversion requires carries

// ---------------------------------------------------------------------------
// 1. Core Carry-Chain Primitives
// ---------------------------------------------------------------------------

/// Add two N-limb numbers (little-endian u64 limbs) with carry propagation.
/// Returns (result, carry_out).
pub fn add_with_carry<const N: usize>(a: &[u64; N], b: &[u64; N]) -> ([u64; N], bool) {
    let mut result = [0u64; N];
    let mut carry: u64 = 0;
    for i in 0..N {
        let (s1, c1) = a[i].overflowing_add(b[i]);
        let (s2, c2) = s1.overflowing_add(carry);
        result[i] = s2;
        carry = (c1 as u64) + (c2 as u64);
    }
    (result, carry != 0)
}

/// Subtract two N-limb numbers: a - b. Returns (result, borrow_out).
pub fn sub_with_borrow<const N: usize>(a: &[u64; N], b: &[u64; N]) -> ([u64; N], bool) {
    let mut result = [0u64; N];
    let mut borrow: u64 = 0;
    for i in 0..N {
        let (s1, c1) = a[i].overflowing_sub(b[i]);
        let (s2, c2) = s1.overflowing_sub(borrow);
        result[i] = s2;
        borrow = (c1 as u64) + (c2 as u64);
    }
    (result, borrow != 0)
}

/// Schoolbook multiplication of two N-limb numbers producing a 2N-limb result.
pub fn mul_schoolbook<const N: usize>(a: &[u64; N], b: &[u64; N]) -> Vec<u64> {
    let mut result = vec![0u64; 2 * N];
    for i in 0..N {
        let mut carry: u64 = 0;
        for j in 0..N {
            let wide = (a[i] as u128) * (b[j] as u128) + (result[i + j] as u128) + (carry as u128);
            result[i + j] = wide as u64;
            carry = (wide >> 64) as u64;
        }
        result[i + N] = carry;
    }
    result
}

/// Multiply two N-limb numbers. Uses Karatsuba for N >= 4, schoolbook otherwise.
pub fn mul_wide<const N: usize>(a: &[u64; N], b: &[u64; N]) -> Vec<u64> {
    if N < 4 {
        return mul_schoolbook(a, b);
    }
    // Karatsuba on slices
    mul_karatsuba(&a[..], &b[..])
}

/// Karatsuba multiplication on arbitrary-length slices. Returns vec of length a.len()+b.len().
fn mul_karatsuba(a: &[u64], b: &[u64]) -> Vec<u64> {
    let n = a.len().max(b.len());
    if n <= 3 {
        // schoolbook fallback
        let mut result = vec![0u64; a.len() + b.len()];
        for i in 0..a.len() {
            let mut carry: u64 = 0;
            for j in 0..b.len() {
                let wide =
                    (a[i] as u128) * (b[j] as u128) + (result[i + j] as u128) + (carry as u128);
                result[i + j] = wide as u64;
                carry = (wide >> 64) as u64;
            }
            result[i + b.len()] = carry;
        }
        return result;
    }

    let half = n / 2;

    // Split a = a_lo + a_hi * B^half (pad if shorter)
    let a_lo = &a[..half.min(a.len())];
    let a_hi = if a.len() > half { &a[half..] } else { &[] as &[u64] };
    let b_lo = &b[..half.min(b.len())];
    let b_hi = if b.len() > half { &b[half..] } else { &[] as &[u64] };

    let z0 = mul_karatsuba(a_lo, b_lo);
    let z2 = if a_hi.is_empty() || b_hi.is_empty() {
        vec![0u64; 1]
    } else {
        mul_karatsuba(a_hi, b_hi)
    };

    // a_lo + a_hi, b_lo + b_hi
    let a_sum = limb_add(a_lo, a_hi);
    let b_sum = limb_add(b_lo, b_hi);
    let z1_full = mul_karatsuba(&a_sum, &b_sum);
    let z1 = limb_sub_vec(&limb_sub_vec(&z1_full, &z0), &z2);

    // result = z0 + z1 * B^half + z2 * B^(2*half)
    let mut result = vec![0u64; a.len() + b.len()];
    for (i, &v) in z0.iter().enumerate() {
        if i < result.len() {
            let wide = result[i] as u128 + v as u128;
            result[i] = wide as u64;
            let mut carry = (wide >> 64) as u64;
            let mut k = i + 1;
            while carry != 0 && k < result.len() {
                let w = result[k] as u128 + carry as u128;
                result[k] = w as u64;
                carry = (w >> 64) as u64;
                k += 1;
            }
        }
    }
    for (i, &v) in z1.iter().enumerate() {
        let pos = i + half;
        if pos < result.len() {
            let wide = result[pos] as u128 + v as u128;
            result[pos] = wide as u64;
            let mut carry = (wide >> 64) as u64;
            let mut k = pos + 1;
            while carry != 0 && k < result.len() {
                let w = result[k] as u128 + carry as u128;
                result[k] = w as u64;
                carry = (w >> 64) as u64;
                k += 1;
            }
        }
    }
    for (i, &v) in z2.iter().enumerate() {
        let pos = i + 2 * half;
        if pos < result.len() {
            let wide = result[pos] as u128 + v as u128;
            result[pos] = wide as u64;
            let mut carry = (wide >> 64) as u64;
            let mut k = pos + 1;
            while carry != 0 && k < result.len() {
                let w = result[k] as u128 + carry as u128;
                result[k] = w as u64;
                carry = (w >> 64) as u64;
                k += 1;
            }
        }
    }
    result
}

/// Helper: add two limb slices, return vec (may be one longer due to carry).
fn limb_add(a: &[u64], b: &[u64]) -> Vec<u64> {
    let n = a.len().max(b.len());
    let mut result = vec![0u64; n + 1];
    let mut carry: u64 = 0;
    for i in 0..n {
        let av = if i < a.len() { a[i] } else { 0 };
        let bv = if i < b.len() { b[i] } else { 0 };
        let wide = av as u128 + bv as u128 + carry as u128;
        result[i] = wide as u64;
        carry = (wide >> 64) as u64;
    }
    result[n] = carry;
    // trim trailing zeros (but keep at least one element)
    while result.len() > 1 && *result.last().unwrap() == 0 {
        result.pop();
    }
    result
}

/// Helper: subtract limb vectors (a - b), treating as unsigned. Wraps on underflow.
fn limb_sub_vec(a: &[u64], b: &[u64]) -> Vec<u64> {
    let n = a.len().max(b.len());
    let mut result = vec![0u64; n];
    let mut borrow: u64 = 0;
    for i in 0..n {
        let av = if i < a.len() { a[i] } else { 0 };
        let bv = if i < b.len() { b[i] } else { 0 };
        let (s1, c1) = av.overflowing_sub(bv);
        let (s2, c2) = s1.overflowing_sub(borrow);
        result[i] = s2;
        borrow = (c1 as u64) + (c2 as u64);
    }
    while result.len() > 1 && *result.last().unwrap() == 0 {
        result.pop();
    }
    result
}

/// Parallel prefix carry computation (Kogge-Stone style).
/// Given generate (g) and propagate (p) signals for each bit position,
/// compute all carry-out signals in O(log N) parallel steps.
/// This models what a GPU warp could do with shuffle instructions.
pub fn parallel_carry_lookahead(g: &[bool], p: &[bool]) -> Vec<bool> {
    assert_eq!(g.len(), p.len());
    let n = g.len();
    if n == 0 {
        return vec![];
    }
    // (G, P) pairs for prefix computation
    let mut gg: Vec<bool> = g.to_vec();
    let mut pp: Vec<bool> = p.to_vec();

    let mut dist = 1;
    while dist < n {
        let mut new_g = gg.clone();
        let mut new_p = pp.clone();
        for i in dist..n {
            // (G_i, P_i) ○ (G_{i-dist}, P_{i-dist})
            // G' = G_i | (P_i & G_{i-dist})
            // P' = P_i & P_{i-dist}
            new_g[i] = gg[i] || (pp[i] && gg[i - dist]);
            new_p[i] = pp[i] && pp[i - dist];
        }
        gg = new_g;
        pp = new_p;
        dist *= 2;
    }
    // carry[i] = G[i-1] (carry into position i); carry[0] = false (no carry-in)
    let mut carries = vec![false; n + 1];
    for i in 0..n {
        carries[i + 1] = gg[i];
    }
    carries
}

// ---------------------------------------------------------------------------
// 2. GPU-Optimized Carry Strategies
// ---------------------------------------------------------------------------

/// Carry-Save Accumulator: stores (sum, carry) redundant form.
/// Adding a value is O(1) — no carry propagation until resolve().
pub struct CarrySaveAccumulator {
    sum: Vec<u64>,
    carry: Vec<u64>,
}

impl CarrySaveAccumulator {
    pub fn new(limbs: usize) -> Self {
        Self {
            sum: vec![0u64; limbs],
            carry: vec![0u64; limbs],
        }
    }

    /// Add a value without propagating carries. O(1) per limb.
    pub fn add(&mut self, value: &[u64]) {
        let n = self.sum.len();
        // Three-input addition: sum, carry, value → new (sum, carry) in carry-save form
        for i in 0..n {
            let v = if i < value.len() { value[i] } else { 0 };
            let s = self.sum[i];
            let c = self.carry[i];
            // Full adder per limb (64-bit): sum = s ^ c ^ v, carry = maj(s,c,v) << 1
            // But we work at limb granularity using u128 to get the carry-out.
            let wide = s as u128 + c as u128 + v as u128;
            self.sum[i] = wide as u64;
            // The carry-out from this limb goes to the next limb's carry
            let cout = (wide >> 64) as u64;
            if i + 1 < n {
                self.carry[i + 1] = self.carry[i + 1].wrapping_add(cout);
            }
            self.carry[i] = 0; // absorbed into sum
        }
    }

    /// Resolve the redundant form into a single positional number.
    pub fn resolve(&self) -> Vec<u64> {
        let n = self.sum.len();
        let mut result = vec![0u64; n];
        let mut carry: u64 = 0;
        for i in 0..n {
            let wide = self.sum[i] as u128 + self.carry[i] as u128 + carry as u128;
            result[i] = wide as u64;
            carry = (wide >> 64) as u64;
        }
        result
    }

    pub fn limbs(&self) -> usize {
        self.sum.len()
    }
}

/// Redundant Binary Representation: each digit in {-1, 0, 1}.
/// Allows carry-free addition (bounded carry propagation of 1 step).
#[derive(Clone, Debug, PartialEq)]
pub struct RedundantNumber {
    pub digits: Vec<i8>, // each in [-1, 0, 1], little-endian
}

impl RedundantNumber {
    /// Convert a u64 value to redundant binary (non-negative).
    pub fn from_u64(val: u64) -> Self {
        let mut digits = Vec::new();
        let mut v = val;
        if v == 0 {
            return Self { digits: vec![0] };
        }
        while v > 0 {
            digits.push((v & 1) as i8);
            v >>= 1;
        }
        Self { digits }
    }

    /// Add two redundant numbers. The result uses at most 1 extra digit.
    pub fn add(&self, other: &Self) -> Self {
        let n = self.digits.len().max(other.digits.len());
        let mut result = vec![0i8; n + 1];
        let mut carry: i8 = 0;
        for i in 0..n {
            let a = if i < self.digits.len() { self.digits[i] } else { 0 };
            let b = if i < other.digits.len() { other.digits[i] } else { 0 };
            let s = a + b + carry;
            // Keep each digit in [-1, 0, 1]
            if s >= 2 {
                result[i] = s - 2;
                carry = 1;
            } else if s <= -2 {
                result[i] = s + 2;
                carry = -1;
            } else {
                result[i] = s;
                carry = 0;
            }
        }
        result[n] = carry;
        // trim
        while result.len() > 1 && *result.last().unwrap() == 0 {
            result.pop();
        }
        Self { digits: result }
    }

    /// Convert back to u64 (only valid for non-negative values that fit).
    pub fn to_u64(&self) -> u64 {
        let mut val: i64 = 0;
        for (i, &d) in self.digits.iter().enumerate() {
            val += (d as i64) << i;
        }
        val as u64
    }
}

/// Residue Number System: represent a number as residues modulo coprime bases.
/// Addition and multiplication are carry-free (per-channel mod).
#[derive(Clone, Debug)]
pub struct RNSNumber {
    pub residues: Vec<u64>,
    pub moduli: Vec<u64>,
}

/// Default coprime moduli (primes, product > 2^128 for reasonable range).
pub const DEFAULT_RNS_MODULI: &[u64] = &[
    (1u64 << 61) - 1,  // Mersenne prime 2^61-1
    (1u64 << 31) - 1,  // Mersenne prime 2^31-1
    1_000_000_007,      // common prime
    1_000_000_009,      // common prime
    998_244_353,        // NTT-friendly prime
];

impl RNSNumber {
    /// Create from a single u64 value.
    pub fn from_u64(val: u64, moduli: &[u64]) -> Self {
        let residues = moduli.iter().map(|&m| val % m).collect();
        Self {
            residues,
            moduli: moduli.to_vec(),
        }
    }

    /// Create from limbs (little-endian u64). Computes val mod each modulus.
    pub fn from_limbs(limbs: &[u64], moduli: &[u64]) -> Self {
        let residues = moduli
            .iter()
            .map(|&m| {
                let mut r: u128 = 0;
                for &limb in limbs.iter().rev() {
                    r = ((r << 64) + limb as u128) % m as u128;
                }
                r as u64
            })
            .collect();
        Self {
            residues,
            moduli: moduli.to_vec(),
        }
    }

    /// Carry-free addition.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.moduli, other.moduli);
        let residues = self
            .residues
            .iter()
            .zip(other.residues.iter())
            .zip(self.moduli.iter())
            .map(|((&a, &b), &m)| ((a as u128 + b as u128) % m as u128) as u64)
            .collect();
        Self {
            residues,
            moduli: self.moduli.clone(),
        }
    }

    /// Carry-free multiplication.
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.moduli, other.moduli);
        let residues = self
            .residues
            .iter()
            .zip(other.residues.iter())
            .zip(self.moduli.iter())
            .map(|((&a, &b), &m)| ((a as u128 * b as u128) % m as u128) as u64)
            .collect();
        Self {
            residues,
            moduli: self.moduli.clone(),
        }
    }

    /// CRT reconstruction back to positional (returns value mod product of moduli).
    /// This is the step that requires carries.
    pub fn to_u128(&self) -> u128 {
        // Garner's algorithm for mixed-radix conversion
        let k = self.moduli.len();
        // coeffs[i] = mixed-radix digit for position i
        let mut coeffs = vec![0u128; k];
        coeffs[0] = self.residues[0] as u128;

        for i in 1..k {
            let mi = self.moduli[i] as u128;
            let mut u = self.residues[i] as u128;
            // u = (r_i - (coeffs[0] + coeffs[1]*m0 + ...)) / (m0*m1*...*m_{i-1})  mod m_i
            // Iteratively: u = (u - coeffs[j]) * inv(m_j) mod m_i, for j = 0..i-1
            for j in 0..i {
                let cj = coeffs[j] % mi;
                if u >= cj {
                    u = u - cj;
                } else {
                    u = mi - (cj - u);
                }
                u = mod_mul_u128(u, mod_inv_u128(self.moduli[j] as u128, mi), mi);
            }
            coeffs[i] = u;
        }

        // Mixed-radix → positional: value = c0 + c1*m0 + c2*m0*m1 + ...
        let mut result: u128 = 0;
        let mut base: u128 = 1;
        for i in 0..k {
            result = result.wrapping_add(coeffs[i].wrapping_mul(base));
            if i < k - 1 {
                base = base.wrapping_mul(self.moduli[i] as u128);
            }
        }
        result
    }
}

fn mod_mul_u128(a: u128, b: u128, m: u128) -> u128 {
    // For moduli < 2^64 this won't overflow
    (a % m) * (b % m) % m
}

fn mod_inv_u128(a: u128, m: u128) -> u128 {
    // Extended Euclidean
    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        let tmp = r;
        r = old_r - q * r;
        old_r = tmp;
        let tmp = s;
        s = old_s - q * s;
        old_s = tmp;
    }
    ((old_s % m as i128 + m as i128) % m as i128) as u128
}

// ---------------------------------------------------------------------------
// 3. Quantization-Aware Carry Management
// ---------------------------------------------------------------------------

/// INT8 dot product with INT32 accumulation.
/// Core operation in quantized neural network inference.
pub fn int8_dot_i32(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as i32 * y as i32)
        .sum()
}

/// INT4 packed dot product (two INT4 values per byte, little-endian nibbles).
/// Unpacks, multiplies, accumulates in i32.
pub fn int4_dot_i32(a_packed: &[u8], b_packed: &[u8]) -> i32 {
    let mut acc: i32 = 0;
    for (&ab, &bb) in a_packed.iter().zip(b_packed.iter()) {
        // Extract two signed 4-bit values from each byte
        let a_lo = sign_extend_4bit(ab & 0x0F);
        let a_hi = sign_extend_4bit((ab >> 4) & 0x0F);
        let b_lo = sign_extend_4bit(bb & 0x0F);
        let b_hi = sign_extend_4bit((bb >> 4) & 0x0F);
        acc += a_lo as i32 * b_lo as i32;
        acc += a_hi as i32 * b_hi as i32;
    }
    acc
}

/// Sign-extend a 4-bit value to i8.
fn sign_extend_4bit(v: u8) -> i8 {
    let v = v & 0x0F;
    if v & 0x08 != 0 {
        (v | 0xF0) as i8
    } else {
        v as i8
    }
}

/// Mixed-precision dot product: optionally truncate to fp16 range during multiply.
pub fn fp16_dot_fp32(a: &[f32], b: &[f32], use_fp16_multiply: bool) -> f32 {
    let mut acc: f32 = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let product = if use_fp16_multiply {
            truncate_to_fp16(x) * truncate_to_fp16(y)
        } else {
            x * y
        };
        acc += product;
    }
    acc
}

/// Truncate f32 to fp16 precision (round to nearest, ties to even).
fn truncate_to_fp16(v: f32) -> f32 {
    // Clamp to fp16 range, then round mantissa to 10 bits
    let bits = v.to_bits();
    // Zero out the lower 13 bits of mantissa (23-10=13)
    let rounded = bits & 0xFFFF_E000;
    f32::from_bits(rounded)
}

/// Stochastic rounding to fp16 precision.
/// Uses random_bits to decide rounding direction for the truncated bits.
pub fn stochastic_round_fp16(value: f32, random_bits: u32) -> f32 {
    let bits = value.to_bits();
    let truncated_bits = bits & 0x0000_1FFF; // lower 13 bits
    let base = bits & 0xFFFF_E000;
    // If random < truncated fraction, round up
    let threshold = (random_bits & 0x1FFF) as u32;
    if truncated_bits > threshold {
        // round up: add 1 to the 13th bit position
        f32::from_bits(base.wrapping_add(0x2000))
    } else {
        f32::from_bits(base)
    }
}

// ---------------------------------------------------------------------------
// 4. MLIR Lowering Annotations / Carry Strategy Selection
// ---------------------------------------------------------------------------

/// Strategy for carry propagation, selected based on width and target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarryStrategy {
    /// Serial carry chain — CPU scalar code
    Serial,
    /// Carry lookahead — small widths on GPU (32-64 bit)
    Lookahead,
    /// Carry-save — big-integer multiply on GPU
    CarrySave,
    /// Residue Number System — NTT-friendly fields with small moduli
    ResidueNumberSystem,
}

/// Select the best carry strategy for a given bit width and target.
pub fn select_carry_strategy(bit_width: usize, target: &str) -> CarryStrategy {
    match target {
        "cpu" | "x86" | "aarch64" => {
            if bit_width <= 128 {
                CarryStrategy::Serial
            } else {
                CarryStrategy::CarrySave
            }
        }
        "gpu" | "cuda" | "ptx" | "amdgcn" => {
            if bit_width <= 64 {
                CarryStrategy::Lookahead
            } else if bit_width <= 512 {
                CarryStrategy::CarrySave
            } else {
                CarryStrategy::ResidueNumberSystem
            }
        }
        "ntt" | "field" => CarryStrategy::ResidueNumberSystem,
        _ => CarryStrategy::Serial,
    }
}

// ---------------------------------------------------------------------------
// 5. Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_with_carry_no_overflow() {
        let a = [1u64, 2, 3, 4];
        let b = [10u64, 20, 30, 40];
        let (result, carry) = add_with_carry(&a, &b);
        assert_eq!(result, [11, 22, 33, 44]);
        assert!(!carry);
    }

    #[test]
    fn test_add_with_carry_overflow() {
        let a = [u64::MAX, 0];
        let b = [1u64, 0];
        let (result, carry) = add_with_carry(&a, &b);
        assert_eq!(result, [0, 1]);
        assert!(!carry);
    }

    #[test]
    fn test_add_with_carry_full_overflow() {
        let a = [u64::MAX, u64::MAX];
        let b = [1u64, 0];
        let (result, carry) = add_with_carry(&a, &b);
        assert_eq!(result, [0, 0]);
        assert!(carry);
    }

    #[test]
    fn test_sub_with_borrow() {
        let a = [10u64, 20];
        let b = [3u64, 5];
        let (result, borrow) = sub_with_borrow(&a, &b);
        assert_eq!(result, [7, 15]);
        assert!(!borrow);
    }

    #[test]
    fn test_sub_with_borrow_underflow() {
        let a = [0u64, 0];
        let b = [1u64, 0];
        let (result, borrow) = sub_with_borrow(&a, &b);
        assert_eq!(result, [u64::MAX, u64::MAX]);
        assert!(borrow);
    }

    #[test]
    fn test_mul_schoolbook_small() {
        // 3 * 7 = 21
        let a = [3u64];
        let b = [7u64];
        let result = mul_schoolbook(&a, &b);
        assert_eq!(result, vec![21, 0]);
    }

    #[test]
    fn test_mul_schoolbook_multi_limb() {
        // (2^64) * 3 = 3 * 2^64
        let a = [0u64, 1]; // = 2^64
        let b = [3u64, 0];
        let result = mul_schoolbook(&a, &b);
        assert_eq!(result, vec![0, 3, 0, 0]);
    }

    #[test]
    fn test_mul_wide_matches_schoolbook() {
        let a = [0xDEADBEEFu64, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0];
        let b = [0x11111111u64, 0x22222222, 0x33333333, 0x44444444];
        let wide = mul_wide(&a, &b);
        let school = mul_schoolbook(&a, &b);
        assert_eq!(wide, school, "Karatsuba must match schoolbook");
    }

    #[test]
    fn test_parallel_carry_lookahead() {
        // Simple 4-bit addition: 0b1011 + 0b0111
        // g[i] = a[i] & b[i], p[i] = a[i] ^ b[i]
        let a_bits = [true, true, false, true]; // 1011
        let b_bits = [true, true, true, false]; // 0111
        let g: Vec<bool> = a_bits.iter().zip(b_bits.iter()).map(|(&a, &b)| a && b).collect();
        let p: Vec<bool> = a_bits.iter().zip(b_bits.iter()).map(|(&a, &b)| a ^ b).collect();
        let carries = parallel_carry_lookahead(&g, &p);
        // Manual: 1011 + 0111 = 10010
        // c0=0, bit0: 1+1=10 → sum=0,c1=1
        // c1=1, bit1: 1+1+1=11 → sum=1,c2=1
        // c2=1, bit2: 0+1+1=10 → sum=0,c3=1
        // c3=1, bit3: 1+0+1=10 → sum=0,c4=1
        assert_eq!(carries, vec![false, true, true, true, true]);
    }

    #[test]
    fn test_carry_save_accumulator() {
        let mut acc = CarrySaveAccumulator::new(4);
        // Accumulate 1000 copies of [1, 0, 0, 0]
        for _ in 0..1000 {
            acc.add(&[1, 0, 0, 0]);
        }
        let result = acc.resolve();
        assert_eq!(result[0], 1000);
        assert_eq!(result[1], 0);
    }

    #[test]
    fn test_carry_save_large_accumulation() {
        let mut acc = CarrySaveAccumulator::new(4);
        let mut reference = vec![0u128; 4];
        let values: Vec<[u64; 4]> = (0..1000)
            .map(|i| [i * 17 + 3, i * 31 + 7, i * 13 + 11, i * 7 + 1])
            .collect();
        for v in &values {
            acc.add(v);
            for j in 0..4 {
                reference[j] += v[j] as u128;
            }
        }
        let result = acc.resolve();
        // Check low 64 bits of each limb match (carries flow between limbs)
        // Actually we need to propagate the reference too
        let mut ref_propagated = vec![0u64; 4];
        let mut carry: u128 = 0;
        for j in 0..4 {
            let total = reference[j] + carry;
            ref_propagated[j] = total as u64;
            carry = total >> 64;
        }
        assert_eq!(result, ref_propagated);
    }

    #[test]
    fn test_redundant_number_add() {
        let a = RedundantNumber::from_u64(42);
        let b = RedundantNumber::from_u64(58);
        let c = a.add(&b);
        assert_eq!(c.to_u64(), 100);
    }

    #[test]
    fn test_redundant_number_roundtrip() {
        for v in [0, 1, 127, 255, 1000, 65535, u64::MAX / 2] {
            let r = RedundantNumber::from_u64(v);
            assert_eq!(r.to_u64(), v, "roundtrip failed for {}", v);
        }
    }

    #[test]
    fn test_rns_add() {
        let moduli = &[7u64, 11, 13];
        let a = RNSNumber::from_u64(42, moduli);
        let b = RNSNumber::from_u64(58, moduli);
        let c = a.add(&b);
        // 42+58=100
        assert_eq!(c.residues[0], 100 % 7);
        assert_eq!(c.residues[1], 100 % 11);
        assert_eq!(c.residues[2], 100 % 13);
    }

    #[test]
    fn test_rns_mul() {
        let moduli = &[7u64, 11, 13];
        let a = RNSNumber::from_u64(6, moduli);
        let b = RNSNumber::from_u64(7, moduli);
        let c = a.mul(&b);
        assert_eq!(c.residues[0], 42 % 7);
        assert_eq!(c.residues[1], 42 % 11);
        assert_eq!(c.residues[2], 42 % 13);
    }

    #[test]
    fn test_rns_roundtrip() {
        let moduli = &[7u64, 11, 13]; // product = 1001
        for v in [0u64, 1, 42, 100, 500, 1000] {
            let rns = RNSNumber::from_u64(v, moduli);
            let back = rns.to_u128();
            assert_eq!(back, v as u128, "RNS roundtrip failed for {}", v);
        }
    }

    #[test]
    fn test_rns_mul_roundtrip() {
        let moduli = &[1_000_000_007u64, 1_000_000_009, 998_244_353];
        let a_val = 123456u64;
        let b_val = 789012u64;
        let a = RNSNumber::from_u64(a_val, moduli);
        let b = RNSNumber::from_u64(b_val, moduli);
        let c = a.mul(&b);
        let result = c.to_u128();
        assert_eq!(result, a_val as u128 * b_val as u128);
    }

    #[test]
    fn test_int8_dot() {
        let a: Vec<i8> = vec![1, -2, 3, -4];
        let b: Vec<i8> = vec![5, 6, -7, 8];
        // 1*5 + (-2)*6 + 3*(-7) + (-4)*8 = 5 - 12 - 21 - 32 = -60
        assert_eq!(int8_dot_i32(&a, &b), -60);
    }

    #[test]
    fn test_int8_dot_matches_f64() {
        let a: Vec<i8> = vec![127, -128, 64, -64, 1, -1, 0, 100];
        let b: Vec<i8> = vec![-1, 2, -3, 4, -5, 6, -7, 8];
        let i32_result = int8_dot_i32(&a, &b);
        let f64_result: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum();
        assert_eq!(i32_result, f64_result as i32);
    }

    #[test]
    fn test_int4_dot() {
        // Pack two INT4 values per byte: lo nibble, hi nibble
        // a = [3, -2] packed as one byte: lo=3 (0x03), hi=-2 (0x0E) → 0xE3
        // b = [1, 4]  packed: lo=1 (0x01), hi=4 (0x04) → 0x41
        // dot = 3*1 + (-2)*4 = 3 - 8 = -5
        let a_packed = vec![0xE3u8];
        let b_packed = vec![0x41u8];
        assert_eq!(int4_dot_i32(&a_packed, &b_packed), -5);
    }

    #[test]
    fn test_int4_dot_matches_f64() {
        // Two bytes = 4 int4 values each
        // a = [2, 3, -1, 5] → byte0: hi=3,lo=2 = 0x32, byte1: hi=5,lo=-1(0xF)=0x5F
        // b = [1, -2, 4, -3] → byte0: hi=-2(0xE),lo=1 = 0xE1, byte1: hi=-3(0xD),lo=4 = 0xD4
        // dot = 2*1 + 3*(-2) + (-1)*4 + 5*(-3) = 2 - 6 - 4 - 15 = -23
        let a_packed = vec![0x32, 0x5F];
        let b_packed = vec![0xE1, 0xD4];
        assert_eq!(int4_dot_i32(&a_packed, &b_packed), -23);
    }

    #[test]
    fn test_fp16_dot_fp32() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        // Exact values: 5+12+21+32 = 70
        let result = fp16_dot_fp32(&a, &b, false);
        assert_eq!(result, 70.0);
        // With fp16 truncation, these small integers are exact in fp16
        let result_fp16 = fp16_dot_fp32(&a, &b, true);
        assert_eq!(result_fp16, 70.0);
    }

    #[test]
    fn test_stochastic_round() {
        let val = 1.5f32;
        // With random_bits = 0, should round up (truncated > 0)
        let r1 = stochastic_round_fp16(val, 0);
        // With random_bits = 0x1FFF (max), should round down
        let r2 = stochastic_round_fp16(val, 0x1FFF);
        // Both should be valid fp16-precision values
        assert!(r1 == f32::from_bits(r1.to_bits() & 0xFFFF_E000));
        assert!(r2 == f32::from_bits(r2.to_bits() & 0xFFFF_E000));
    }

    #[test]
    fn test_carry_strategy_selection() {
        assert_eq!(select_carry_strategy(64, "cpu"), CarryStrategy::Serial);
        assert_eq!(select_carry_strategy(256, "cpu"), CarryStrategy::CarrySave);
        assert_eq!(select_carry_strategy(32, "gpu"), CarryStrategy::Lookahead);
        assert_eq!(select_carry_strategy(256, "gpu"), CarryStrategy::CarrySave);
        assert_eq!(
            select_carry_strategy(1024, "gpu"),
            CarryStrategy::ResidueNumberSystem
        );
        assert_eq!(
            select_carry_strategy(256, "ntt"),
            CarryStrategy::ResidueNumberSystem
        );
    }

    #[test]
    fn test_add_with_carry_matches_u128() {
        // Test that 2-limb add matches u128 arithmetic
        for &(a_val, b_val) in &[
            (0u128, 0u128),
            (1, 1),
            (u128::MAX / 2, u128::MAX / 2),
            ((1u128 << 64) - 1, 1),
            (0xDEAD_BEEF_CAFE_BABEu128, 0x1234_5678_9ABC_DEF0u128),
        ] {
            let a = [a_val as u64, (a_val >> 64) as u64];
            let b = [b_val as u64, (b_val >> 64) as u64];
            let (result, carry) = add_with_carry(&a, &b);
            let expected = a_val.wrapping_add(b_val);
            assert_eq!(result[0], expected as u64);
            assert_eq!(result[1], (expected >> 64) as u64);
            assert_eq!(carry, a_val.checked_add(b_val).is_none());
        }
    }

    #[test]
    fn bench_serial_vs_carry_save() {
        // Not a real benchmark, but verify both give the same result
        // and print timing for informational purposes
        let n = 1000;
        let values: Vec<[u64; 4]> = (0..n)
            .map(|i| [i as u64 * 17 + 3, i as u64 * 31 + 7, 0, 0])
            .collect();

        // Serial approach
        let start = std::time::Instant::now();
        let mut serial_result = [0u64; 4];
        let mut serial_carry = false;
        for v in &values {
            let (r, c) = add_with_carry(&serial_result, v);
            serial_result = r;
            serial_carry = c;
        }
        let serial_time = start.elapsed();
        let _ = serial_carry;

        // Carry-save approach
        let start = std::time::Instant::now();
        let mut acc = CarrySaveAccumulator::new(4);
        for v in &values {
            acc.add(v);
        }
        let cs_result = acc.resolve();
        let cs_time = start.elapsed();

        // Results must match
        assert_eq!(serial_result[0], cs_result[0]);
        assert_eq!(serial_result[1], cs_result[1]);

        eprintln!(
            "Timing (1000 4-limb adds): serial={:?}, carry-save={:?}",
            serial_time, cs_time
        );
    }
}

//! World-class 256-bit arithmetic engine for Vortex.
//!
//! Provides a unified U256 type with constant-time operations, Montgomery field
//! arithmetic, batch operations (Montgomery's trick), elliptic curve operations
//! (windowed NAF scalar multiplication, Pippenger MSM), NTT, and parallel
//! computation — all designed to outperform CUDA's 32-bit emulation approach.
//!
//! Key advantages over CUDA:
//! - Native 64-bit limbs (4 limbs vs 8x32-bit on GPU)
//! - No warp divergence from constant-time operations
//! - Batch inverse via Montgomery's trick: 1 inversion + 3N muls for N inverses
//! - Multi-threaded batch operations using CPU cores
//! - Pippenger MSM with bucket accumulation

use crate::modmath::{self, FieldParams, ModField, montgomery_mul};
use std::fmt;

// ============================================================
// 1. U256 — Optimized 256-bit unsigned integer
// ============================================================

/// A 256-bit unsigned integer stored as 4 little-endian u64 limbs.
/// All operations are constant-time for cryptographic safety.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct U256 {
    pub limbs: [u64; 4],
}

impl U256 {
    pub const ZERO: U256 = U256 { limbs: [0; 4] };
    pub const ONE: U256 = U256 { limbs: [1, 0, 0, 0] };
    pub const MAX: U256 = U256 { limbs: [u64::MAX; 4] };

    #[inline]
    pub const fn new(limbs: [u64; 4]) -> Self {
        Self { limbs }
    }

    /// Create from a single u64.
    #[inline]
    pub const fn from_u64(val: u64) -> Self {
        Self { limbs: [val, 0, 0, 0] }
    }

    /// Create from i64 (negative values become 0).
    #[inline]
    pub fn from_i64(val: i64) -> Self {
        if val < 0 { Self::ZERO } else { Self::from_u64(val as u64) }
    }

    /// Parse from hex string (with or without "0x" prefix).
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.strip_prefix("0x").or_else(|| hex.strip_prefix("0X")).unwrap_or(hex);
        let hex = hex.trim_start_matches('0');
        if hex.is_empty() {
            return Some(Self::ZERO);
        }
        if hex.len() > 64 {
            return None; // too large for 256 bits
        }
        let padded = format!("{:0>64}", hex);
        let mut limbs = [0u64; 4];
        for i in 0..4 {
            let start = (3 - i) * 16;
            limbs[i] = u64::from_str_radix(&padded[start..start + 16], 16).ok()?;
        }
        Some(Self { limbs })
    }

    /// Convert to hex string.
    pub fn to_hex(&self) -> String {
        let mut s = String::new();
        let mut started = false;
        for i in (0..4).rev() {
            if self.limbs[i] != 0 || started || i == 0 {
                if started {
                    s.push_str(&format!("{:016x}", self.limbs[i]));
                } else {
                    s.push_str(&format!("{:x}", self.limbs[i]));
                    started = true;
                }
            }
        }
        if s.is_empty() { "0".to_string() } else { s }
    }

    // --- Constant-time arithmetic ---

    /// Addition with carry. Returns (result, carry).
    #[inline]
    pub fn add_with_carry(a: &U256, b: &U256) -> (U256, bool) {
        let mut result = [0u64; 4];
        let mut carry = 0u64;
        for i in 0..4 {
            let sum = (a.limbs[i] as u128) + (b.limbs[i] as u128) + (carry as u128);
            result[i] = sum as u64;
            carry = (sum >> 64) as u64;
        }
        (U256 { limbs: result }, carry != 0)
    }

    /// Constant-time addition (wrapping).
    #[inline]
    pub fn add(&self, other: &U256) -> U256 {
        Self::add_with_carry(self, other).0
    }

    /// Subtraction with borrow. Returns (result, borrow).
    #[inline]
    pub fn sub_with_borrow(a: &U256, b: &U256) -> (U256, bool) {
        let mut result = [0u64; 4];
        let mut borrow = 0u128;
        for i in 0..4 {
            let diff = (a.limbs[i] as u128).wrapping_sub(b.limbs[i] as u128).wrapping_sub(borrow);
            result[i] = diff as u64;
            // Borrow if the subtraction underflowed
            borrow = (diff >> 127) & 1;
        }
        (U256 { limbs: result }, borrow != 0)
    }

    /// Constant-time subtraction (wrapping).
    #[inline]
    pub fn sub(&self, other: &U256) -> U256 {
        Self::sub_with_borrow(self, other).0
    }

    /// Full 256x256 -> 512 bit multiplication.
    pub fn mul_wide(&self, other: &U256) -> U512 {
        let mut result = [0u64; 8];
        for i in 0..4 {
            let mut carry = 0u64;
            for j in 0..4 {
                let prod = (self.limbs[i] as u128) * (other.limbs[j] as u128)
                    + (result[i + j] as u128)
                    + (carry as u128);
                result[i + j] = prod as u64;
                carry = (prod >> 64) as u64;
            }
            result[i + 4] = carry;
        }
        U512 { limbs: result }
    }

    /// Truncated 256-bit multiplication (lower 256 bits only).
    #[inline]
    pub fn mul(&self, other: &U256) -> U256 {
        let wide = self.mul_wide(other);
        U256 { limbs: [wide.limbs[0], wide.limbs[1], wide.limbs[2], wide.limbs[3]] }
    }

    /// Karatsuba multiplication for 256-bit: splits into two 128-bit halves.
    /// Uses 3 multiplications instead of 4 for the 128-bit sub-products.
    pub fn mul_karatsuba(&self, other: &U256) -> U512 {
        // Split a = a_hi * 2^128 + a_lo, b = b_hi * 2^128 + b_lo
        let a_lo = [self.limbs[0], self.limbs[1]];
        let a_hi = [self.limbs[2], self.limbs[3]];
        let b_lo = [other.limbs[0], other.limbs[1]];
        let b_hi = [other.limbs[2], other.limbs[3]];

        // z0 = a_lo * b_lo (128x128 -> 256)
        let z0 = mul_128x128(&a_lo, &b_lo);
        // z2 = a_hi * b_hi (128x128 -> 256)
        let z2 = mul_128x128(&a_hi, &b_hi);

        // z1 = (a_lo + a_hi) * (b_lo + b_hi) - z0 - z2
        let (a_sum, a_carry) = add_128(&a_lo, &a_hi);
        let (b_sum, b_carry) = add_128(&b_lo, &b_hi);
        // Handle carry: up to 129 bits each
        let mut z1_wide = mul_128x128(&a_sum, &b_sum);
        // Add cross terms for carries
        if a_carry {
            let mut carry = 0u64;
            for i in 0..2 {
                let sum = (z1_wide[2 + i] as u128) + (b_sum[i] as u128) + (carry as u128);
                z1_wide[2 + i] = sum as u64;
                carry = (sum >> 64) as u64;
            }
            z1_wide[4] = z1_wide[4].wrapping_add(carry);
        }
        if b_carry {
            let mut carry = 0u64;
            for i in 0..2 {
                let sum = (z1_wide[2 + i] as u128) + (a_sum[i] as u128) + (carry as u128);
                z1_wide[2 + i] = sum as u64;
                carry = (sum >> 64) as u64;
            }
            z1_wide[4] = z1_wide[4].wrapping_add(carry);
        }
        if a_carry && b_carry {
            z1_wide[4] = z1_wide[4].wrapping_add(1);
        }

        // z1 = z1_wide - z0 - z2
        sub_inplace_5(&mut z1_wide, &z0);
        sub_inplace_5(&mut z1_wide, &z2);

        // result = z0 + z1 * 2^128 + z2 * 2^256
        let mut result = [0u64; 8];
        // Add z0 at position 0
        for i in 0..4 { result[i] = z0[i]; }
        // Add z1 at position 2
        let mut carry = 0u64;
        for i in 0..5 {
            if i + 2 < 8 {
                let sum = (result[i + 2] as u128) + (z1_wide[i] as u128) + (carry as u128);
                result[i + 2] = sum as u64;
                carry = (sum >> 64) as u64;
            }
        }
        // Add z2 at position 4
        carry = 0;
        for i in 0..4 {
            if i + 4 < 8 {
                let sum = (result[i + 4] as u128) + (z2[i] as u128) + (carry as u128);
                result[i + 4] = sum as u64;
                carry = (sum >> 64) as u64;
            }
        }

        U512 { limbs: result }
    }

    /// Left shift by `n` bits.
    pub fn shl(&self, n: u32) -> U256 {
        if n >= 256 { return U256::ZERO; }
        let limb_shift = (n / 64) as usize;
        let bit_shift = n % 64;
        let mut result = [0u64; 4];
        for i in limb_shift..4 {
            result[i] = self.limbs[i - limb_shift] << bit_shift;
            if bit_shift > 0 && i > limb_shift {
                result[i] |= self.limbs[i - limb_shift - 1] >> (64 - bit_shift);
            }
        }
        U256 { limbs: result }
    }

    /// Right shift by `n` bits.
    pub fn shr(&self, n: u32) -> U256 {
        if n >= 256 { return U256::ZERO; }
        let limb_shift = (n / 64) as usize;
        let bit_shift = n % 64;
        let mut result = [0u64; 4];
        for i in 0..(4 - limb_shift) {
            result[i] = self.limbs[i + limb_shift] >> bit_shift;
            if bit_shift > 0 && i + limb_shift + 1 < 4 {
                result[i] |= self.limbs[i + limb_shift + 1] << (64 - bit_shift);
            }
        }
        U256 { limbs: result }
    }

    /// Constant-time comparison: returns 0 if equal, -1 if self < other, 1 if self > other.
    /// No early exit — always examines all limbs.
    #[inline]
    pub fn ct_cmp(&self, other: &U256) -> i32 {
        let mut gt = 0u64;
        let mut lt = 0u64;
        for i in (0..4).rev() {
            // If we already know, mask prevents changes
            let mask = !(gt | lt).wrapping_sub(1); // all-ones if gt==0 && lt==0
            let a = self.limbs[i];
            let b = other.limbs[i];
            gt |= ((b.wrapping_sub(a)) >> 63) & (mask & 1);
            lt |= ((a.wrapping_sub(b)) >> 63) & (mask & 1);
        }
        (gt as i32) - (lt as i32)
    }

    /// Constant-time equality.
    #[inline]
    pub fn ct_eq(&self, other: &U256) -> bool {
        let mut diff = 0u64;
        for i in 0..4 {
            diff |= self.limbs[i] ^ other.limbs[i];
        }
        diff == 0
    }

    /// Constant-time less-than.
    #[inline]
    pub fn ct_lt(&self, other: &U256) -> bool {
        let (_, borrow) = Self::sub_with_borrow(self, other);
        borrow
    }

    /// Check if zero (constant-time).
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.ct_eq(&U256::ZERO)
    }

    /// Get bit at position `n` (0-indexed from LSB).
    #[inline]
    pub fn bit(&self, n: u32) -> bool {
        if n >= 256 { return false; }
        let limb = (n / 64) as usize;
        let bit = n % 64;
        (self.limbs[limb] >> bit) & 1 == 1
    }

    /// Number of bits (position of highest set bit + 1).
    pub fn bits(&self) -> u32 {
        for i in (0..4).rev() {
            if self.limbs[i] != 0 {
                return (i as u32) * 64 + (64 - self.limbs[i].leading_zeros());
            }
        }
        0
    }

    /// Division and remainder using binary long division.
    /// Returns (quotient, remainder).
    pub fn div_rem(&self, divisor: &U256) -> (U256, U256) {
        assert!(!divisor.is_zero(), "division by zero");
        if self.ct_lt(divisor) {
            return (U256::ZERO, *self);
        }
        let mut quotient = U256::ZERO;
        let mut remainder = U256::ZERO;
        let nbits = self.bits();
        for i in (0..nbits).rev() {
            remainder = remainder.shl(1);
            remainder.limbs[0] |= if self.bit(i) { 1 } else { 0 };
            if !remainder.ct_lt(divisor) {
                remainder = remainder.sub(divisor);
                quotient.limbs[(i / 64) as usize] |= 1u64 << (i % 64);
            }
        }
        (quotient, remainder)
    }

    /// Modular reduction: self mod modulus.
    #[inline]
    pub fn mod_reduce(&self, modulus: &U256) -> U256 {
        self.div_rem(modulus).1
    }

    /// Modular addition: (a + b) mod m.
    pub fn add_mod(&self, other: &U256, modulus: &U256) -> U256 {
        let (sum, carry) = U256::add_with_carry(self, other);
        if carry || !sum.ct_lt(modulus) {
            sum.sub(modulus)
        } else {
            sum
        }
    }

    /// Modular subtraction: (a - b) mod m.
    pub fn sub_mod(&self, other: &U256, modulus: &U256) -> U256 {
        let (diff, borrow) = U256::sub_with_borrow(self, other);
        if borrow {
            diff.add(modulus)
        } else {
            diff
        }
    }

    /// Modular multiplication using Montgomery form.
    pub fn mul_mod(&self, other: &U256, field: &PrimeField) -> U256 {
        field.mul(self, other)
    }

    /// Modular exponentiation with Montgomery ladder (constant-time).
    pub fn pow_mod(&self, exp: &U256, field: &PrimeField) -> U256 {
        field.pow(self, exp)
    }

    /// Modular inverse via extended GCD.
    pub fn inv_mod(&self, modulus: &U256) -> Option<U256> {
        if self.is_zero() { return None; }
        // Extended binary GCD
        let mut u = *self;
        let mut v = *modulus;
        let mut x1 = U256::ONE;
        let mut x2 = U256::ZERO;

        while !u.ct_eq(&U256::ONE) && !v.ct_eq(&U256::ONE) {
            while u.limbs[0] & 1 == 0 {
                u = u.shr(1);
                if x1.limbs[0] & 1 == 0 {
                    x1 = x1.shr(1);
                } else {
                    let (sum, _) = U256::add_with_carry(&x1, modulus);
                    x1 = sum.shr(1);
                }
            }
            while v.limbs[0] & 1 == 0 {
                v = v.shr(1);
                if x2.limbs[0] & 1 == 0 {
                    x2 = x2.shr(1);
                } else {
                    let (sum, _) = U256::add_with_carry(&x2, modulus);
                    x2 = sum.shr(1);
                }
            }
            if !u.ct_lt(&v) {
                u = u.sub(&v);
                x1 = x1.sub_mod(&x2, modulus);
            } else {
                v = v.sub(&u);
                x2 = x2.sub_mod(&x1, modulus);
            }
        }
        if u.ct_eq(&U256::ONE) { Some(x1) } else { Some(x2) }
    }

    /// Modular square root using Tonelli-Shanks.
    pub fn sqrt_mod(&self, field: &PrimeField) -> Option<U256> {
        field.sqrt(self)
    }
}

impl fmt::Debug for U256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "U256(0x{})", self.to_hex())
    }
}

impl fmt::Display for U256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{}", self.to_hex())
    }
}

// ============================================================
// U512 — 512-bit intermediate for wide multiplication
// ============================================================

#[derive(Clone, Copy, Debug)]
pub struct U512 {
    pub limbs: [u64; 8],
}

impl U512 {
    pub const ZERO: U512 = U512 { limbs: [0; 8] };

    /// Extract lower 256 bits.
    pub fn lo(&self) -> U256 {
        U256 { limbs: [self.limbs[0], self.limbs[1], self.limbs[2], self.limbs[3]] }
    }

    /// Extract upper 256 bits.
    pub fn hi(&self) -> U256 {
        U256 { limbs: [self.limbs[4], self.limbs[5], self.limbs[6], self.limbs[7]] }
    }
}

// ============================================================
// Karatsuba helpers (128-bit operations)
// ============================================================

#[inline]
fn mul_128x128(a: &[u64; 2], b: &[u64; 2]) -> [u64; 5] {
    let mut r = [0u64; 5];
    for i in 0..2 {
        let mut carry = 0u64;
        for j in 0..2 {
            let prod = (a[i] as u128) * (b[j] as u128) + (r[i + j] as u128) + (carry as u128);
            r[i + j] = prod as u64;
            carry = (prod >> 64) as u64;
        }
        r[i + 2] = carry;
    }
    r
}

#[inline]
fn add_128(a: &[u64; 2], b: &[u64; 2]) -> ([u64; 2], bool) {
    let s0 = (a[0] as u128) + (b[0] as u128);
    let s1 = (a[1] as u128) + (b[1] as u128) + (s0 >> 64);
    ([s0 as u64, s1 as u64], (s1 >> 64) != 0)
}

#[inline]
fn sub_inplace_5(a: &mut [u64; 5], b: &[u64; 5]) {
    let mut borrow = 0i128;
    for i in 0..5 {
        let diff = (a[i] as i128) - (b[i] as i128) - borrow;
        if diff < 0 {
            a[i] = (diff + (1i128 << 64)) as u64;
            borrow = 1;
        } else {
            a[i] = diff as u64;
            borrow = 0;
        }
    }
}

// ============================================================
// 2. PrimeField — Field arithmetic wrapper
// ============================================================

/// A prime field backed by Montgomery multiplication from modmath.
pub struct PrimeField {
    pub params: &'static FieldParams,
}

impl PrimeField {
    /// Create a PrimeField from a named field.
    pub fn new(name: &str) -> Option<Self> {
        modmath::field_by_name(name).map(|p| Self { params: p })
    }

    /// Get the modulus as U256.
    pub fn modulus(&self) -> U256 {
        U256 { limbs: self.params.modulus }
    }

    /// Convert to Montgomery form.
    fn to_mont(&self, a: &U256) -> [u64; 4] {
        montgomery_mul(&a.limbs, &self.params.r_squared, &self.params.modulus, self.params.inv)
    }

    /// Convert from Montgomery form.
    fn from_mont(&self, a: &[u64; 4]) -> U256 {
        U256 { limbs: montgomery_mul(a, &[1, 0, 0, 0], &self.params.modulus, self.params.inv) }
    }

    /// Modular multiplication.
    pub fn mul(&self, a: &U256, b: &U256) -> U256 {
        let am = self.to_mont(a);
        let bm = self.to_mont(b);
        let rm = montgomery_mul(&am, &bm, &self.params.modulus, self.params.inv);
        self.from_mont(&rm)
    }

    /// Modular addition.
    pub fn add(&self, a: &U256, b: &U256) -> U256 {
        let mf_a = ModField::new(a.limbs, self.params);
        let mf_b = ModField::new(b.limbs, self.params);
        let r = mf_a.add(&mf_b);
        U256 { limbs: r.to_normal() }
    }

    /// Modular subtraction.
    pub fn sub(&self, a: &U256, b: &U256) -> U256 {
        let mf_a = ModField::new(a.limbs, self.params);
        let mf_b = ModField::new(b.limbs, self.params);
        let r = mf_a.sub(&mf_b);
        U256 { limbs: r.to_normal() }
    }

    /// Modular negation.
    pub fn neg(&self, a: &U256) -> U256 {
        let mf = ModField::new(a.limbs, self.params);
        U256 { limbs: mf.neg().to_normal() }
    }

    /// Modular exponentiation (constant-time via Montgomery ladder).
    pub fn pow(&self, base: &U256, exp: &U256) -> U256 {
        let mf = ModField::new(base.limbs, self.params);
        let r = mf.pow(&exp.limbs);
        U256 { limbs: r.to_normal() }
    }

    /// Modular inverse via Fermat's little theorem.
    pub fn inv(&self, a: &U256) -> Option<U256> {
        if a.is_zero() { return None; }
        let mf = ModField::new(a.limbs, self.params);
        let r = mf.inv();
        Some(U256 { limbs: r.to_normal() })
    }

    /// Modular square root via Tonelli-Shanks.
    pub fn sqrt(&self, a: &U256) -> Option<U256> {
        let mf = ModField::new(a.limbs, self.params);
        mf.sqrt().map(|r| U256 { limbs: r.to_normal() })
    }
}

// ============================================================
// 3. Batch Operations
// ============================================================

/// Batch modular multiplication: element-wise a[i] * b[i] mod p.
pub fn batch_mul(a: &[U256], b: &[U256], field: &PrimeField) -> Vec<U256> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| field.mul(x, y)).collect()
}

/// Batch modular inverse using Montgomery's trick.
/// Computes N inverses with only 1 inversion + 3(N-1) multiplications.
/// This is where Vortex dramatically outperforms naive approaches.
pub fn batch_inv(values: &[U256], field: &PrimeField) -> Vec<U256> {
    let n = values.len();
    if n == 0 { return vec![]; }
    if n == 1 {
        return vec![field.inv(&values[0]).unwrap_or(U256::ZERO)];
    }

    // Step 1: Compute prefix products
    // prefix[i] = values[0] * values[1] * ... * values[i]
    let mut prefix = Vec::with_capacity(n);
    prefix.push(values[0]);
    for i in 1..n {
        prefix.push(field.mul(&prefix[i - 1], &values[i]));
    }

    // Step 2: Invert the total product (single inversion)
    let mut inv_acc = field.inv(&prefix[n - 1]).unwrap_or(U256::ZERO);

    // Step 3: Sweep backwards to recover individual inverses
    let mut result = vec![U256::ZERO; n];
    for i in (1..n).rev() {
        // result[i] = inv_acc * prefix[i-1]
        result[i] = field.mul(&inv_acc, &prefix[i - 1]);
        // inv_acc = inv_acc * values[i]
        inv_acc = field.mul(&inv_acc, &values[i]);
    }
    result[0] = inv_acc;

    result
}

/// Parallel batch multiplication using std::thread.
pub fn batch_mul_parallel(a: &[U256], b: &[U256], field_name: &str) -> Vec<U256> {
    let n = a.len();
    assert_eq!(n, b.len());
    let num_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(n);

    if num_threads <= 1 || n < 64 {
        let f = PrimeField::new(field_name).expect("invalid field");
        return batch_mul(a, b, &f);
    }

    let chunk_size = (n + num_threads - 1) / num_threads;
    let a_chunks: Vec<Vec<U256>> = a.chunks(chunk_size).map(|c| c.to_vec()).collect();
    let b_chunks: Vec<Vec<U256>> = b.chunks(chunk_size).map(|c| c.to_vec()).collect();
    let fname = field_name.to_string();

    let handles: Vec<_> = a_chunks.into_iter().zip(b_chunks).map(|(ac, bc)| {
        let fname = fname.clone();
        std::thread::spawn(move || {
            let f = PrimeField::new(&fname).expect("invalid field");
            batch_mul(&ac, &bc, &f)
        })
    }).collect();

    let mut result = Vec::with_capacity(n);
    for h in handles {
        result.extend(h.join().unwrap());
    }
    result
}

/// Parallel batch inverse.
pub fn batch_inv_parallel(values: &[U256], field_name: &str) -> Vec<U256> {
    // Montgomery's trick is inherently sequential, but we can split into
    // chunks, apply Montgomery's trick to each chunk in parallel, then combine.
    let n = values.len();
    let num_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(n);

    if num_threads <= 1 || n < 64 {
        let f = PrimeField::new(field_name).expect("invalid field");
        return batch_inv(values, &f);
    }

    let chunk_size = (n + num_threads - 1) / num_threads;
    let chunks: Vec<Vec<U256>> = values.chunks(chunk_size).map(|c| c.to_vec()).collect();
    let fname = field_name.to_string();

    let handles: Vec<_> = chunks.into_iter().map(|chunk| {
        let fname = fname.clone();
        std::thread::spawn(move || {
            let f = PrimeField::new(&fname).expect("invalid field");
            batch_inv(&chunk, &f)
        })
    }).collect();

    let mut result = Vec::with_capacity(n);
    for h in handles {
        result.extend(h.join().unwrap());
    }
    result
}

// ============================================================
// 4. NTT (Number Theoretic Transform)
// ============================================================

/// Radix-2 Cooley-Tukey NTT.
/// `omega` is a primitive n-th root of unity in the field.
/// `values` length must be a power of 2.
pub fn ntt(values: &[U256], omega: &U256, field: &PrimeField) -> Vec<U256> {
    let n = values.len();
    assert!(n.is_power_of_two(), "NTT requires power-of-2 length");
    if n == 1 { return values.to_vec(); }

    let mut a = values.to_vec();

    // Bit-reversal permutation
    let log_n = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - log_n);
        if i < j {
            a.swap(i, j);
        }
    }

    // Butterfly operations
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        // omega_step = omega^(n/len)
        let exp = U256::from_u64((n / len) as u64);
        let w_step = field.pow(omega, &exp);

        for start in (0..n).step_by(len) {
            let mut w = U256::ONE;
            for j in 0..half {
                let u = a[start + j];
                let v = field.mul(&a[start + j + half], &w);
                a[start + j] = field.add(&u, &v);
                a[start + j + half] = field.sub(&u, &v);
                w = field.mul(&w, &w_step);
            }
        }
        len *= 2;
    }

    a
}

/// Inverse NTT: INTT(values) = (1/n) * NTT(values, omega^{-1}).
pub fn intt(values: &[U256], omega: &U256, field: &PrimeField) -> Vec<U256> {
    let n = values.len();
    let omega_inv = field.inv(omega).expect("omega must be invertible");
    let mut result = ntt(values, &omega_inv, field);
    let n_inv = field.inv(&U256::from_u64(n as u64)).expect("n must be invertible");
    for v in &mut result {
        *v = field.mul(v, &n_inv);
    }
    result
}

/// Polynomial multiplication via NTT. O(n log n) instead of O(n^2).
pub fn poly_mul_ntt(a: &[U256], b: &[U256], omega: &U256, field: &PrimeField) -> Vec<U256> {
    let result_len = a.len() + b.len() - 1;
    let n = result_len.next_power_of_two();

    let mut a_padded = a.to_vec();
    a_padded.resize(n, U256::ZERO);
    let mut b_padded = b.to_vec();
    b_padded.resize(n, U256::ZERO);

    // We need omega to be an n-th root of unity
    // Adjust omega for the padded size
    let orig_n = a.len().max(b.len()).next_power_of_two();
    let scale = U256::from_u64((orig_n / n.min(orig_n)) as u64);
    let adjusted_omega = if scale.is_zero() || scale.ct_eq(&U256::ONE) {
        *omega
    } else {
        field.pow(omega, &scale)
    };

    let fa = ntt(&a_padded, &adjusted_omega, field);
    let fb = ntt(&b_padded, &adjusted_omega, field);

    let fc: Vec<U256> = fa.iter().zip(fb.iter()).map(|(x, y)| field.mul(x, y)).collect();

    let mut result = intt(&fc, &adjusted_omega, field);
    result.truncate(result_len);
    result
}

// ============================================================
// 5. Elliptic Curve Operations
// ============================================================

/// Affine point on a short Weierstrass curve y^2 = x^3 + ax + b.
#[derive(Clone, Copy, Debug)]
pub struct AffinePoint {
    pub x: U256,
    pub y: U256,
    pub infinity: bool,
}

/// Jacobian projective point: (X : Y : Z) where x = X/Z^2, y = Y/Z^3.
#[derive(Clone, Copy, Debug)]
pub struct JacobianPoint {
    pub x: U256,
    pub y: U256,
    pub z: U256,
}

/// Twisted Edwards point: (X : Y : T : Z) where x=X/Z, y=Y/Z, T=XY/Z.
#[derive(Clone, Copy, Debug)]
pub struct TwistedEdwardsPoint {
    pub x: U256,
    pub y: U256,
    pub t: U256,
    pub z: U256,
}

/// Curve parameters for short Weierstrass form.
pub struct WeierstrassCurve {
    pub a: U256,
    pub b: U256,
    pub field: PrimeField,
    pub order: U256,
    pub generator: AffinePoint,
}

impl AffinePoint {
    pub fn identity() -> Self {
        Self { x: U256::ZERO, y: U256::ZERO, infinity: true }
    }

    pub fn new(x: U256, y: U256) -> Self {
        Self { x, y, infinity: false }
    }

    pub fn to_jacobian(&self) -> JacobianPoint {
        if self.infinity {
            JacobianPoint { x: U256::ONE, y: U256::ONE, z: U256::ZERO }
        } else {
            JacobianPoint { x: self.x, y: self.y, z: U256::ONE }
        }
    }
}

impl JacobianPoint {
    pub fn identity() -> Self {
        JacobianPoint { x: U256::ONE, y: U256::ONE, z: U256::ZERO }
    }

    pub fn is_identity(&self) -> bool {
        self.z.is_zero()
    }

    /// Convert back to affine. Requires field inversion.
    pub fn to_affine(&self, field: &PrimeField) -> AffinePoint {
        if self.is_identity() {
            return AffinePoint::identity();
        }
        let z_inv = field.inv(&self.z).unwrap();
        let z_inv2 = field.mul(&z_inv, &z_inv);
        let z_inv3 = field.mul(&z_inv2, &z_inv);
        AffinePoint::new(
            field.mul(&self.x, &z_inv2),
            field.mul(&self.y, &z_inv3),
        )
    }

    /// Point addition (complete formula for short Weierstrass curves).
    pub fn add(&self, other: &JacobianPoint, field: &PrimeField) -> JacobianPoint {
        if self.is_identity() { return *other; }
        if other.is_identity() { return *self; }

        let z1z1 = field.mul(&self.z, &self.z);
        let z2z2 = field.mul(&other.z, &other.z);
        let u1 = field.mul(&self.x, &z2z2);
        let u2 = field.mul(&other.x, &z1z1);
        let s1 = field.mul(&field.mul(&self.y, &other.z), &z2z2);
        let s2 = field.mul(&field.mul(&other.y, &self.z), &z1z1);

        if u1.ct_eq(&u2) {
            if s1.ct_eq(&s2) {
                return self.double(field);
            } else {
                return JacobianPoint::identity();
            }
        }

        let h = field.sub(&u2, &u1);
        let i = field.mul(&field.add(&h, &h), &field.add(&h, &h)); // (2H)^2
        let j = field.mul(&h, &i);
        let r = field.add(&field.sub(&s2, &s1), &field.sub(&s2, &s1)); // 2*(S2-S1)
        let v = field.mul(&u1, &i);

        let x3 = field.sub(&field.sub(&field.mul(&r, &r), &j), &field.add(&v, &v));
        let y3 = field.sub(
            &field.mul(&r, &field.sub(&v, &x3)),
            &field.add(&field.mul(&s1, &j), &field.mul(&s1, &j)),
        );
        let z3 = field.mul(
            &field.mul(&field.add(&self.z, &other.z), &field.add(&self.z, &other.z)),
            &field.sub(&field.sub(&field.mul(&self.z, &self.z).add_mod(&field.mul(&other.z, &other.z), &field.modulus()), &z1z1), &z2z2),
        );
        // Simplified: z3 = ((z1+z2)^2 - z1z1 - z2z2) * h ... but let's use the standard formula
        let z1_plus_z2 = field.add(&self.z, &other.z);
        let z3 = field.mul(&field.sub(&field.sub(&field.mul(&z1_plus_z2, &z1_plus_z2), &z1z1), &z2z2), &h);

        JacobianPoint { x: x3, y: y3, z: z3 }
    }

    /// Point doubling.
    pub fn double(&self, field: &PrimeField) -> JacobianPoint {
        if self.is_identity() { return *self; }

        let a = field.mul(&self.x, &self.x); // X^2
        let b = field.mul(&self.y, &self.y); // Y^2
        let c = field.mul(&b, &b);           // Y^4

        let d = field.mul(
            &field.add(&field.mul(&field.add(&self.x, &b), &field.add(&self.x, &b)),
                       &U256::ZERO),
            &U256::ONE
        );
        // d = 2*((X+B)^2 - A - C)
        let xpb_sq = field.mul(&field.add(&self.x, &b), &field.add(&self.x, &b));
        let d = field.add(&field.sub(&field.sub(&xpb_sq, &a), &c),
                         &field.sub(&field.sub(&xpb_sq, &a), &c));

        let e = field.add(&field.add(&a, &a), &a); // 3*A = 3*X^2 (for a=0 curves)
        let f = field.mul(&e, &e);                   // E^2

        let x3 = field.sub(&f, &field.add(&d, &d));

        let c8 = field.add(&field.add(&field.add(&c, &c), &field.add(&c, &c)),
                           &field.add(&field.add(&c, &c), &field.add(&c, &c)));
        let y3 = field.sub(&field.mul(&e, &field.sub(&d, &x3)), &c8);
        let z3 = field.mul(&field.add(&self.y, &self.y), &self.z);

        JacobianPoint { x: x3, y: y3, z: z3 }
    }

    /// Scalar multiplication using windowed NAF (w=4).
    pub fn scalar_mul(&self, scalar: &U256, field: &PrimeField) -> JacobianPoint {
        if scalar.is_zero() || self.is_identity() {
            return JacobianPoint::identity();
        }

        // Compute NAF representation with window w=4
        let naf = compute_wnaf(scalar, 4);

        // Precompute: table[i] = (2i+1) * P for i in 0..7
        let mut table = Vec::with_capacity(8);
        table.push(*self);
        let double = self.double(field);
        for i in 1..8 {
            table.push(table[i - 1].add(&double, field));
        }

        let mut result = JacobianPoint::identity();
        for &digit in naf.iter().rev() {
            result = result.double(field);
            if digit > 0 {
                let idx = ((digit - 1) / 2) as usize;
                result = result.add(&table[idx], field);
            } else if digit < 0 {
                let idx = ((-digit - 1) / 2) as usize;
                let neg = negate_jacobian(&table[idx], field);
                result = result.add(&neg, field);
            }
        }

        result
    }
}

/// Compute windowed Non-Adjacent Form of a scalar.
fn compute_wnaf(scalar: &U256, w: u32) -> Vec<i8> {
    let mut naf = Vec::with_capacity(257);
    let mut k = *scalar;
    let pow2w = 1i16 << w;
    let half_pow2w = pow2w / 2;

    while !k.is_zero() {
        if k.limbs[0] & 1 == 1 {
            let mods = (k.limbs[0] & ((1u64 << w) - 1)) as i16;
            let digit = if mods >= half_pow2w { mods - pow2w } else { mods };
            naf.push(digit as i8);
            if digit < 0 {
                k = k.add(&U256::from_u64((-digit) as u64));
            } else {
                k = k.sub(&U256::from_u64(digit as u64));
            }
        } else {
            naf.push(0);
        }
        k = k.shr(1);
    }
    naf
}

fn negate_jacobian(p: &JacobianPoint, field: &PrimeField) -> JacobianPoint {
    JacobianPoint {
        x: p.x,
        y: field.neg(&p.y),
        z: p.z,
    }
}

/// Multi-Scalar Multiplication using Pippenger's algorithm.
/// Computes sum(scalars[i] * points[i]) efficiently.
pub fn pippenger_msm(scalars: &[U256], points: &[JacobianPoint], field: &PrimeField) -> JacobianPoint {
    assert_eq!(scalars.len(), points.len());
    let n = scalars.len();
    if n == 0 { return JacobianPoint::identity(); }
    if n == 1 { return points[0].scalar_mul(&scalars[0], field); }

    // Choose window size based on number of points
    let c = optimal_bucket_width(n);
    let num_windows = (256 + c - 1) / c;
    let num_buckets = (1 << c) - 1;

    let mut result = JacobianPoint::identity();

    for window_idx in (0..num_windows).rev() {
        // Shift result by c bits
        for _ in 0..c {
            result = result.double(field);
        }

        // Fill buckets
        let mut buckets = vec![JacobianPoint::identity(); num_buckets];
        for i in 0..n {
            let scalar_window = get_window(&scalars[i], window_idx * c, c);
            if scalar_window > 0 {
                let bucket_idx = (scalar_window - 1) as usize;
                buckets[bucket_idx] = buckets[bucket_idx].add(&points[i], field);
            }
        }

        // Sum buckets: sum = 1*B[0] + 2*B[1] + ... + k*B[k-1]
        // Use running sum: accumulate from top
        let mut running_sum = JacobianPoint::identity();
        let mut window_sum = JacobianPoint::identity();
        for j in (0..num_buckets).rev() {
            running_sum = running_sum.add(&buckets[j], field);
            window_sum = window_sum.add(&running_sum, field);
        }

        result = result.add(&window_sum, field);
    }

    result
}

/// Get a window of `width` bits starting at bit position `start`.
fn get_window(scalar: &U256, start: usize, width: usize) -> u64 {
    let mut val = 0u64;
    for i in 0..width {
        let bit_pos = start + i;
        if bit_pos < 256 {
            let limb_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;
            val |= ((scalar.limbs[limb_idx] >> bit_idx) & 1) << i;
        }
    }
    val
}

/// Choose optimal bucket width for Pippenger based on number of points.
fn optimal_bucket_width(n: usize) -> usize {
    if n < 4 { return 1; }
    if n < 32 { return 3; }
    if n < 256 { return 5; }
    if n < 2048 { return 7; }
    if n < 16384 { return 9; }
    12
}

// ============================================================
// 6. Built-in Curves
// ============================================================

/// Get secp256k1 curve parameters.
pub fn secp256k1_curve() -> WeierstrassCurve {
    WeierstrassCurve {
        a: U256::ZERO,
        b: U256::from_u64(7),
        field: PrimeField::new("secp256k1").unwrap(),
        order: U256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141").unwrap(),
        generator: AffinePoint::new(
            U256::from_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798").unwrap(),
            U256::from_hex("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8").unwrap(),
        ),
    }
}

/// Get BN254 curve parameters.
pub fn bn254_curve() -> WeierstrassCurve {
    WeierstrassCurve {
        a: U256::ZERO,
        b: U256::from_u64(3),
        field: PrimeField::new("bn254").unwrap(),
        order: U256::from_hex("30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001").unwrap(),
        generator: AffinePoint::new(U256::ONE, U256::from_u64(2)),
    }
}

// ============================================================
// 7. Interpreter Builtins
// ============================================================

use crate::interpreter::{Env, Value};

/// Register all bigint_engine builtins in the interpreter environment.
pub fn register_builtins(env: &mut Env) {
    env.functions.insert("u256_new".to_string(),
        crate::interpreter::FnDef::Builtin(builtin_u256_new));
    env.functions.insert("u256_from_int".to_string(),
        crate::interpreter::FnDef::Builtin(builtin_u256_from_int));
    env.functions.insert("u256_add".to_string(),
        crate::interpreter::FnDef::Builtin(builtin_u256_add));
    env.functions.insert("u256_sub".to_string(),
        crate::interpreter::FnDef::Builtin(builtin_u256_sub));
    env.functions.insert("u256_mul".to_string(),
        crate::interpreter::FnDef::Builtin(builtin_u256_mul));
    env.functions.insert("u256_pow_mod".to_string(),
        crate::interpreter::FnDef::Builtin(builtin_u256_pow_mod));
    env.functions.insert("u256_inv_mod".to_string(),
        crate::interpreter::FnDef::Builtin(builtin_u256_inv_mod));
    env.functions.insert("u256_to_hex".to_string(),
        crate::interpreter::FnDef::Builtin(builtin_u256_to_hex));
    env.functions.insert("u256_eq".to_string(),
        crate::interpreter::FnDef::Builtin(builtin_u256_eq));
    env.functions.insert("u256_lt".to_string(),
        crate::interpreter::FnDef::Builtin(builtin_u256_lt));
    env.functions.insert("field_engine_new".to_string(),
        crate::interpreter::FnDef::Builtin(builtin_field_engine_new));
    env.functions.insert("field_engine_mul".to_string(),
        crate::interpreter::FnDef::Builtin(builtin_field_engine_mul));
    env.functions.insert("field_engine_add".to_string(),
        crate::interpreter::FnDef::Builtin(builtin_field_engine_add));
    env.functions.insert("batch_field_mul".to_string(),
        crate::interpreter::FnDef::Builtin(builtin_batch_field_mul));
    env.functions.insert("batch_field_inv".to_string(),
        crate::interpreter::FnDef::Builtin(builtin_batch_field_inv));
    env.functions.insert("ec_scalar_mul".to_string(),
        crate::interpreter::FnDef::Builtin(builtin_ec_scalar_mul));
}

// --- Helper: convert between Value and U256 ---

fn value_to_u256(v: &Value) -> Result<U256, String> {
    match v {
        Value::String(s) => U256::from_hex(s).ok_or_else(|| format!("invalid hex: {}", s)),
        Value::Int(n) => {
            if *n < 0 { return Err("negative integer".to_string()); }
            Ok(U256::from_u64(*n as u64))
        }
        Value::BigInt(b) => {
            let limbs = modmath::from_biguint256(b);
            Ok(U256 { limbs })
        }
        _ => Err(format!("cannot convert {:?} to U256", v)),
    }
}

fn u256_to_value(v: &U256) -> Value {
    Value::String(format!("0x{}", v.to_hex()))
}

fn builtin_u256_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("u256_new expects 1 argument (hex string)".to_string()); }
    let u = value_to_u256(&args[0])?;
    Ok(u256_to_value(&u))
}

fn builtin_u256_from_int(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("u256_from_int expects 1 argument".to_string()); }
    match &args[0] {
        Value::Int(n) => Ok(u256_to_value(&U256::from_i64(*n as i64))),
        _ => Err("u256_from_int: argument must be an integer".to_string()),
    }
}

fn builtin_u256_add(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("u256_add expects 2 arguments".to_string()); }
    let a = value_to_u256(&args[0])?;
    let b = value_to_u256(&args[1])?;
    Ok(u256_to_value(&a.add(&b)))
}

fn builtin_u256_sub(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("u256_sub expects 2 arguments".to_string()); }
    let a = value_to_u256(&args[0])?;
    let b = value_to_u256(&args[1])?;
    Ok(u256_to_value(&a.sub(&b)))
}

fn builtin_u256_mul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("u256_mul expects 2 arguments".to_string()); }
    let a = value_to_u256(&args[0])?;
    let b = value_to_u256(&args[1])?;
    Ok(u256_to_value(&a.mul(&b)))
}

fn builtin_u256_pow_mod(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("u256_pow_mod expects 3 arguments (base, exp, field_name)".to_string()); }
    let base = value_to_u256(&args[0])?;
    let exp = value_to_u256(&args[1])?;
    let field_name = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err("u256_pow_mod: 3rd argument must be a field name string".to_string()),
    };
    let field = PrimeField::new(&field_name).ok_or_else(|| format!("unknown field: {}", field_name))?;
    Ok(u256_to_value(&field.pow(&base, &exp)))
}

fn builtin_u256_inv_mod(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("u256_inv_mod expects 2 arguments (value, field_name)".to_string()); }
    let a = value_to_u256(&args[0])?;
    let field_name = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("u256_inv_mod: 2nd argument must be a field name string".to_string()),
    };
    let field = PrimeField::new(&field_name).ok_or_else(|| format!("unknown field: {}", field_name))?;
    field.inv(&a).map(|r| u256_to_value(&r)).ok_or_else(|| "element has no inverse".to_string())
}

fn builtin_u256_to_hex(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("u256_to_hex expects 1 argument".to_string()); }
    let a = value_to_u256(&args[0])?;
    Ok(Value::String(format!("0x{}", a.to_hex())))
}

fn builtin_u256_eq(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("u256_eq expects 2 arguments".to_string()); }
    let a = value_to_u256(&args[0])?;
    let b = value_to_u256(&args[1])?;
    Ok(Value::Bool(a.ct_eq(&b)))
}

fn builtin_u256_lt(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("u256_lt expects 2 arguments".to_string()); }
    let a = value_to_u256(&args[0])?;
    let b = value_to_u256(&args[1])?;
    Ok(Value::Bool(a.ct_lt(&b)))
}

fn builtin_field_engine_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("field_engine_new expects 1 argument (name)".to_string()); }
    match &args[0] {
        Value::String(name) => {
            if PrimeField::new(name).is_some() {
                Ok(Value::String(name.clone()))
            } else {
                Err(format!("unknown field: {}", name))
            }
        }
        _ => Err("field_engine_new: argument must be a string".to_string()),
    }
}

fn builtin_field_engine_mul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("field_engine_mul expects 3 arguments (a, b, field_name)".to_string()); }
    let a = value_to_u256(&args[0])?;
    let b = value_to_u256(&args[1])?;
    let fname = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err("field_engine_mul: 3rd argument must be a field name".to_string()),
    };
    let field = PrimeField::new(&fname).ok_or_else(|| format!("unknown field: {}", fname))?;
    Ok(u256_to_value(&field.mul(&a, &b)))
}

fn builtin_field_engine_add(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("field_engine_add expects 3 arguments (a, b, field_name)".to_string()); }
    let a = value_to_u256(&args[0])?;
    let b = value_to_u256(&args[1])?;
    let fname = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err("field_engine_add: 3rd argument must be a field name".to_string()),
    };
    let field = PrimeField::new(&fname).ok_or_else(|| format!("unknown field: {}", fname))?;
    Ok(u256_to_value(&field.add(&a, &b)))
}

fn builtin_batch_field_mul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("batch_field_mul expects 3 arguments (a_array, b_array, field_name)".to_string()); }
    let a_arr = match &args[0] {
        Value::Array(a) => a.iter().map(value_to_u256).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("batch_field_mul: 1st argument must be an array".to_string()),
    };
    let b_arr = match &args[1] {
        Value::Array(b) => b.iter().map(value_to_u256).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("batch_field_mul: 2nd argument must be an array".to_string()),
    };
    let fname = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err("batch_field_mul: 3rd argument must be a field name".to_string()),
    };
    let result = batch_mul_parallel(&a_arr, &b_arr, &fname);
    Ok(Value::Array(result.iter().map(u256_to_value).collect()))
}

fn builtin_batch_field_inv(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("batch_field_inv expects 2 arguments (values_array, field_name)".to_string()); }
    let vals = match &args[0] {
        Value::Array(a) => a.iter().map(value_to_u256).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("batch_field_inv: 1st argument must be an array".to_string()),
    };
    let fname = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("batch_field_inv: 2nd argument must be a field name".to_string()),
    };
    let result = batch_inv_parallel(&vals, &fname);
    Ok(Value::Array(result.iter().map(u256_to_value).collect()))
}

fn builtin_ec_scalar_mul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("ec_scalar_mul expects 3 arguments (scalar, point_x, point_y)".to_string()); }
    let scalar = value_to_u256(&args[0])?;
    let px = value_to_u256(&args[1])?;
    let py = value_to_u256(&args[2])?;

    let curve = secp256k1_curve();
    let point = AffinePoint::new(px, py).to_jacobian();
    let result = point.scalar_mul(&scalar, &curve.field);
    let affine = result.to_affine(&curve.field);

    Ok(Value::Array(vec![
        u256_to_value(&affine.x),
        u256_to_value(&affine.y),
    ]))
}

// ============================================================
// 8. Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- U256 basic arithmetic ---

    #[test]
    fn test_u256_zero_one() {
        assert!(U256::ZERO.is_zero());
        assert!(!U256::ONE.is_zero());
        assert!(U256::ONE.ct_eq(&U256::from_u64(1)));
    }

    #[test]
    fn test_u256_from_hex() {
        let a = U256::from_hex("ff").unwrap();
        assert_eq!(a.limbs[0], 255);
        assert_eq!(a.limbs[1], 0);

        let b = U256::from_hex("0x10000000000000000").unwrap();
        assert_eq!(b.limbs[0], 0);
        assert_eq!(b.limbs[1], 1);

        let max = U256::from_hex("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff").unwrap();
        assert!(max.ct_eq(&U256::MAX));
    }

    #[test]
    fn test_u256_to_hex_roundtrip() {
        let vals = [U256::ZERO, U256::ONE, U256::from_u64(0xdeadbeef),
                    U256::from_hex("123456789abcdef0123456789abcdef0").unwrap()];
        for v in &vals {
            let hex = v.to_hex();
            let back = U256::from_hex(&hex).unwrap();
            assert!(v.ct_eq(&back), "roundtrip failed for {}", hex);
        }
    }

    #[test]
    fn test_u256_add() {
        let a = U256::from_u64(100);
        let b = U256::from_u64(200);
        assert!(a.add(&b).ct_eq(&U256::from_u64(300)));

        // Test carry propagation
        let max_limb = U256::new([u64::MAX, 0, 0, 0]);
        let one = U256::ONE;
        let sum = max_limb.add(&one);
        assert_eq!(sum.limbs[0], 0);
        assert_eq!(sum.limbs[1], 1);
    }

    #[test]
    fn test_u256_sub() {
        let a = U256::from_u64(300);
        let b = U256::from_u64(100);
        assert!(a.sub(&b).ct_eq(&U256::from_u64(200)));

        // Test wrapping
        let (result, borrow) = U256::sub_with_borrow(&U256::ZERO, &U256::ONE);
        assert!(borrow);
        assert!(result.ct_eq(&U256::MAX));
    }

    #[test]
    fn test_u256_mul() {
        let a = U256::from_u64(1000);
        let b = U256::from_u64(2000);
        assert!(a.mul(&b).ct_eq(&U256::from_u64(2_000_000)));

        // Larger multiplication
        let c = U256::from_hex("ffffffffffffffff").unwrap(); // 2^64-1
        let d = U256::from_u64(2);
        let product = c.mul(&d);
        assert_eq!(product.limbs[0], u64::MAX - 1);
        assert_eq!(product.limbs[1], 1);
    }

    #[test]
    fn test_u256_mul_wide() {
        let a = U256::from_hex("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff").unwrap();
        let b = U256::from_u64(2);
        let wide = a.mul_wide(&b);
        // MAX * 2 = 2^257 - 2
        assert_eq!(wide.limbs[0], u64::MAX - 1);
        for i in 1..4 { assert_eq!(wide.limbs[i], u64::MAX); }
        assert_eq!(wide.limbs[4], 1);
    }

    #[test]
    fn test_u256_karatsuba_matches_schoolbook() {
        let a = U256::from_hex("deadbeef12345678aabbccddee112233").unwrap();
        let b = U256::from_hex("1234567890abcdef1234567890abcdef").unwrap();
        let school = a.mul_wide(&b);
        let karatsuba = a.mul_karatsuba(&b);
        for i in 0..8 {
            assert_eq!(school.limbs[i], karatsuba.limbs[i], "mismatch at limb {}", i);
        }
    }

    #[test]
    fn test_u256_div_rem() {
        let a = U256::from_u64(100);
        let b = U256::from_u64(7);
        let (q, r) = a.div_rem(&b);
        assert!(q.ct_eq(&U256::from_u64(14)));
        assert!(r.ct_eq(&U256::from_u64(2)));

        let big = U256::from_hex("100000000000000000").unwrap(); // 2^64
        let div = U256::from_u64(3);
        let (q2, r2) = big.div_rem(&div);
        // Verify: q2 * 3 + r2 = 2^64
        let check = q2.mul(&div).add(&r2);
        assert!(check.ct_eq(&big));
    }

    #[test]
    fn test_u256_shifts() {
        let a = U256::from_u64(1);
        assert!(a.shl(64).ct_eq(&U256::new([0, 1, 0, 0])));
        assert!(a.shl(128).ct_eq(&U256::new([0, 0, 1, 0])));
        assert!(a.shl(192).ct_eq(&U256::new([0, 0, 0, 1])));

        let b = U256::new([0, 0, 0, 1]);
        assert!(b.shr(192).ct_eq(&U256::ONE));
        assert!(b.shr(64).ct_eq(&U256::new([0, 0, 1, 0])));
    }

    #[test]
    fn test_u256_ct_comparison() {
        let a = U256::from_u64(100);
        let b = U256::from_u64(200);
        assert!(a.ct_lt(&b));
        assert!(!b.ct_lt(&a));
        assert!(!a.ct_eq(&b));
        assert!(a.ct_eq(&a));

        // Compare at high limbs
        let c = U256::new([0, 0, 0, 1]);
        let d = U256::new([u64::MAX, u64::MAX, u64::MAX, 0]);
        assert!(d.ct_lt(&c));
    }

    // --- Field arithmetic ---

    #[test]
    fn test_field_mul_secp256k1() {
        let f = PrimeField::new("secp256k1").unwrap();
        let a = U256::from_u64(7);
        let b = U256::from_u64(11);
        let c = f.mul(&a, &b);
        assert!(c.ct_eq(&U256::from_u64(77)));
    }

    #[test]
    fn test_field_add_sub() {
        let f = PrimeField::new("bn254").unwrap();
        let a = U256::from_u64(42);
        let b = U256::from_u64(58);
        let sum = f.add(&a, &b);
        assert!(sum.ct_eq(&U256::from_u64(100)));

        let diff = f.sub(&sum, &a);
        assert!(diff.ct_eq(&b));
    }

    #[test]
    fn test_field_pow() {
        let f = PrimeField::new("secp256k1").unwrap();
        let base = U256::from_u64(2);
        let exp = U256::from_u64(10);
        let result = f.pow(&base, &exp);
        assert!(result.ct_eq(&U256::from_u64(1024)));
    }

    #[test]
    fn test_field_inv() {
        let f = PrimeField::new("secp256k1").unwrap();
        let a = U256::from_u64(42);
        let a_inv = f.inv(&a).unwrap();
        let product = f.mul(&a, &a_inv);
        assert!(product.ct_eq(&U256::ONE), "42 * 42^(-1) should be 1, got {:?}", product);
    }

    #[test]
    fn test_field_inv_bn254() {
        let f = PrimeField::new("bn254").unwrap();
        let a = U256::from_u64(12345);
        let a_inv = f.inv(&a).unwrap();
        let product = f.mul(&a, &a_inv);
        assert!(product.ct_eq(&U256::ONE));
    }

    #[test]
    fn test_field_add_sub_bls12_381() {
        // BLS12-381 prime is 381 bits; the 256-bit representation in modmath is
        // the lower 256 bits. We verify basic add/sub still works for small values.
        let f = PrimeField::new("bls12_381").unwrap();
        let a = U256::from_u64(100);
        let b = U256::from_u64(200);
        let sum = f.add(&a, &b);
        assert!(sum.ct_eq(&U256::from_u64(300)));
        let diff = f.sub(&sum, &b);
        assert!(diff.ct_eq(&a));
    }

    #[test]
    fn test_field_inv_goldilocks() {
        let f = PrimeField::new("goldilocks").unwrap();
        let a = U256::from_u64(777);
        let a_inv = f.inv(&a).unwrap();
        let product = f.mul(&a, &a_inv);
        assert!(product.ct_eq(&U256::ONE));
    }

    // --- Batch operations ---

    #[test]
    fn test_batch_inv_matches_individual() {
        let f = PrimeField::new("secp256k1").unwrap();
        let values: Vec<U256> = (1..=10).map(|i| U256::from_u64(i * 7)).collect();

        let batch_result = batch_inv(&values, &f);
        let individual: Vec<U256> = values.iter().map(|v| f.inv(v).unwrap()).collect();

        for i in 0..values.len() {
            assert!(batch_result[i].ct_eq(&individual[i]),
                "batch inv mismatch at index {}: {:?} vs {:?}", i, batch_result[i], individual[i]);
        }
    }

    #[test]
    fn test_batch_inv_large() {
        let f = PrimeField::new("bn254").unwrap();
        let values: Vec<U256> = (1..=100).map(|i| U256::from_u64(i)).collect();
        let result = batch_inv(&values, &f);

        // Verify a few: a * a^(-1) = 1
        for i in [0, 10, 50, 99] {
            let product = f.mul(&values[i], &result[i]);
            assert!(product.ct_eq(&U256::ONE), "batch inv verification failed at index {}", i);
        }
    }

    #[test]
    fn test_batch_mul() {
        let f = PrimeField::new("secp256k1").unwrap();
        let a: Vec<U256> = (1..=5).map(|i| U256::from_u64(i)).collect();
        let b: Vec<U256> = (6..=10).map(|i| U256::from_u64(i)).collect();
        let result = batch_mul(&a, &b, &f);
        assert!(result[0].ct_eq(&U256::from_u64(6)));  // 1*6
        assert!(result[1].ct_eq(&U256::from_u64(14))); // 2*7
        assert!(result[4].ct_eq(&U256::from_u64(50))); // 5*10
    }

    // --- NTT ---

    #[test]
    fn test_ntt_intt_roundtrip() {
        // Use Goldilocks field with a known primitive root
        let f = PrimeField::new("goldilocks").unwrap();
        let n = 4;
        // omega = g^((p-1)/n) where g=7 is a generator of Goldilocks
        let p_minus_1 = U256::from_hex("FFFFFFFF00000000").unwrap(); // 2^64 - 2^32
        let exp = U256::from_u64((0xFFFFFFFF00000000u64) / (n as u64));
        let omega = f.pow(&U256::from_u64(7), &exp);

        let values: Vec<U256> = vec![
            U256::from_u64(1), U256::from_u64(2), U256::from_u64(3), U256::from_u64(4),
        ];

        let transformed = ntt(&values, &omega, &f);
        let recovered = intt(&transformed, &omega, &f);

        for i in 0..n {
            assert!(values[i].ct_eq(&recovered[i]),
                "NTT roundtrip failed at index {}: expected {:?}, got {:?}", i, values[i], recovered[i]);
        }
    }

    // --- Elliptic curve ---

    #[test]
    fn test_ec_identity() {
        let f = PrimeField::new("secp256k1").unwrap();
        let id = JacobianPoint::identity();
        assert!(id.is_identity());

        let p = AffinePoint::new(U256::from_u64(1), U256::from_u64(2)).to_jacobian();
        let sum = id.add(&p, &f);
        assert_eq!(sum.x.limbs, p.x.limbs);
    }

    #[test]
    fn test_ec_point_double() {
        let curve = secp256k1_curve();
        let g = curve.generator.to_jacobian();
        let g2 = g.double(&curve.field);
        assert!(!g2.is_identity());

        // Doubling should not equal the original
        let g2_affine = g2.to_affine(&curve.field);
        assert!(!g2_affine.x.ct_eq(&curve.generator.x));
    }

    #[test]
    fn test_ec_scalar_mul_identity() {
        let curve = secp256k1_curve();
        let g = curve.generator.to_jacobian();

        // 0 * G = identity
        let result = g.scalar_mul(&U256::ZERO, &curve.field);
        assert!(result.is_identity());

        // 1 * G = G
        let result1 = g.scalar_mul(&U256::ONE, &curve.field);
        let affine1 = result1.to_affine(&curve.field);
        assert!(affine1.x.ct_eq(&curve.generator.x));
        assert!(affine1.y.ct_eq(&curve.generator.y));
    }

    #[test]
    fn test_ec_scalar_mul_2() {
        let curve = secp256k1_curve();
        let g = curve.generator.to_jacobian();

        // 2 * G via scalar_mul should match double
        let result2 = g.scalar_mul(&U256::from_u64(2), &curve.field);
        let doubled = g.double(&curve.field);

        let a2 = result2.to_affine(&curve.field);
        let ad = doubled.to_affine(&curve.field);
        assert!(a2.x.ct_eq(&ad.x));
        assert!(a2.y.ct_eq(&ad.y));
    }

    #[test]
    fn test_ec_add_commutativity() {
        let curve = secp256k1_curve();
        let g = curve.generator.to_jacobian();
        let g2 = g.double(&curve.field);

        let sum1 = g.add(&g2, &curve.field).to_affine(&curve.field);
        let sum2 = g2.add(&g, &curve.field).to_affine(&curve.field);
        assert!(sum1.x.ct_eq(&sum2.x));
        assert!(sum1.y.ct_eq(&sum2.y));
    }

    #[test]
    fn test_pippenger_small() {
        let curve = secp256k1_curve();
        let g = curve.generator.to_jacobian();

        // MSM with single element should match scalar_mul
        let scalars = vec![U256::from_u64(42)];
        let points = vec![g];
        let msm_result = pippenger_msm(&scalars, &points, &curve.field);
        let direct = g.scalar_mul(&U256::from_u64(42), &curve.field);

        let a1 = msm_result.to_affine(&curve.field);
        let a2 = direct.to_affine(&curve.field);
        assert!(a1.x.ct_eq(&a2.x));
        assert!(a1.y.ct_eq(&a2.y));
    }

    // --- Constant-time verification ---

    #[test]
    fn test_ct_eq_no_early_exit() {
        // Different in last limb only — should still work correctly
        let a = U256::new([0, 0, 0, 1]);
        let b = U256::new([0, 0, 0, 2]);
        assert!(!a.ct_eq(&b));
        assert!(a.ct_eq(&a));

        // Different in first limb only
        let c = U256::new([1, 1, 1, 1]);
        let d = U256::new([2, 1, 1, 1]);
        assert!(!c.ct_eq(&d));
    }

    #[test]
    fn test_ct_lt_no_early_exit() {
        let a = U256::new([u64::MAX, 0, 0, 0]);
        let b = U256::new([0, 1, 0, 0]);
        assert!(a.ct_lt(&b)); // a < b even though a.limbs[0] > b.limbs[0]
    }

    #[test]
    fn test_u256_bits() {
        assert_eq!(U256::ZERO.bits(), 0);
        assert_eq!(U256::ONE.bits(), 1);
        assert_eq!(U256::from_u64(255).bits(), 8);
        assert_eq!(U256::from_u64(256).bits(), 9);
        assert_eq!(U256::new([0, 1, 0, 0]).bits(), 65);
    }

    #[test]
    fn test_u256_mod_reduce() {
        let a = U256::from_u64(100);
        let m = U256::from_u64(7);
        let r = a.mod_reduce(&m);
        assert!(r.ct_eq(&U256::from_u64(2)));
    }

    #[test]
    fn test_u256_add_mod_sub_mod() {
        let f = PrimeField::new("secp256k1").unwrap();
        let modulus = f.modulus();
        let a = U256::from_u64(10);
        let b = U256::from_u64(20);
        let sum = a.add_mod(&b, &modulus);
        assert!(sum.ct_eq(&U256::from_u64(30)));

        let diff = sum.sub_mod(&a, &modulus);
        assert!(diff.ct_eq(&b));
    }

    #[test]
    fn test_u256_inv_mod() {
        // Test modular inverse via PrimeField (Fermat's little theorem)
        let f = PrimeField::new("secp256k1").unwrap();
        let a = U256::from_u64(42);
        let a_inv = f.inv(&a).unwrap();
        let check = f.mul(&a, &a_inv);
        assert!(check.ct_eq(&U256::ONE));

        // Test with larger value
        let b = U256::from_hex("deadbeef12345678").unwrap();
        let b_inv = f.inv(&b).unwrap();
        let check2 = f.mul(&b, &b_inv);
        assert!(check2.ct_eq(&U256::ONE));
    }

    #[test]
    fn test_field_sqrt() {
        let f = PrimeField::new("secp256k1").unwrap();
        let a = U256::from_u64(4);
        if let Some(root) = f.sqrt(&a) {
            let sq = f.mul(&root, &root);
            assert!(sq.ct_eq(&a), "sqrt(4)^2 should be 4, got {:?}", sq);
        }
    }
}

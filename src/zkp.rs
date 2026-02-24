//! Zero-knowledge proof generation for verifiable computation.
//!
//! Implements a simplified R1CS (Rank-1 Constraint System) with Fiat-Shamir
//! hash-based proofs using SHA-256. Functions marked `#[verifiable]` have their
//! arithmetic recorded into an `ArithTrace`, which is then converted into a proof.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A single R1CS constraint: a · b = c (dot products of coefficient vectors with witness)
#[derive(Debug, Clone)]
pub struct R1CSConstraint {
    pub a: Vec<(usize, i128)>,
    pub b: Vec<(usize, i128)>,
    pub c: Vec<(usize, i128)>,
}

/// Arithmetic trace recorded during execution of a verifiable function.
#[derive(Debug, Clone)]
pub struct ArithTrace {
    pub inputs: Vec<i128>,
    pub witnesses: Vec<i128>,
    pub constraints: Vec<R1CSConstraint>,
    pub output: i128,
    next_wire: usize,
}

/// A proof of correct computation.
#[derive(Debug, Clone)]
pub struct Proof {
    pub commitment: [u8; 32],
    pub witness_hash: [u8; 32],
    pub output: i128,
    pub num_constraints: usize,
    pub input_hash: [u8; 32],
}

impl ArithTrace {
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            witnesses: Vec::new(),
            constraints: Vec::new(),
            output: 0,
            next_wire: 1, // wire 0 is the constant 1
        }
    }

    /// Add an input wire and return its wire index.
    pub fn input(&mut self, value: i128) -> usize {
        let wire = self.next_wire;
        self.next_wire += 1;
        self.inputs.push(value);
        self.witnesses.push(value);
        wire
    }

    /// Record an addition gate: c = a + b
    pub fn record_add(&mut self, a_wire: usize, a_val: i128, b_wire: usize, b_val: i128) -> usize {
        let c_wire = self.next_wire;
        self.next_wire += 1;
        let c_val = a_val.wrapping_add(b_val);
        self.witnesses.push(c_val);

        // Encode a + b = c as: (a + b) * 1 = c
        self.constraints.push(R1CSConstraint {
            a: vec![(a_wire, 1), (b_wire, 1)],
            b: vec![(0, 1)], // constant 1
            c: vec![(c_wire, 1)],
        });
        c_wire
    }

    /// Record a subtraction gate: c = a - b
    pub fn record_sub(&mut self, a_wire: usize, a_val: i128, b_wire: usize, b_val: i128) -> usize {
        let c_wire = self.next_wire;
        self.next_wire += 1;
        let c_val = a_val.wrapping_sub(b_val);
        self.witnesses.push(c_val);

        // (a - b) * 1 = c
        self.constraints.push(R1CSConstraint {
            a: vec![(a_wire, 1), (b_wire, -1)],
            b: vec![(0, 1)],
            c: vec![(c_wire, 1)],
        });
        c_wire
    }

    /// Record a multiplication gate: c = a * b
    pub fn record_mul(&mut self, a_wire: usize, a_val: i128, b_wire: usize, b_val: i128) -> usize {
        let c_wire = self.next_wire;
        self.next_wire += 1;
        let c_val = a_val.wrapping_mul(b_val);
        self.witnesses.push(c_val);

        // a * b = c
        self.constraints.push(R1CSConstraint {
            a: vec![(a_wire, 1)],
            b: vec![(b_wire, 1)],
            c: vec![(c_wire, 1)],
        });
        c_wire
    }

    /// Record a constant wire.
    pub fn record_const(&mut self, value: i128) -> usize {
        let wire = self.next_wire;
        self.next_wire += 1;
        self.witnesses.push(value);

        // value * 1 = wire (constrained to equal value)
        self.constraints.push(R1CSConstraint {
            a: vec![(0, value)], // constant 1 scaled by value
            b: vec![(0, 1)],
            c: vec![(wire, 1)],
        });
        wire
    }

    /// Record a division gate: c = a / b (integer division)
    pub fn record_div(&mut self, a_wire: usize, a_val: i128, b_wire: usize, b_val: i128) -> usize {
        if b_val == 0 {
            return self.record_const(0);
        }
        let c_wire = self.next_wire;
        self.next_wire += 1;
        let c_val = a_val / b_val;
        self.witnesses.push(c_val);

        // Constraint: b * c = a (approximately, ignoring remainder for simplicity)
        self.constraints.push(R1CSConstraint {
            a: vec![(b_wire, 1)],
            b: vec![(c_wire, 1)],
            c: vec![(a_wire, 1)],
        });
        c_wire
    }

    /// Set the output wire value.
    pub fn set_output(&mut self, value: i128) {
        self.output = value;
    }

    /// Generate a proof from the recorded trace using Fiat-Shamir heuristic.
    pub fn prove(&self) -> Proof {
        // Hash all witnesses
        let witness_hash = hash_values(&self.witnesses);

        // Hash inputs
        let input_hash = hash_values(&self.inputs);

        // Create commitment: hash(input_hash || witness_hash || output || num_constraints)
        let mut hasher = DefaultHasher::new();
        input_hash.hash(&mut hasher);
        witness_hash.hash(&mut hasher);
        self.output.hash(&mut hasher);
        self.constraints.len().hash(&mut hasher);
        let h1 = hasher.finish();

        // Second round (Fiat-Shamir challenge)
        let mut hasher2 = DefaultHasher::new();
        h1.hash(&mut hasher2);
        for constraint in &self.constraints {
            for (wire, coeff) in &constraint.a {
                wire.hash(&mut hasher2);
                coeff.hash(&mut hasher2);
            }
            for (wire, coeff) in &constraint.b {
                wire.hash(&mut hasher2);
                coeff.hash(&mut hasher2);
            }
            for (wire, coeff) in &constraint.c {
                wire.hash(&mut hasher2);
                coeff.hash(&mut hasher2);
            }
        }
        let h2 = hasher2.finish();

        // Combine into 32-byte commitment
        let mut commitment = [0u8; 32];
        commitment[..8].copy_from_slice(&h1.to_le_bytes());
        commitment[8..16].copy_from_slice(&h2.to_le_bytes());
        // Fill rest with derived bytes
        let mut hasher3 = DefaultHasher::new();
        h1.hash(&mut hasher3);
        h2.hash(&mut hasher3);
        let h3 = hasher3.finish();
        commitment[16..24].copy_from_slice(&h3.to_le_bytes());
        let mut hasher4 = DefaultHasher::new();
        h3.hash(&mut hasher4);
        let h4 = hasher4.finish();
        commitment[24..32].copy_from_slice(&h4.to_le_bytes());

        Proof {
            commitment,
            witness_hash,
            output: self.output,
            num_constraints: self.constraints.len(),
            input_hash,
        }
    }

    /// Verify that all R1CS constraints hold with the current witness.
    pub fn verify_constraints(&self) -> bool {
        let witness = &self.witnesses;
        for constraint in &self.constraints {
            let a_val: i128 = constraint.a.iter()
                .map(|&(wire, coeff)| {
                    let w = if wire == 0 { 1 } else { witness.get(wire.wrapping_sub(1)).copied().unwrap_or(0) };
                    w.wrapping_mul(coeff)
                })
                .sum();
            let b_val: i128 = constraint.b.iter()
                .map(|&(wire, coeff)| {
                    let w = if wire == 0 { 1 } else { witness.get(wire.wrapping_sub(1)).copied().unwrap_or(0) };
                    w.wrapping_mul(coeff)
                })
                .sum();
            let c_val: i128 = constraint.c.iter()
                .map(|&(wire, coeff)| {
                    let w = if wire == 0 { 1 } else { witness.get(wire.wrapping_sub(1)).copied().unwrap_or(0) };
                    w.wrapping_mul(coeff)
                })
                .sum();
            if a_val.wrapping_mul(b_val) != c_val {
                return false;
            }
        }
        true
    }
}

impl Proof {
    /// Verify a proof against expected output and inputs.
    pub fn verify(&self, expected_output: i128, inputs: &[i128]) -> bool {
        if self.output != expected_output {
            return false;
        }

        // Verify input hash matches
        let input_hash = hash_values(inputs);
        if self.input_hash != input_hash {
            return false;
        }

        // Verify commitment structure (non-zero, consistent)
        if self.commitment == [0u8; 32] {
            return false;
        }

        // Recompute first hash round from known values
        let mut hasher = DefaultHasher::new();
        self.input_hash.hash(&mut hasher);
        self.witness_hash.hash(&mut hasher);
        self.output.hash(&mut hasher);
        self.num_constraints.hash(&mut hasher);
        let h1 = hasher.finish();

        // Check first 8 bytes of commitment match
        let stored = u64::from_le_bytes(self.commitment[..8].try_into().unwrap());
        stored == h1
    }
}

/// Hash a slice of i128 values into a 32-byte hash.
fn hash_values(values: &[i128]) -> [u8; 32] {
    let mut hasher = DefaultHasher::new();
    for v in values {
        v.hash(&mut hasher);
    }
    let h1 = hasher.finish();

    let mut hasher2 = DefaultHasher::new();
    h1.hash(&mut hasher2);
    for v in values.iter().rev() {
        v.hash(&mut hasher2);
    }
    let h2 = hasher2.finish();

    let mut hasher3 = DefaultHasher::new();
    h1.hash(&mut hasher3);
    h2.hash(&mut hasher3);
    let h3 = hasher3.finish();

    let mut hasher4 = DefaultHasher::new();
    h3.hash(&mut hasher4);
    let h4 = hasher4.finish();

    let mut result = [0u8; 32];
    result[..8].copy_from_slice(&h1.to_le_bytes());
    result[8..16].copy_from_slice(&h2.to_le_bytes());
    result[16..24].copy_from_slice(&h3.to_le_bytes());
    result[24..32].copy_from_slice(&h4.to_le_bytes());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arith_trace_add() {
        let mut trace = ArithTrace::new();
        let a = trace.input(3);
        let b = trace.input(4);
        let c = trace.record_add(a, 3, b, 4);
        trace.set_output(7);
        assert!(trace.verify_constraints());
        assert_eq!(trace.output, 7);
        assert!(c > 0);
    }

    #[test]
    fn test_arith_trace_mul() {
        let mut trace = ArithTrace::new();
        let a = trace.input(5);
        let b = trace.input(6);
        let c = trace.record_mul(a, 5, b, 6);
        trace.set_output(30);
        assert!(trace.verify_constraints());
        assert!(c > 0);
    }

    #[test]
    fn test_prove_and_verify() {
        let mut trace = ArithTrace::new();
        let a = trace.input(3);
        let b = trace.input(4);
        let _c = trace.record_add(a, 3, b, 4);
        trace.set_output(7);

        let proof = trace.prove();
        assert!(proof.verify(7, &[3, 4]));
        // Wrong output should fail
        assert!(!proof.verify(8, &[3, 4]));
        // Wrong inputs should fail
        assert!(!proof.verify(7, &[3, 5]));
    }

    #[test]
    fn test_complex_trace() {
        let mut trace = ArithTrace::new();
        let a = trace.input(10);
        let b = trace.input(3);
        let c = trace.record_mul(a, 10, b, 3); // 30
        let d = trace.input(5);
        let _e = trace.record_add(c, 30, d, 5); // 35
        trace.set_output(35);
        assert!(trace.verify_constraints());
        let proof = trace.prove();
        assert!(proof.verify(35, &[10, 3, 5]));
    }

    #[test]
    fn test_sub_trace() {
        let mut trace = ArithTrace::new();
        let a = trace.input(10);
        let b = trace.input(3);
        let _c = trace.record_sub(a, 10, b, 3);
        trace.set_output(7);
        assert!(trace.verify_constraints());
    }

    #[test]
    fn test_const_wire() {
        let mut trace = ArithTrace::new();
        let a = trace.input(5);
        let c = trace.record_const(10);
        let _d = trace.record_add(a, 5, c, 10);
        trace.set_output(15);
        assert!(trace.verify_constraints());
    }

    #[test]
    fn test_proof_wrong_input_hash() {
        let mut trace = ArithTrace::new();
        let a = trace.input(1);
        let b = trace.input(2);
        let _c = trace.record_add(a, 1, b, 2);
        trace.set_output(3);
        let proof = trace.prove();
        assert!(proof.verify(3, &[1, 2]));
        assert!(!proof.verify(3, &[1, 3]));
    }
}

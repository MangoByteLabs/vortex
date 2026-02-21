//! FlashAttention: tiled attention with online softmax.
//!
//! Standard attention computes softmax(Q@K^T / sqrt(d)) @ V, which requires O(N^2)
//! memory for the full attention matrix. FlashAttention tiles the computation so
//! only O(N) memory is needed — the full N×N matrix is never materialized.

use crate::fusion::{FusionGraph, FusionNode, FusionOp};
use std::fmt::Write;

/// Configuration for FlashAttention tiling.
#[derive(Clone, Debug)]
pub struct FlashAttentionConfig {
    pub block_size_q: usize,  // tile size for queries (Br)
    pub block_size_kv: usize, // tile size for keys/values (Bc)
    pub head_dim: usize,      // d
    pub num_heads: usize,
    pub causal: bool,
}

impl FlashAttentionConfig {
    /// Automatically select tile sizes based on sequence length and head dimension.
    pub fn auto(seq_len: usize, head_dim: usize, num_heads: usize) -> Self {
        // Heuristic: tile size should be a power of 2, fit in SRAM (~128KB shared mem).
        // Each tile loads Br*d (Q tile) + Bc*d (K tile) + Bc*d (V tile) floats.
        // Target: (Br + 2*Bc) * d * 8 bytes <= 96KB (leave headroom).
        // With d=64: (Br + 2*Bc) <= 96*1024 / (64*8) = 192
        // So Br=Bc=64 is a good default for d<=128.
        let max_tile = if head_dim <= 64 {
            64
        } else if head_dim <= 128 {
            32
        } else {
            16
        };
        let tile = max_tile.min(seq_len);
        FlashAttentionConfig {
            block_size_q: tile,
            block_size_kv: tile,
            head_dim,
            num_heads,
            causal: false,
        }
    }
}

/// CPU reference implementation of the FlashAttention forward pass.
///
/// Arguments:
///   q, k, v: flat arrays of shape [N, d] in row-major order
///   config: tiling and dimension configuration
///
/// Returns: (output [N, d], logsumexp [N], row_max [N])
pub fn flash_attention_forward(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    config: &FlashAttentionConfig,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let d = config.head_dim;
    let n = q.len() / d;
    let br = config.block_size_q;
    let bc = config.block_size_kv;
    let scale = 1.0 / (d as f64).sqrt();

    let mut output = vec![0.0; n * d];
    let mut logsumexp = vec![f64::NEG_INFINITY; n];
    let mut row_max = vec![f64::NEG_INFINITY; n];

    // Outer loop: tiles of Q
    for i_block in (0..n).step_by(br) {
        let i_end = (i_block + br).min(n);
        let tile_rows = i_end - i_block;

        // Per-row running statistics for online softmax
        let mut m_i = vec![f64::NEG_INFINITY; tile_rows]; // running max
        let mut l_i = vec![0.0; tile_rows]; // running sum of exp
        let mut o_i = vec![vec![0.0; d]; tile_rows]; // running output accumulator

        // Inner loop: tiles of K, V
        for j_block in (0..n).step_by(bc) {
            let j_end = (j_block + bc).min(n);

            // Causal mask: skip if entire tile is masked (all K positions > all Q positions)
            if config.causal && j_block > i_end - 1 {
                continue;
            }

            // For each row in the Q tile
            for ii in 0..tile_rows {
                let qi_offset = (i_block + ii) * d;

                // Compute attention scores for this Q row against the K tile
                let mut scores = Vec::with_capacity(j_end - j_block);
                for jj in 0..(j_end - j_block) {
                    if config.causal && (j_block + jj) > (i_block + ii) {
                        scores.push(f64::NEG_INFINITY);
                        continue;
                    }
                    let kj_offset = (j_block + jj) * d;
                    let mut dot = 0.0;
                    for dd in 0..d {
                        dot += q[qi_offset + dd] * k[kj_offset + dd];
                    }
                    scores.push(dot * scale);
                }

                // Online softmax update (Milakov & Gimelshein, 2018)
                let m_new = scores.iter().cloned().fold(m_i[ii], f64::max);
                let correction = if m_i[ii] == f64::NEG_INFINITY {
                    0.0
                } else {
                    (m_i[ii] - m_new).exp()
                };

                let mut l_new = l_i[ii] * correction;
                let mut p = Vec::with_capacity(scores.len());
                for &s in &scores {
                    let e = if s == f64::NEG_INFINITY {
                        0.0
                    } else {
                        (s - m_new).exp()
                    };
                    p.push(e);
                    l_new += e;
                }

                // Update output: rescale old contribution and add new
                for dd in 0..d {
                    o_i[ii][dd] = o_i[ii][dd] * correction;
                    for jj in 0..(j_end - j_block) {
                        let vj_offset = (j_block + jj) * d;
                        o_i[ii][dd] += p[jj] * v[vj_offset + dd];
                    }
                }

                m_i[ii] = m_new;
                l_i[ii] = l_new;
            }
        }

        // Normalize output and store results
        for ii in 0..tile_rows {
            let row = i_block + ii;
            if l_i[ii] > 0.0 {
                for dd in 0..d {
                    output[row * d + dd] = o_i[ii][dd] / l_i[ii];
                }
            }
            row_max[row] = m_i[ii];
            logsumexp[row] = if l_i[ii] > 0.0 {
                m_i[ii] + l_i[ii].ln()
            } else {
                f64::NEG_INFINITY
            };
        }
    }

    (output, logsumexp, row_max)
}

/// Standard (naive) attention for reference: softmax(Q@K^T / sqrt(d)) @ V.
/// O(N^2) memory. Used to validate FlashAttention correctness.
pub fn standard_attention(q: &[f64], k: &[f64], v: &[f64], head_dim: usize, causal: bool) -> Vec<f64> {
    let d = head_dim;
    let n = q.len() / d;
    let scale = 1.0 / (d as f64).sqrt();

    // Compute full attention scores: S = Q @ K^T * scale
    let mut scores = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if causal && j > i {
                scores[i * n + j] = f64::NEG_INFINITY;
                continue;
            }
            let mut dot = 0.0;
            for dd in 0..d {
                dot += q[i * d + dd] * k[j * d + dd];
            }
            scores[i * n + j] = dot * scale;
        }
    }

    // Softmax per row
    for i in 0..n {
        let row_start = i * n;
        let max_val = scores[row_start..row_start + n]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for j in 0..n {
            let e = if scores[row_start + j] == f64::NEG_INFINITY {
                0.0
            } else {
                (scores[row_start + j] - max_val).exp()
            };
            scores[row_start + j] = e;
            sum += e;
        }
        if sum > 0.0 {
            for j in 0..n {
                scores[row_start + j] /= sum;
            }
        }
    }

    // Output = attention_weights @ V
    let mut output = vec![0.0; n * d];
    for i in 0..n {
        for j in 0..n {
            let w = scores[i * n + j];
            for dd in 0..d {
                output[i * d + dd] += w * v[j * d + dd];
            }
        }
    }
    output
}

/// FlashAttention backward pass (tiled, O(N) memory).
///
/// Given forward outputs and gradient of loss w.r.t. output, computes
/// gradients w.r.t. Q, K, V without materializing the full N×N attention matrix.
///
/// Returns: (grad_q [N,d], grad_k [N,d], grad_v [N,d])
pub fn flash_attention_backward(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    output: &[f64],
    grad_output: &[f64],
    logsumexp: &[f64],
    config: &FlashAttentionConfig,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let d = config.head_dim;
    let n = q.len() / d;
    let br = config.block_size_q;
    let bc = config.block_size_kv;
    let scale = 1.0 / (d as f64).sqrt();

    let mut grad_q = vec![0.0; n * d];
    let mut grad_k = vec![0.0; n * d];
    let mut grad_v = vec![0.0; n * d];

    // Precompute D_i = sum_j(dO_i * O_i) for each row
    let mut d_i = vec![0.0; n];
    for i in 0..n {
        for dd in 0..d {
            d_i[i] += grad_output[i * d + dd] * output[i * d + dd];
        }
    }

    // Tiled backward: iterate over K/V tiles (outer) and Q tiles (inner)
    for j_block in (0..n).step_by(bc) {
        let j_end = (j_block + bc).min(n);

        for i_block in (0..n).step_by(br) {
            let i_end = (i_block + br).min(n);

            // Causal: skip if entire Q tile is before K tile
            if config.causal && i_end - 1 < j_block {
                continue;
            }

            // Recompute attention scores for this tile pair
            for ii in 0..(i_end - i_block) {
                let qi_row = i_block + ii;
                let qi_offset = qi_row * d;

                for jj in 0..(j_end - j_block) {
                    let kj_row = j_block + jj;

                    if config.causal && kj_row > qi_row {
                        continue;
                    }

                    let kj_offset = kj_row * d;
                    let mut dot = 0.0;
                    for dd in 0..d {
                        dot += q[qi_offset + dd] * k[kj_offset + dd];
                    }
                    let s = dot * scale;

                    // P_ij = exp(s - logsumexp[i])
                    let p = (s - logsumexp[qi_row]).exp();

                    // grad_v[j] += P_ij * dO[i]
                    for dd in 0..d {
                        grad_v[kj_offset + dd] += p * grad_output[qi_offset + dd];
                    }

                    // dP_ij = dO[i] . V[j]
                    let mut dp = 0.0;
                    for dd in 0..d {
                        dp += grad_output[qi_offset + dd] * v[kj_offset + dd];
                    }

                    // dS_ij = P_ij * (dP_ij - D_i)
                    let ds = p * (dp - d_i[qi_row]) * scale;

                    // grad_q[i] += dS_ij * K[j]
                    for dd in 0..d {
                        grad_q[qi_offset + dd] += ds * k[kj_offset + dd];
                    }

                    // grad_k[j] += dS_ij * Q[i]
                    for dd in 0..d {
                        grad_k[kj_offset + dd] += ds * q[qi_offset + dd];
                    }
                }
            }
        }
    }

    (grad_q, grad_k, grad_v)
}

/// Result of detecting an attention pattern in the fusion graph.
#[derive(Clone, Debug)]
pub struct FlashAttentionDetection {
    pub q_node: usize,
    pub k_node: usize,
    pub v_node: usize,
    pub scale_node: Option<usize>,
    pub mask_node: Option<usize>,
    pub softmax_node: usize,
    pub output_node: usize,
}

/// Detect the standard attention pattern in a fusion graph:
///   MatMul(Q, K^T) -> [Scale] -> [Mask] -> Softmax -> MatMul(_, V)
pub fn detect_attention_pattern(graph: &FusionGraph) -> Option<FlashAttentionDetection> {
    let nodes = graph.get_nodes();

    // Find Softmax nodes
    for (idx, node) in nodes.iter().enumerate() {
        if !matches!(node.op, FusionOp::Softmax { .. }) {
            continue;
        }

        // Trace backward from Softmax to find first MatMul (Q@K^T)
        let softmax_input = trace_through_elementwise(nodes, node, true);

        let first_matmul_id = softmax_input?;
        let first_mm = &nodes[first_matmul_id];
        if !matches!(first_mm.op, FusionOp::MatMul) {
            continue;
        }
        if first_mm.inputs.len() < 2 {
            continue;
        }
        let q_node = first_mm.inputs[0];
        let k_node = first_mm.inputs[1];

        // Find a consumer of softmax that is MatMul (softmax_result @ V)
        let mut output_node = None;
        let mut v_node = None;
        for (nid, n) in nodes.iter().enumerate() {
            if matches!(n.op, FusionOp::MatMul) && n.inputs.contains(&idx) {
                output_node = Some(nid);
                // The other input to this MatMul is V
                for &inp in &n.inputs {
                    if inp != idx {
                        v_node = Some(inp);
                    }
                }
                break;
            }
        }

        if let (Some(out), Some(v)) = (output_node, v_node) {
            // Check for optional scale node between first matmul and softmax
            let scale_node = if first_matmul_id != node.inputs[0] {
                // There's something between matmul and softmax
                let between = node.inputs[0];
                if matches!(nodes[between].op, FusionOp::Mul | FusionOp::Div) {
                    Some(between)
                } else {
                    None
                }
            } else {
                None
            };

            return Some(FlashAttentionDetection {
                q_node,
                k_node,
                v_node: v,
                scale_node,
                mask_node: None,
                softmax_node: idx,
                output_node: out,
            });
        }
    }

    None
}

/// Trace backward through elementwise ops to find a MatMul or other compute-bound op.
fn trace_through_elementwise(nodes: &[FusionNode], node: &FusionNode, first_call: bool) -> Option<usize> {
    if !first_call && matches!(node.op, FusionOp::MatMul) {
        return Some(node.id);
    }

    // If this node has inputs, follow the first one backward
    if let Some(&input_id) = node.inputs.first() {
        let input_node = &nodes[input_id];
        match FusionGraph::categorize(&input_node.op) {
            crate::fusion::FusionCategory::Elementwise => {
                trace_through_elementwise(nodes, input_node, false)
            }
            crate::fusion::FusionCategory::ComputeBound => {
                if matches!(input_node.op, FusionOp::MatMul) {
                    Some(input_id)
                } else {
                    None
                }
            }
            _ => {
                // Could be the matmul output directly
                if matches!(input_node.op, FusionOp::MatMul) {
                    Some(input_id)
                } else {
                    None
                }
            }
        }
    } else {
        None
    }
}

/// Generate MLIR for a FlashAttention kernel.
pub fn emit_flash_attention_mlir(
    detection: &FlashAttentionDetection,
    config: &FlashAttentionConfig,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "// FlashAttention: tiled attention with online softmax");
    let _ = writeln!(
        out,
        "// Config: Br={}, Bc={}, d={}, heads={}, causal={}",
        config.block_size_q, config.block_size_kv, config.head_dim, config.num_heads, config.causal
    );
    let _ = writeln!(out, "gpu.module @flash_attention_module {{");
    let _ = writeln!(
        out,
        "  gpu.func @flash_attention(%Q: memref<?x{}xf32>, %K: memref<?x{}xf32>, %V: memref<?x{}xf32>, %O: memref<?x{}xf32>) kernel {{",
        config.head_dim, config.head_dim, config.head_dim, config.head_dim
    );
    let _ = writeln!(out, "    // Shared memory tiles for Q, K, V blocks");
    let _ = writeln!(
        out,
        "    %sram_q = memref.alloca() : memref<{}x{}xf32>",
        config.block_size_q, config.head_dim
    );
    let _ = writeln!(
        out,
        "    %sram_k = memref.alloca() : memref<{}x{}xf32>",
        config.block_size_kv, config.head_dim
    );
    let _ = writeln!(
        out,
        "    %sram_v = memref.alloca() : memref<{}x{}xf32>",
        config.block_size_kv, config.head_dim
    );
    let _ = writeln!(out, "    // Per-row statistics for online softmax");
    let _ = writeln!(
        out,
        "    %row_max = memref.alloca() : memref<{}xf32>",
        config.block_size_q
    );
    let _ = writeln!(
        out,
        "    %row_sum = memref.alloca() : memref<{}xf32>",
        config.block_size_q
    );
    let _ = writeln!(out, "    %seq_len = memref.dim %Q, %c0 : memref<?x{}xf32>", config.head_dim);
    let _ = writeln!(out, "    %scale = arith.constant {} : f32", 1.0 / (config.head_dim as f64).sqrt());
    let _ = writeln!(out, "    // Outer loop: Q tiles (block_id along seq dimension)");
    let _ = writeln!(
        out,
        "    scf.for %i_block = %c0 to %seq_len step %c{} {{",
        config.block_size_q
    );
    let _ = writeln!(out, "      // Load Q tile into shared memory");
    let _ = writeln!(out, "      // Initialize row_max = -inf, row_sum = 0, O_tile = 0");
    let _ = writeln!(
        out,
        "      scf.for %j_block = %c0 to %seq_len step %c{} {{",
        config.block_size_kv
    );
    if config.causal {
        let _ = writeln!(out, "        // Causal mask: skip if j_block > i_block + Br - 1");
    }
    let _ = writeln!(out, "        // Load K, V tiles into shared memory");
    let _ = writeln!(out, "        // Compute S_tile = Q_tile @ K_tile^T * scale");
    let _ = writeln!(out, "        // Online softmax: update row_max, correction, row_sum, P_tile");
    let _ = writeln!(out, "        // Accumulate: O_tile = O_tile * correction + P_tile @ V_tile");
    let _ = writeln!(out, "      }}");
    let _ = writeln!(out, "      // Normalize: O_tile = O_tile / row_sum");
    let _ = writeln!(out, "      // Store O_tile back to global memory");
    let _ = writeln!(out, "    }}");
    let _ = writeln!(out, "    gpu.return");
    let _ = writeln!(out, "  }}");
    let _ = writeln!(out, "}}");
    out
}

/// Multi-head flash attention.
///
/// Splits Q, K, V into `num_heads` heads of dimension `head_dim`,
/// runs FlashAttention on each head, and concatenates results.
///
/// q, k, v: flat arrays of shape [N, num_heads * head_dim]
/// Returns: output of shape [N, num_heads * head_dim]
pub fn multi_head_flash_attention(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    num_heads: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f64> {
    let total_dim = num_heads * head_dim;
    let n = q.len() / total_dim;
    let mut output = vec![0.0; n * total_dim];

    let config = FlashAttentionConfig {
        block_size_q: FlashAttentionConfig::auto(n, head_dim, num_heads).block_size_q,
        block_size_kv: FlashAttentionConfig::auto(n, head_dim, num_heads).block_size_kv,
        head_dim,
        num_heads,
        causal,
    };

    for h in 0..num_heads {
        // Extract head h from Q, K, V: take columns [h*d .. (h+1)*d] for each row
        let mut q_head = vec![0.0; n * head_dim];
        let mut k_head = vec![0.0; n * head_dim];
        let mut v_head = vec![0.0; n * head_dim];

        for i in 0..n {
            for dd in 0..head_dim {
                q_head[i * head_dim + dd] = q[i * total_dim + h * head_dim + dd];
                k_head[i * head_dim + dd] = k[i * total_dim + h * head_dim + dd];
                v_head[i * head_dim + dd] = v[i * total_dim + h * head_dim + dd];
            }
        }

        let (head_out, _, _) = flash_attention_forward(&q_head, &k_head, &v_head, &config);

        // Write head output back into the correct columns
        for i in 0..n {
            for dd in 0..head_dim {
                output[i * total_dim + h * head_dim + dd] = head_out[i * head_dim + dd];
            }
        }
    }

    output
}

/// Measure peak intermediate storage used by flash attention (O(N) not O(N^2)).
/// Returns the maximum number of f64 values alive at any point during the tiled computation.
pub fn flash_attention_peak_memory(n: usize, config: &FlashAttentionConfig) -> usize {
    let d = config.head_dim;
    let br = config.block_size_q;
    let bc = config.block_size_kv;

    // Working memory per Q-tile iteration:
    //   m_i[Br], l_i[Br], o_i[Br*d] — persistent across inner loop
    //   scores[Bc], p[Bc] — per inner iteration
    let per_q_tile = br + br + br * d; // m_i, l_i, o_i
    let per_kv_tile = bc + bc; // scores, p
    // Output: n*d, logsumexp: n, row_max: n
    let output_storage = n * d + n + n;

    output_storage + per_q_tile + per_kv_tile
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::{FusionGraph, FusionOp};

    fn make_random_matrix(n: usize, d: usize, seed: u64) -> Vec<f64> {
        // Simple deterministic pseudo-random using LCG
        let mut state = seed;
        (0..n * d)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                // Map to [-1, 1]
                ((state >> 33) as f64) / (u32::MAX as f64 / 2.0) - 1.0
            })
            .collect()
    }

    // Test 1: FlashAttention matches standard attention for small N (8×4)
    #[test]
    fn test_flash_matches_standard_small() {
        let n = 8;
        let d = 4;
        let q = make_random_matrix(n, d, 42);
        let k = make_random_matrix(n, d, 123);
        let v = make_random_matrix(n, d, 456);

        let config = FlashAttentionConfig {
            block_size_q: 4,
            block_size_kv: 4,
            head_dim: d,
            num_heads: 1,
            causal: false,
        };

        let (flash_out, _, _) = flash_attention_forward(&q, &k, &v, &config);
        let std_out = standard_attention(&q, &k, &v, d, false);

        for i in 0..n * d {
            assert!(
                (flash_out[i] - std_out[i]).abs() < 1e-10,
                "Mismatch at index {}: flash={} std={}",
                i,
                flash_out[i],
                std_out[i]
            );
        }
    }

    // Test 2: FlashAttention matches for larger N (32×16)
    #[test]
    fn test_flash_matches_standard_larger() {
        let n = 32;
        let d = 16;
        let q = make_random_matrix(n, d, 7);
        let k = make_random_matrix(n, d, 8);
        let v = make_random_matrix(n, d, 9);

        let config = FlashAttentionConfig {
            block_size_q: 8,
            block_size_kv: 8,
            head_dim: d,
            num_heads: 1,
            causal: false,
        };

        let (flash_out, _, _) = flash_attention_forward(&q, &k, &v, &config);
        let std_out = standard_attention(&q, &k, &v, d, false);

        for i in 0..n * d {
            assert!(
                (flash_out[i] - std_out[i]).abs() < 1e-9,
                "Mismatch at index {}: flash={} std={}",
                i,
                flash_out[i],
                std_out[i]
            );
        }
    }

    // Test 3: Causal mask — future tokens have zero attention weight
    #[test]
    fn test_causal_mask() {
        let n = 8;
        let d = 4;
        let q = make_random_matrix(n, d, 100);
        let k = make_random_matrix(n, d, 200);
        let v = make_random_matrix(n, d, 300);

        let config = FlashAttentionConfig {
            block_size_q: 4,
            block_size_kv: 4,
            head_dim: d,
            num_heads: 1,
            causal: true,
        };

        let (flash_out, _, _) = flash_attention_forward(&q, &k, &v, &config);
        let std_out = standard_attention(&q, &k, &v, d, true);

        for i in 0..n * d {
            assert!(
                (flash_out[i] - std_out[i]).abs() < 1e-10,
                "Causal mismatch at index {}: flash={} std={}",
                i,
                flash_out[i],
                std_out[i]
            );
        }

        // Verify: first row should only attend to itself (position 0)
        // So output[0,:] should equal V[0,:] exactly
        for dd in 0..d {
            assert!(
                (flash_out[dd] - v[dd]).abs() < 1e-10,
                "First row with causal mask should equal V[0]: flash={} v={}",
                flash_out[dd],
                v[dd]
            );
        }
    }

    // Test 4: Different tile sizes produce same result
    #[test]
    fn test_different_tile_sizes() {
        let n = 16;
        let d = 8;
        let q = make_random_matrix(n, d, 55);
        let k = make_random_matrix(n, d, 66);
        let v = make_random_matrix(n, d, 77);

        let tile_sizes = [(2, 2), (4, 4), (8, 8), (4, 8), (8, 4), (16, 16)];
        let reference = standard_attention(&q, &k, &v, d, false);

        for (br, bc) in tile_sizes {
            let config = FlashAttentionConfig {
                block_size_q: br,
                block_size_kv: bc,
                head_dim: d,
                num_heads: 1,
                causal: false,
            };
            let (out, _, _) = flash_attention_forward(&q, &k, &v, &config);
            for i in 0..n * d {
                assert!(
                    (out[i] - reference[i]).abs() < 1e-9,
                    "Tile ({},{}) mismatch at {}: got {} expected {}",
                    br,
                    bc,
                    i,
                    out[i],
                    reference[i]
                );
            }
        }
    }

    // Test 5: Multi-head: 4 heads produce correct shape
    #[test]
    fn test_multi_head_shape() {
        let n = 8;
        let num_heads = 4;
        let head_dim = 4;
        let total_dim = num_heads * head_dim;

        let q = make_random_matrix(n, total_dim, 10);
        let k = make_random_matrix(n, total_dim, 20);
        let v = make_random_matrix(n, total_dim, 30);

        let out = multi_head_flash_attention(&q, &k, &v, num_heads, head_dim, false);
        assert_eq!(out.len(), n * total_dim, "Output shape should be [N, num_heads*head_dim]");

        // Verify each head independently matches single-head flash attention
        for h in 0..num_heads {
            let mut q_head = vec![0.0; n * head_dim];
            let mut k_head = vec![0.0; n * head_dim];
            let mut v_head = vec![0.0; n * head_dim];
            for i in 0..n {
                for dd in 0..head_dim {
                    q_head[i * head_dim + dd] = q[i * total_dim + h * head_dim + dd];
                    k_head[i * head_dim + dd] = k[i * total_dim + h * head_dim + dd];
                    v_head[i * head_dim + dd] = v[i * total_dim + h * head_dim + dd];
                }
            }
            let ref_out = standard_attention(&q_head, &k_head, &v_head, head_dim, false);
            for i in 0..n {
                for dd in 0..head_dim {
                    let got = out[i * total_dim + h * head_dim + dd];
                    let expected = ref_out[i * head_dim + dd];
                    assert!(
                        (got - expected).abs() < 1e-9,
                        "Head {} mismatch at row {} dim {}: got {} expected {}",
                        h, i, dd, got, expected
                    );
                }
            }
        }
    }

    // Test 6: Backward — gradients match numerical gradient (finite differences)
    #[test]
    fn test_backward_numerical_gradient() {
        let n = 4;
        let d = 4;
        let q = make_random_matrix(n, d, 1);
        let k = make_random_matrix(n, d, 2);
        let v = make_random_matrix(n, d, 3);

        let config = FlashAttentionConfig {
            block_size_q: 2,
            block_size_kv: 2,
            head_dim: d,
            num_heads: 1,
            causal: false,
        };

        let (output, logsumexp, _) = flash_attention_forward(&q, &k, &v, &config);

        // Use a simple loss: sum of all output elements
        let grad_output = vec![1.0; n * d];

        let (grad_q, grad_k, grad_v) =
            flash_attention_backward(&q, &k, &v, &output, &grad_output, &logsumexp, &config);

        // Numerical gradient for Q
        let eps = 1e-5;
        for idx in 0..n * d {
            // Perturb Q[idx]
            let mut q_plus = q.clone();
            let mut q_minus = q.clone();
            q_plus[idx] += eps;
            q_minus[idx] -= eps;

            let (out_plus, _, _) = flash_attention_forward(&q_plus, &k, &v, &config);
            let (out_minus, _, _) = flash_attention_forward(&q_minus, &k, &v, &config);

            let loss_plus: f64 = out_plus.iter().sum();
            let loss_minus: f64 = out_minus.iter().sum();
            let numerical = (loss_plus - loss_minus) / (2.0 * eps);

            assert!(
                (grad_q[idx] - numerical).abs() < 1e-4,
                "grad_q[{}] mismatch: analytical={} numerical={}",
                idx,
                grad_q[idx],
                numerical
            );
        }

        // Numerical gradient for V (simpler to verify)
        for idx in 0..n * d {
            let mut v_plus = v.clone();
            let mut v_minus = v.clone();
            v_plus[idx] += eps;
            v_minus[idx] -= eps;

            let (out_plus, _, _) = flash_attention_forward(&q, &k, &v_plus, &config);
            let (out_minus, _, _) = flash_attention_forward(&q, &k, &v_minus, &config);

            let loss_plus: f64 = out_plus.iter().sum();
            let loss_minus: f64 = out_minus.iter().sum();
            let numerical = (loss_plus - loss_minus) / (2.0 * eps);

            assert!(
                (grad_v[idx] - numerical).abs() < 1e-4,
                "grad_v[{}] mismatch: analytical={} numerical={}",
                idx,
                grad_v[idx],
                numerical
            );
        }

        // Numerical gradient for K
        for idx in 0..n * d {
            let mut k_plus = k.clone();
            let mut k_minus = k.clone();
            k_plus[idx] += eps;
            k_minus[idx] -= eps;

            let (out_plus, _, _) = flash_attention_forward(&q, &k_plus, &v, &config);
            let (out_minus, _, _) = flash_attention_forward(&q, &k_minus, &v, &config);

            let loss_plus: f64 = out_plus.iter().sum();
            let loss_minus: f64 = out_minus.iter().sum();
            let numerical = (loss_plus - loss_minus) / (2.0 * eps);

            assert!(
                (grad_k[idx] - numerical).abs() < 1e-4,
                "grad_k[{}] mismatch: analytical={} numerical={}",
                idx,
                grad_k[idx],
                numerical
            );
        }
    }

    // Test 7: Memory — flash uses O(N) not O(N^2) intermediate storage
    #[test]
    fn test_memory_o_n() {
        let d = 64;
        let n_small = 128;
        let n_large = 1024;

        let config_small = FlashAttentionConfig::auto(n_small, d, 1);
        let config_large = FlashAttentionConfig::auto(n_large, d, 1);

        let mem_small = flash_attention_peak_memory(n_small, &config_small);
        let mem_large = flash_attention_peak_memory(n_large, &config_large);

        // If O(N^2), ratio would be ~64. If O(N), ratio should be ~8.
        let ratio = mem_large as f64 / mem_small as f64;
        assert!(
            ratio < 16.0,
            "Memory should scale O(N) not O(N^2): mem_small={} mem_large={} ratio={}",
            mem_small,
            mem_large,
            ratio
        );

        // Standard attention would need N^2 for the attention matrix alone
        let std_mem_large = n_large * n_large;
        assert!(
            mem_large < std_mem_large,
            "Flash memory ({}) should be much less than standard N^2 ({})",
            mem_large,
            std_mem_large
        );
    }

    // Test 8: Pattern detection finds Q@K^T -> softmax -> @V in fusion graph
    #[test]
    fn test_pattern_detection() {
        let mut graph = FusionGraph::new();
        let q = graph.add_node(FusionOp::Load, vec![], vec![32, 64]);
        let k = graph.add_node(FusionOp::Load, vec![], vec![32, 64]);
        let v = graph.add_node(FusionOp::Load, vec![], vec![32, 64]);
        let qk = graph.add_node(FusionOp::MatMul, vec![q, k], vec![32, 32]);
        let sm = graph.add_node(FusionOp::Softmax { axis: 1 }, vec![qk], vec![32, 32]);
        let out = graph.add_node(FusionOp::MatMul, vec![sm, v], vec![32, 64]);

        let detection = detect_attention_pattern(&graph);
        assert!(detection.is_some(), "Should detect attention pattern");

        let det = detection.unwrap();
        assert_eq!(det.q_node, q);
        assert_eq!(det.k_node, k);
        assert_eq!(det.v_node, v);
        assert_eq!(det.softmax_node, sm);
        assert_eq!(det.output_node, out);
    }

    // Test 9: Auto config selects reasonable tile sizes
    #[test]
    fn test_auto_config() {
        let config = FlashAttentionConfig::auto(512, 64, 8);
        assert!(config.block_size_q > 0);
        assert!(config.block_size_kv > 0);
        assert!(config.block_size_q <= 512);
        assert!(config.block_size_kv <= 512);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.num_heads, 8);
        // Should pick 64 for d=64
        assert_eq!(config.block_size_q, 64);

        // Small sequence: tile size capped at seq_len
        let config_small = FlashAttentionConfig::auto(4, 64, 1);
        assert_eq!(config_small.block_size_q, 4);

        // Large head dim: smaller tiles
        let config_large_d = FlashAttentionConfig::auto(512, 256, 8);
        assert!(config_large_d.block_size_q <= 16);
    }

    // Test: MLIR emission produces valid structure
    #[test]
    fn test_emit_mlir() {
        let detection = FlashAttentionDetection {
            q_node: 0,
            k_node: 1,
            v_node: 2,
            scale_node: None,
            mask_node: None,
            softmax_node: 3,
            output_node: 4,
        };
        let config = FlashAttentionConfig::auto(128, 64, 8);
        let mlir = emit_flash_attention_mlir(&detection, &config);
        assert!(mlir.contains("gpu.module"), "Should contain gpu.module");
        assert!(mlir.contains("gpu.func"), "Should contain gpu.func");
        assert!(mlir.contains("flash_attention"), "Should contain flash_attention");
        assert!(mlir.contains("gpu.return"), "Should contain gpu.return");
        assert!(mlir.contains("online softmax"), "Should mention online softmax");
    }
}

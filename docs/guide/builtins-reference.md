# Builtins Reference

Complete reference of all built-in functions available in the Vortex interpreter.

---

## I/O

| Function | Signature | Description |
|----------|-----------|-------------|
| `print` | `print(value)` | Print without newline |
| `println` | `println(value)` | Print with newline |
| `read_file` | `read_file(path) -> String` | Read file contents |
| `write_file` | `write_file(path, content)` | Write string to file |
| `append_file` | `append_file(path, content)` | Append to file |
| `file_exists` | `file_exists(path) -> bool` | Check if file exists |
| `read_lines` | `read_lines(path) -> [String]` | Read file as array of lines |
| `load_csv` | `load_csv(path) -> [[String]]` | Load CSV as 2D string array |
| `read_bytes` | `read_bytes(path) -> [i64]` | Read file as byte array |
| `write_bytes` | `write_bytes(path, bytes)` | Write byte array to file |

## Math

| Function | Signature | Description |
|----------|-----------|-------------|
| `sqrt` | `sqrt(x) -> f64` | Square root |
| `sin` | `sin(x) -> f64` | Sine |
| `cos` | `cos(x) -> f64` | Cosine |
| `tan` | `tan(x) -> f64` | Tangent |
| `exp` | `exp(x) -> f64` | e^x |
| `log` | `log(x) -> f64` | Natural logarithm |
| `log2` | `log2(x) -> f64` | Base-2 logarithm |
| `log10` | `log10(x) -> f64` | Base-10 logarithm |
| `abs` | `abs(x) -> f64` | Absolute value |
| `pow` | `pow(base, exp) -> f64` | Power |
| `floor` | `floor(x) -> f64` | Floor |
| `ceil` | `ceil(x) -> f64` | Ceiling |
| `round` | `round(x) -> f64` | Round to nearest |
| `min` | `min(a, b) -> f64` | Minimum |
| `max` | `max(a, b) -> f64` | Maximum |
| `range` | `range(start, end) -> [i64]` | Generate integer range |

Constants: `PI` (3.14159...), `E` (2.71828...)

## Strings

| Function | Signature | Description |
|----------|-----------|-------------|
| `len` | `len(s) -> i64` | String or array length |
| `to_string` | `to_string(x) -> String` | Convert to string |
| `format` | `format(fmt, args...) -> String` | String formatting with `{}` |
| `split` | `split(s, delim) -> [String]` | Split string |
| `join` | `join(arr, delim) -> String` | Join array into string |
| `trim` | `trim(s) -> String` | Trim whitespace |
| `starts_with` | `starts_with(s, prefix) -> bool` | Check prefix |
| `ends_with` | `ends_with(s, suffix) -> bool` | Check suffix |
| `contains_str` | `contains_str(s, sub) -> bool` | Check substring |
| `replace` | `replace(s, from, to) -> String` | Replace occurrences |
| `to_upper` | `to_upper(s) -> String` | Uppercase |
| `to_lower` | `to_lower(s) -> String` | Lowercase |
| `substr` | `substr(s, start, len) -> String` | Substring |
| `char_at` | `char_at(s, index) -> String` | Character at index |
| `string_len` | `string_len(s) -> i64` | String length (explicit) |
| `parse_int` | `parse_int(s) -> i64` | Parse string to integer |
| `parse_float` | `parse_float(s) -> f64` | Parse string to float |

## Arrays & Functional

| Function | Signature | Description |
|----------|-----------|-------------|
| `len` | `len(arr) -> i64` | Array length |
| `push` | `push(arr, val) -> [T]` | Append element (returns new array) |
| `map` | `map(arr, fn) -> [T]` | Map function over array |
| `filter` | `filter(arr, fn) -> [T]` | Filter array by predicate |
| `fold` | `fold(arr, init, fn) -> T` | Reduce array |
| `flat_map` | `flat_map(arr, fn) -> [T]` | Map and flatten |
| `zip` | `zip(a, b) -> [(T, U)]` | Zip two arrays |
| `enumerate` | `enumerate(arr) -> [(i64, T)]` | Add indices |
| `sort` | `sort(arr) -> [T]` | Sort array |
| `reverse` | `reverse(arr) -> [T]` | Reverse array |
| `sum` | `sum(arr) -> f64` | Sum of numeric array |
| `any` | `any(arr, fn) -> bool` | Any element matches |
| `all` | `all(arr, fn) -> bool` | All elements match |
| `range` | `range(start, end) -> [i64]` | Integer range |

## HashMap

| Function | Signature | Description |
|----------|-----------|-------------|
| `hashmap` | `hashmap() -> HashMap` | Create empty map |
| `hashmap_insert` | `hashmap_insert(m, key, val) -> HashMap` | Insert entry |
| `hashmap_get` | `hashmap_get(m, key) -> Option` | Get value |
| `hashmap_remove` | `hashmap_remove(m, key) -> HashMap` | Remove entry |
| `hashmap_contains` | `hashmap_contains(m, key) -> bool` | Check key |
| `hashmap_keys` | `hashmap_keys(m) -> [String]` | All keys |
| `hashmap_values` | `hashmap_values(m) -> [T]` | All values |
| `hashmap_len` | `hashmap_len(m) -> i64` | Number of entries |

## Option & Result

| Function | Signature | Description |
|----------|-----------|-------------|
| `some` | `some(val) -> Option` | Wrap in Some |
| `none` | `none() -> Option` | Create None |
| `unwrap` | `unwrap(opt) -> T` | Unwrap (panics on None/Err) |
| `unwrap_or` | `unwrap_or(opt, default) -> T` | Unwrap with default |
| `is_some` | `is_some(opt) -> bool` | Check if Some |
| `is_none` | `is_none(opt) -> bool` | Check if None |
| `ok` | `ok(val) -> Result` | Wrap in Ok |
| `err` | `err(val) -> Result` | Wrap in Err |
| `is_ok` | `is_ok(r) -> bool` | Check if Ok |
| `is_err` | `is_err(r) -> bool` | Check if Err |

## Assertions

| Function | Signature | Description |
|----------|-----------|-------------|
| `assert` | `assert(cond, msg)` | Assert condition is true |
| `assert_eq` | `assert_eq(a, b)` | Assert equality |

## Neural Network Layers

| Function | Signature | Description |
|----------|-----------|-------------|
| `nn_linear` | `nn_linear(in, out) -> LayerID` | Dense layer |
| `nn_conv2d` | `nn_conv2d(in_ch, out_ch, kernel) -> LayerID` | Convolutional layer |
| `nn_transformer` | `nn_transformer(dim, heads, ff, blocks) -> LayerID` | Transformer |
| `nn_lstm` | `nn_lstm(in, hidden, out) -> LayerID` | LSTM layer |
| `nn_gru` | `nn_gru(in, hidden, out) -> LayerID` | GRU layer |
| `nn_embedding` | `nn_embedding(vocab, dim) -> LayerID` | Embedding layer |
| `nn_relu` | `nn_relu(id) -> LayerID` | ReLU activation |
| `nn_sigmoid` | `nn_sigmoid(id) -> LayerID` | Sigmoid activation |
| `nn_tanh` | `nn_tanh(id) -> LayerID` | Tanh activation |
| `nn_gelu` | `nn_gelu(id) -> LayerID` | GELU activation |
| `nn_softmax` | `nn_softmax(id) -> LayerID` | Softmax |
| `nn_layer_norm` | `nn_layer_norm(id) -> LayerID` | Layer normalization |
| `nn_batch_norm` | `nn_batch_norm(id) -> LayerID` | Batch normalization |
| `nn_dropout` | `nn_dropout(id) -> LayerID` | Dropout |

## Neural Network Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `nn_sequential` | `nn_sequential(layers) -> ModelID` | Compose model |
| `nn_forward` | `nn_forward(model, input) -> [f64]` | Forward pass |
| `nn_train` | `nn_train(model, data, labels, opt, epochs, lr) -> f64` | Train |
| `nn_train_verbose` | `nn_train_verbose(model, data, labels, opt, epochs, lr, every) -> f64` | Train with logging |
| `nn_predict` | `nn_predict(model, input) -> [f64]` | Inference |
| `nn_save` | `nn_save(model, path)` | Save weights |
| `nn_load` | `nn_load(model, path)` | Load weights |
| `nn_adam` | `nn_adam(model, lr)` | Configure Adam |
| `nn_sgd` | `nn_sgd(model, lr)` | Configure SGD |
| `nn_cross_entropy` | `nn_cross_entropy(pred, target) -> f64` | Cross-entropy loss |
| `nn_evaluate` | `nn_evaluate(model, data, labels) -> stats` | Evaluate metrics |
| `nn_num_params` | `nn_num_params(model) -> i64` | Parameter count |
| `nn_clone` | `nn_clone(model) -> ModelID` | Deep copy model |

## Tensor Autodiff

| Function | Signature | Description |
|----------|-----------|-------------|
| `tensor_tape_new` | `tensor_tape_new()` | Initialize tape |
| `tensor_tape_clear` | `tensor_tape_clear()` | Clear tape |
| `tensor_param` | `tensor_param(shape, data) -> TensorID` | Trainable parameter |
| `tensor_input` | `tensor_input(shape, data) -> TensorID` | Input tensor |
| `tensor_matmul` | `tensor_matmul(a, b) -> TensorID` | Matrix multiply |
| `tensor_add` | `tensor_add(a, b) -> TensorID` | Add |
| `tensor_sub` | `tensor_sub(a, b) -> TensorID` | Subtract |
| `tensor_mul` | `tensor_mul(a, b) -> TensorID` | Multiply |
| `tensor_relu` | `tensor_relu(t) -> TensorID` | ReLU |
| `tensor_sigmoid` | `tensor_sigmoid(t) -> TensorID` | Sigmoid |
| `tensor_tanh` | `tensor_tanh(t) -> TensorID` | Tanh |
| `tensor_gelu` | `tensor_gelu(t) -> TensorID` | GELU |
| `tensor_softmax` | `tensor_softmax(t) -> TensorID` | Softmax |
| `tensor_layer_norm` | `tensor_layer_norm(t) -> TensorID` | Layer norm |
| `tensor_cross_entropy` | `tensor_cross_entropy(pred, target) -> TensorID` | CE loss |
| `tensor_sum` | `tensor_sum(t) -> TensorID` | Sum |
| `tensor_mean` | `tensor_mean(t) -> TensorID` | Mean |
| `tensor_transpose` | `tensor_transpose(t) -> TensorID` | Transpose |
| `tensor_reshape` | `tensor_reshape(t, shape) -> TensorID` | Reshape |
| `tensor_broadcast_add` | `tensor_broadcast_add(t, b) -> TensorID` | Broadcast add |
| `tensor_backward` | `tensor_backward(t)` | Backward pass |
| `tensor_grad` | `tensor_grad(t) -> [f64]` | Get gradients |
| `tensor_data` | `tensor_data(t) -> [f64]` | Get data |
| `tensor_sgd` | `tensor_sgd(params, lr)` | SGD step |
| `tensor_adam` | `tensor_adam(params, lr)` | Adam step |
| `tensor_zero_grad` | `tensor_zero_grad()` | Zero gradients |

## Scalar Autodiff (Tape)

| Function | Signature | Description |
|----------|-----------|-------------|
| `tape_new` | `tape_new() -> TapeID` | Create autodiff tape |
| `tape_var` | `tape_var(tape, value) -> VarID` | Create variable |
| `tape_add` | `tape_add(tape, a, b) -> VarID` | Add |
| `tape_mul` | `tape_mul(tape, a, b) -> VarID` | Multiply |
| `tape_sub` | `tape_sub(tape, a, b) -> VarID` | Subtract |
| `tape_div` | `tape_div(tape, a, b) -> VarID` | Divide |
| `tape_exp` | `tape_exp(tape, x) -> VarID` | Exp |
| `tape_log` | `tape_log(tape, x) -> VarID` | Log |
| `tape_tanh` | `tape_tanh(tape, x) -> VarID` | Tanh |
| `tape_relu` | `tape_relu(tape, x) -> VarID` | ReLU |
| `tape_sigmoid` | `tape_sigmoid(tape, x) -> VarID` | Sigmoid |
| `tape_sin` | `tape_sin(tape, x) -> VarID` | Sin |
| `tape_cos` | `tape_cos(tape, x) -> VarID` | Cos |
| `tape_backward` | `tape_backward(tape, var)` | Compute gradients |
| `tape_grad` | `tape_grad(tape, var) -> f64` | Get gradient |
| `tape_value` | `tape_value(tape, var) -> f64` | Get value |
| `ad_sgd_step` | `ad_sgd_step(tape, vars, lr)` | SGD step |
| `ad_adam_step` | `ad_adam_step(tape, vars, lr)` | Adam step |
| `ad_mse_loss` | `ad_mse_loss(tape, pred, target) -> VarID` | MSE loss |
| `ad_cross_entropy_loss` | `ad_cross_entropy_loss(tape, pred, target) -> VarID` | CE loss |
| `zero_grad` | `zero_grad(tape)` | Zero gradients |

## Cryptography

### Elliptic Curves (secp256k1)

| Function | Signature | Description |
|----------|-----------|-------------|
| `secp256k1_generator` | `secp256k1_generator() -> ECPoint` | Generator point G |
| `scalar_mul` | `scalar_mul(scalar, point) -> ECPoint` | Scalar multiplication |
| `point_add` | `point_add(p1, p2) -> ECPoint` | Point addition |
| `point_x` | `point_x(p) -> BigInt` | X coordinate |
| `point_y` | `point_y(p) -> BigInt` | Y coordinate |
| `point_negate` | `point_negate(p) -> ECPoint` | Negate point |
| `point_sub` | `point_sub(p1, p2) -> ECPoint` | Point subtraction |
| `point_validate` | `point_validate(p) -> bool` | Validate on curve |
| `point_compress` | `point_compress(p) -> String` | Compress point |
| `point_decompress` | `point_decompress(s) -> ECPoint` | Decompress point |

### Signatures

| Function | Signature | Description |
|----------|-----------|-------------|
| `ecdsa_sign` | `ecdsa_sign(privkey_hex, msg) -> Sig` | ECDSA sign |
| `ecdsa_verify` | `ecdsa_verify(pubkey, msg, sig) -> bool` | ECDSA verify |
| `schnorr_sign` | `schnorr_sign(privkey_hex, msg) -> Sig` | Schnorr sign (BIP-340) |
| `schnorr_verify` | `schnorr_verify(pubkey_hex, msg, sig) -> bool` | Schnorr verify |

### Hashing & BigInt

| Function | Signature | Description |
|----------|-----------|-------------|
| `sha256` | `sha256(data) -> String` | SHA-256 hash |
| `sha256d` | `sha256d(data) -> String` | Double SHA-256 |
| `bigint_from_hex` | `bigint_from_hex(hex) -> BigInt` | Parse hex to BigInt |
| `to_hex` | `to_hex(bigint) -> String` | BigInt to hex |

### Field Arithmetic

| Function | Signature | Description |
|----------|-----------|-------------|
| `field_new` | `field_new(hex, prime_hex) -> FieldElem` | Create field element |
| `field_add` | `field_add(a, b) -> FieldElem` | Field addition |
| `field_sub` | `field_sub(a, b) -> FieldElem` | Field subtraction |
| `field_mul` | `field_mul(a, b) -> FieldElem` | Field multiplication |
| `field_inv` | `field_inv(a) -> FieldElem` | Field inverse |
| `field_pow` | `field_pow(a, exp) -> FieldElem` | Field exponentiation |
| `field_neg` | `field_neg(a) -> FieldElem` | Field negation |
| `field_zero` | `field_zero(prime) -> FieldElem` | Zero element |
| `field_one` | `field_one(prime) -> FieldElem` | One element |
| `field_eq` | `field_eq(a, b) -> bool` | Equality check |
| `field_prime` | `field_prime(a) -> String` | Get prime modulus |
| `field_from_hex` | `field_from_hex(hex) -> FieldElem` | From hex (secp256k1 prime) |
| `scalar_new` | `scalar_new(hex) -> FieldElem` | Scalar field element |
| `scalar_inv` | `scalar_inv(a) -> FieldElem` | Scalar inverse |
| `scalar_neg` | `scalar_neg(a) -> FieldElem` | Scalar negation |

### Montgomery / Modular

| Function | Signature | Description |
|----------|-----------|-------------|
| `modfield_new` | `modfield_new(val, prime) -> ModField` | Montgomery form |
| `modfield_add` | `modfield_add(a, b) -> ModField` | Modular add |
| `modfield_sub` | `modfield_sub(a, b) -> ModField` | Modular sub |
| `modfield_mul` | `modfield_mul(a, b) -> ModField` | Modular mul |
| `modfield_inv` | `modfield_inv(a) -> ModField` | Modular inverse |
| `modfield_pow` | `modfield_pow(a, exp) -> ModField` | Modular power |
| `modfield_neg` | `modfield_neg(a) -> ModField` | Modular negation |
| `montgomery_mul` | `montgomery_mul(a, b) -> ModField` | Montgomery multiply |

### Advanced Crypto

| Function | Signature | Description |
|----------|-----------|-------------|
| `ntt` | `ntt(coeffs, prime) -> [i64]` | Number Theoretic Transform |
| `intt` | `intt(coeffs, prime) -> [i64]` | Inverse NTT |
| `msm` | `msm(scalars, points) -> ECPoint` | Multi-scalar multiplication |
| `pairing` | `pairing(g1, g2) -> PairingResult` | Bilinear pairing |
| `pairing_check` | `pairing_check(pairs) -> bool` | Pairing verification |
| `poly_mul` | `poly_mul(a, b) -> [f64]` | Polynomial multiplication |
| `poly_eval` | `poly_eval(coeffs, x) -> f64` | Polynomial evaluation |
| `poly_interpolate` | `poly_interpolate(points) -> [f64]` | Lagrange interpolation |
| `fp2_new` | `fp2_new(a, b) -> Fp2` | Extension field element |
| `fp2_mul` | `fp2_mul(a, b) -> Fp2` | Fp2 multiplication |
| `fp2_add` | `fp2_add(a, b) -> Fp2` | Fp2 addition |
| `fp2_inv` | `fp2_inv(a) -> Fp2` | Fp2 inverse |
| `g1_generator` | `g1_generator() -> G1Point` | BLS12-381 G1 generator |
| `g2_generator` | `g2_generator() -> G2Point` | BLS12-381 G2 generator |
| `g1_scalar_mul` | `g1_scalar_mul(s, p) -> G1Point` | G1 scalar multiply |

### Verifiable Inference / ZK

| Function | Signature | Description |
|----------|-----------|-------------|
| `zk_compile_model` | `zk_compile_model(model) -> Circuit` | Compile model to ZK circuit |
| `zk_prove_inference` | `zk_prove_inference(circuit, input) -> Proof` | Prove inference |
| `zk_verify` | `zk_verify(proof) -> bool` | Verify proof |
| `fhe_encrypt` | `fhe_encrypt(data) -> Ciphertext` | FHE encrypt |
| `fhe_decrypt` | `fhe_decrypt(ct) -> [f64]` | FHE decrypt |
| `fhe_inference` | `fhe_inference(model, ct) -> Ciphertext` | Encrypted inference |

## GPU Runtime

| Function | Signature | Description |
|----------|-----------|-------------|
| `gpu_alloc` | `gpu_alloc(size) -> BufID` | Allocate buffer |
| `gpu_free` | `gpu_free(buf)` | Free buffer |
| `gpu_matmul` | `gpu_matmul(a, b) -> [f64]` | Matrix multiply |
| `gpu_add` | `gpu_add(a, b) -> [f64]` | Elementwise add |
| `gpu_mul` | `gpu_mul(a, b) -> [f64]` | Elementwise multiply |
| `gpu_relu` | `gpu_relu(a) -> [f64]` | ReLU |
| `gpu_softmax` | `gpu_softmax(a) -> [f64]` | Softmax |
| `gpu_copy_to_host` | `gpu_copy_to_host(buf) -> [f64]` | Download |
| `gpu_copy_to_device` | `gpu_copy_to_device(data) -> BufID` | Upload |
| `gpu_available` | `gpu_available() -> bool` | Check GPU |
| `gpu_native_matmul` | `gpu_native_matmul(a, b) -> [f64]` | Native GPU matmul |
| `gpu_train_step` | `gpu_train_step(model, data, lr)` | GPU training step |
| `gpu_benchmark` | `gpu_benchmark(op, size, iters) -> stats` | Benchmark |

## Quantization

| Function | Signature | Description |
|----------|-----------|-------------|
| `quantize` | `quantize(tensor, bits) -> QTensor` | Quantize |
| `dequantize` | `dequantize(qt) -> [f64]` | Dequantize |
| `quantized_matmul` | `quantized_matmul(a, b) -> QTensor` | Quantized matmul |
| `compression_ratio` | `compression_ratio(qt) -> f64` | Compression ratio |

## Neuromorphic / Spiking

| Function | Signature | Description |
|----------|-----------|-------------|
| `spike_train` | `spike_train(timesteps, neurons) -> SpikeTrain` | Create spike train |
| `spike_from_dense` | `spike_from_dense(data) -> SpikeTrain` | From dense tensor |
| `spike_to_dense` | `spike_to_dense(st) -> [f64]` | To dense tensor |
| `spike_overlap` | `spike_overlap(a, b) -> f64` | Spike correlation |
| `lif_layer` | `lif_layer(input, weights, threshold) -> SpikeTrain` | LIF neuron layer |
| `spike_attention` | `spike_attention(q, k, v) -> [f64]` | Spike-based attention |

## State Space Models (SSM)

| Function | Signature | Description |
|----------|-----------|-------------|
| `ssm_scan` | `ssm_scan(A, B, C, x) -> [f64]` | Sequential SSM scan |
| `selective_ssm` | `selective_ssm(x, A, B, C, delta) -> [f64]` | Selective SSM (Mamba) |
| `parallel_scan` | `parallel_scan(x, coeffs) -> [f64]` | Parallel prefix scan |
| `ssm_parallel_scan` | `ssm_parallel_scan(A, B, C, x) -> [f64]` | Parallel SSM scan |

## Differentiable Memory

| Function | Signature | Description |
|----------|-----------|-------------|
| `memory_new` | `memory_new(capacity, key_dim, val_dim) -> Memory` | Create memory |
| `memory_read` | `memory_read(mem, key) -> [f64]` | Read by key |
| `memory_write` | `memory_write(mem, key, value)` | Write key-value |
| `memory_content_lookup` | `memory_content_lookup(mem, query) -> [f64]` | Content-based lookup |

## ODEs & Liquid Networks

| Function | Signature | Description |
|----------|-----------|-------------|
| `ode_solve_euler` | `ode_solve_euler(f, y0, t0, t1, steps) -> [f64]` | Euler integration |
| `ode_solve_rk4` | `ode_solve_rk4(f, y0, t0, t1, steps) -> [f64]` | RK4 integration |
| `rk45_solve` | `rk45_solve(f, y0, t_span, tol) -> [f64]` | Adaptive RK45 |
| `liquid_cell` | `liquid_cell(x, hidden, tau, dt) -> [f64]` | Liquid time-constant cell |
| `cfc_cell` | `cfc_cell(x, hidden, tau) -> [f64]` | Closed-form continuous cell |

## Signal Processing

| Function | Signature | Description |
|----------|-----------|-------------|
| `fft` | `fft(signal) -> [f64]` | Fast Fourier Transform |
| `ifft` | `ifft(spectrum) -> [f64]` | Inverse FFT |
| `fft_convolve` | `fft_convolve(a, b) -> [f64]` | FFT convolution |
| `fnet_mix` | `fnet_mix(x) -> [f64]` | FNet-style mixing |
| `linear_attention` | `linear_attention(q, k, v) -> [f64]` | Linear attention |

## Sparse Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `to_sparse` | `to_sparse(tensor) -> SparseTensor` | Convert to sparse |
| `sparse_topk` | `sparse_topk(tensor, k) -> SparseIdx` | Top-k sparse |
| `sparse_gather` | `sparse_gather(tensor, idx) -> [f64]` | Sparse gather |
| `sparse_scatter` | `sparse_scatter(vals, idx, size) -> [f64]` | Sparse scatter |
| `sparse_matmul` | `sparse_matmul(a, b) -> [f64]` | Sparse matrix multiply |

## Dynamic Tensors

| Function | Signature | Description |
|----------|-----------|-------------|
| `dyn_tensor` | `dyn_tensor(data, shape) -> DynTensor` | Create dynamic tensor |
| `compact` | `compact(tensor) -> DynTensor` | Remove padding |
| `pad` | `pad(tensor, size) -> DynTensor` | Pad to size |
| `stream_compact` | `stream_compact(tensor) -> DynTensor` | Stream compaction |

## Learning Rules

| Function | Signature | Description |
|----------|-----------|-------------|
| `goodness` | `goodness(activations) -> f64` | Forward-forward goodness |
| `hebbian_update` | `hebbian_update(weights, pre, post, lr) -> [f64]` | Hebbian learning |
| `ff_layer` | `ff_layer(input, weights, threshold) -> [f64]` | Forward-forward layer |
| `predictive_coding_update` | `predictive_coding_update(pred, target, lr) -> [f64]` | Predictive coding |
| `oja_update` | `oja_update(weights, input, lr) -> [f64]` | Oja's rule |
| `chunked_scan` | `chunked_scan(x, chunk_size) -> [f64]` | Chunked scan |

## Advanced Architectures

| Function | Signature | Description |
|----------|-----------|-------------|
| `spike_ssm_new` | `spike_ssm_new(in, hidden, out) -> ModelID` | Spike-SSM hybrid |
| `spike_ssm_forward` | `spike_ssm_forward(model, input) -> [f64]` | Forward pass |
| `spike_ssm_train_step` | `spike_ssm_train_step(model, input, target, lr) -> f64` | Train step |
| `spike_ssm_stats` | `spike_ssm_stats(model) -> String` | Model statistics |
| `tiered_moe_new` | `tiered_moe_new(in, expert, n_experts, tiers) -> ID` | Tiered MoE |
| `tiered_moe_forward` | `tiered_moe_forward(model, input) -> [f64]` | MoE forward |
| `tiered_moe_stats` | `tiered_moe_stats(model) -> String` | MoE stats |
| `hetero_layer_new` | `hetero_layer_new(in, out) -> ID` | Heterogeneous layer |
| `hetero_layer_forward` | `hetero_layer_forward(layer, input) -> [f64]` | Hetero forward |
| `hetero_layer_stats` | `hetero_layer_stats(layer) -> String` | Hetero stats |
| `multiscale_model_new` | `multiscale_model_new(in, scales) -> ID` | Multiscale model |
| `multiscale_model_forward` | `multiscale_model_forward(model, input) -> [f64]` | Multiscale forward |
| `multiscale_model_stats` | `multiscale_model_stats(model) -> String` | Multiscale stats |

## Continuous Learning

| Function | Signature | Description |
|----------|-----------|-------------|
| `continuous_learner_new` | `continuous_learner_new(layer_sizes) -> ID` | Create learner |
| `continuous_learner_infer` | `continuous_learner_infer(id, input) -> [f64]` | Inference |
| `continuous_learner_learn` | `continuous_learner_learn(id, input, target, lr) -> f64` | Online learning |
| `continuous_learner_stats` | `continuous_learner_stats(id) -> String` | Stats |

## Self-Modifying Models

| Function | Signature | Description |
|----------|-----------|-------------|
| `dynamic_model_new` | `dynamic_model_new(in, hidden, out) -> ID` | Create |
| `dynamic_model_forward` | `dynamic_model_forward(id, input) -> [f64]` | Forward |
| `dynamic_model_add_layer` | `dynamic_model_add_layer(id, size)` | Add layer |
| `dynamic_model_remove_layer` | `dynamic_model_remove_layer(id, index)` | Remove layer |
| `dynamic_model_search_step` | `dynamic_model_search_step(id, input, target, lr) -> f64` | NAS step |
| `dynamic_model_stats` | `dynamic_model_stats(id) -> String` | Stats |

## Adaptive Inference

| Function | Signature | Description |
|----------|-----------|-------------|
| `adaptive_model_new` | `adaptive_model_new(in, hidden, out, depths) -> ID` | Create |
| `adaptive_model_forward` | `adaptive_model_forward(id, input) -> [f64]` | Forward (adaptive depth) |
| `adaptive_model_stats` | `adaptive_model_stats(id) -> String` | Stats |
| `adaptive_model_tune` | `adaptive_model_tune(id, threshold)` | Tune exit threshold |

## Symbolic Reasoning

| Function | Signature | Description |
|----------|-----------|-------------|
| `symbolic_eval` | `symbolic_eval(op, args) -> Value` | Evaluate symbolic op (gcd, is_prime, etc.) |
| `hybrid_layer_new` | `hybrid_layer_new(in, out) -> ID` | Neural-symbolic hybrid |
| `hybrid_layer_forward` | `hybrid_layer_forward(id, input) -> [f64]` | Hybrid forward |

## Attention Variants

| Function | Signature | Description |
|----------|-----------|-------------|
| `softmax` | `softmax(arr) -> [f64]` | Softmax |
| `gelu` | `gelu(x) -> f64` | GELU activation |
| `layer_norm` | `layer_norm(arr) -> [f64]` | Layer normalization |
| `attention` | `attention(q, k, v) -> [f64]` | Scaled dot-product attention |
| `rope` | `rope(x, pos) -> [f64]` | Rotary position encoding |
| `flash_attention` | `flash_attention(q, k, v, block) -> [f64]` | Flash attention |
| `flash_attention_backward` | `flash_attention_backward(q, k, v, grad, block) -> grads` | Flash attention backward |

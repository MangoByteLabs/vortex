# Neural Networks in Vortex

Vortex has neural network primitives built directly into the language. No imports, no framework installation, no Python glue.

## Creating Layers

### Linear (Dense) Layer

```vortex
let layer = nn_linear(input_size, output_size)
// e.g., nn_linear(784, 128) -- 784 inputs, 128 outputs
```

### Convolutional Layer

```vortex
let conv = nn_conv2d(in_channels, out_channels, kernel_size)
// e.g., nn_conv2d(3, 16, 3) -- 3 input channels, 16 filters, 3x3 kernel
```

### Transformer

```vortex
let transformer = nn_transformer(d_model, n_heads, ff_dim, n_blocks)
// e.g., nn_transformer(64, 4, 128, 2)
```

### LSTM / GRU

```vortex
let lstm = nn_lstm(input_size, hidden_size, output_size)
let gru = nn_gru(input_size, hidden_size, output_size)
```

### Embedding

```vortex
let emb = nn_embedding(vocab_size, embed_dim)
```

## Activation & Normalization Layers

```vortex
nn_relu(layer_id)
nn_sigmoid(layer_id)
nn_tanh(layer_id)
nn_gelu(layer_id)
nn_softmax(layer_id)
nn_layer_norm(layer_id)
nn_batch_norm(layer_id)
nn_dropout(layer_id)
```

## Composing Models

Use `nn_sequential` to stack layers into a model:

```vortex
let l1 = nn_linear(2, 16)
let l2 = nn_linear(16, 8)
let l3 = nn_linear(8, 1)
let model = nn_sequential([l1, l2, l3])
```

## Training

### nn_train

Train a model with one function call:

```vortex
let losses = nn_train(model, data, labels, optimizer, epochs, learning_rate)
```

**Parameters:**
- `model` -- Model ID from `nn_sequential`
- `data` -- 2D array of input samples `[[x1, x2], [x1, x2], ...]`
- `labels` -- 2D array of target values `[[y1], [y2], ...]`
- `optimizer` -- `"adam"`, `"sgd"`, or `"adamw"`
- `epochs` -- Number of training iterations (integer)
- `learning_rate` -- Float, typically `0.001` to `0.01`

**Returns:** Final loss value.

### nn_train_verbose

Like `nn_train` but prints loss at regular intervals:

```vortex
nn_train_verbose(model, data, labels, "adam", 1000, 0.01, 100)
// Prints loss every 100 epochs
```

## Inference

```vortex
let output = nn_predict(model, input)
// input: 1D array matching the model's input size
// output: array of predicted values

let output = nn_forward(model, input)
// Same as nn_predict (alias)
```

## Model Inspection

```vortex
let params = nn_num_params(model)    // Total parameter count
let stats = nn_evaluate(model, test_data, test_labels)  // Evaluation metrics
let copy = nn_clone(model)           // Deep copy a model
```

## Saving and Loading

```vortex
nn_save(model, "model.json")   // Save weights to JSON
nn_load(model, "model.json")   // Load weights from JSON
```

## Loss Functions

```vortex
let loss = nn_cross_entropy(predictions, targets)
```

MSE loss is used by default in `nn_train`. Specify `"cross_entropy"` or `"bce"` in the MCP server for other losses.

## Optimizers

```vortex
nn_adam(model, lr)     // Configure Adam optimizer
nn_sgd(model, lr)      // Configure SGD optimizer
```

## Full Example: XOR Classifier

```vortex
fn main() {
    // Define model: 2 -> 8 -> 1
    let l1 = nn_linear(2, 8)
    let l2 = nn_linear(8, 1)
    let model = nn_sequential([l1, l2])

    // Training data
    let data   = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    let labels = [[0.0], [1.0], [1.0], [0.0]]

    // Train with Adam optimizer for 1000 epochs
    let loss = nn_train(model, data, labels, "adam", 1000, 0.01)
    println(format("Final loss: {}", to_string(loss)))

    // Predict
    println("Predictions:")
    for x in data {
        let pred = nn_predict(model, x)
        println(format("  {} -> {}", to_string(x), to_string(pred)))
    }

    // Save the trained model
    nn_save(model, "xor_model.json")
}
```

## Tensor Autodiff System

For lower-level control, Vortex provides a tensor autodiff tape:

```vortex
fn main() {
    tensor_tape_new()

    // Create parameters (tracked for gradients)
    let W1 = tensor_param([4, 2], [0.1, 0.2, -0.1, 0.3, 0.2, -0.2, 0.1, 0.4])
    let b1 = tensor_param([1, 4], [0.0, 0.0, 0.0, 0.0])

    // Forward pass
    let x = tensor_input([1, 2], [1.0, 0.5])
    let h = tensor_relu(tensor_add(tensor_matmul(x, W1), b1))

    // Backward pass
    tensor_backward(h)
    let grad_W1 = tensor_grad(W1)

    // Update weights
    tensor_sgd([W1, b1], 0.01)
    tensor_zero_grad()
}
```

### Tensor Operations

| Function | Description |
|----------|-------------|
| `tensor_param(shape, data)` | Create a trainable parameter |
| `tensor_input(shape, data)` | Create an input tensor (no grad) |
| `tensor_matmul(a, b)` | Matrix multiplication |
| `tensor_add(a, b)` | Elementwise addition |
| `tensor_sub(a, b)` | Elementwise subtraction |
| `tensor_mul(a, b)` | Elementwise multiplication |
| `tensor_relu(t)` | ReLU activation |
| `tensor_sigmoid(t)` | Sigmoid activation |
| `tensor_tanh(t)` | Tanh activation |
| `tensor_gelu(t)` | GELU activation |
| `tensor_softmax(t)` | Softmax |
| `tensor_layer_norm(t)` | Layer normalization |
| `tensor_cross_entropy(pred, target)` | Cross-entropy loss |
| `tensor_sum(t)` | Sum all elements |
| `tensor_mean(t)` | Mean of all elements |
| `tensor_transpose(t)` | Transpose |
| `tensor_reshape(t, shape)` | Reshape tensor |
| `tensor_broadcast_add(t, b)` | Broadcast addition |
| `tensor_backward(t)` | Compute gradients |
| `tensor_grad(t)` | Get gradient tensor |
| `tensor_data(t)` | Get raw data |
| `tensor_sgd(params, lr)` | SGD update step |
| `tensor_adam(params, lr)` | Adam update step |
| `tensor_zero_grad()` | Zero all gradients |

## Continuous Learning

Models that learn while serving inference:

```vortex
let model = continuous_learner_new([4, 8, 2])

// Learn from new data on the fly
let loss = continuous_learner_learn(model, input, target, 0.01)

// Inference
let output = continuous_learner_infer(model, input)

// Check stats
let stats = continuous_learner_stats(model)
```

## Advanced Architectures

### Spike-SSM Hybrid

```vortex
let model = spike_ssm_new(input_dim, hidden_dim, output_dim)
let out = spike_ssm_forward(model, input)
let loss = spike_ssm_train_step(model, input, target, lr)
```

### Tiered Mixture of Experts

```vortex
let moe = tiered_moe_new(input_dim, expert_dim, num_experts, num_tiers)
let out = tiered_moe_forward(moe, input)
```

### Self-Modifying Models

```vortex
let model = dynamic_model_new(input_dim, hidden_dim, output_dim)
dynamic_model_add_layer(model, size)
dynamic_model_remove_layer(model, index)
dynamic_model_search_step(model, input, target, lr)
```

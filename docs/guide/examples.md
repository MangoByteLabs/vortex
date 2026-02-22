# Vortex Examples

## Hello World

```vortex
fn main() {
    println("Hello, Vortex!")
}
```

```bash
vortex run hello.vx
```

## Fibonacci

```vortex
fn fibonacci(n: i64) -> i64 {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}

fn main() {
    for i in range(0, 15) {
        println(format("fib({}) = {}", to_string(i), to_string(fibonacci(i))))
    }
}
```

## Structs and Methods

```vortex
struct Point {
    x: f64
    y: f64
}

fn distance(p: Point) -> f64 {
    return sqrt(p.x * p.x + p.y * p.y)
}

fn main() {
    let p = Point { x: 3.0, y: 4.0 }
    println(distance(p))    // 5.0
}
```

## Functional Programming

```vortex
fn main() {
    let nums = range(1, 11)

    let squares = map(nums, |x| x * x)
    println(squares)

    let evens = filter(nums, |x| x % 2 == 0)
    println(evens)

    let total = fold(nums, 0, |acc, x| acc + x)
    println(format("Sum 1..10 = {}", to_string(total)))

    let pairs = zip(range(0, 5), ["a", "b", "c", "d", "e"])
    println(pairs)
}
```

## Neural Network Classifier

Train a model to learn the XOR function:

```vortex
fn main() {
    let l1 = nn_linear(2, 8)
    let l2 = nn_linear(8, 1)
    let model = nn_sequential([l1, l2])

    let data   = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    let labels = [[0.0], [1.0], [1.0], [0.0]]

    let loss = nn_train(model, data, labels, "adam", 1000, 0.01)
    println(format("Final loss: {}", to_string(loss)))

    println("Predictions:")
    for x in data {
        println(format("  {} -> {}", to_string(x), to_string(nn_predict(model, x))))
    }
}
```

## Transformer Forward Pass

A complete mini-transformer implementation in pure Vortex:

```vortex
fn dot(a: [f64], b: [f64]) -> f64 {
    var s = 0.0
    for i in 0..len(a) {
        s += a[i] * b[i]
    }
    return s
}

fn vec_add(a: [f64], b: [f64]) -> [f64] {
    var result = []
    for i in 0..len(a) {
        result = push(result, a[i] + b[i])
    }
    return result
}

fn matvec(mat: [f64], v: [f64]) -> [f64] {
    return map(mat, |row| dot(row, v))
}

fn single_head_attn(q: [f64], k_vecs: [f64], v_vecs: [f64], d_head: i64) -> [f64] {
    let scale = (d_head * 1.0) ** 0.5
    var scores = []
    for i in 0..len(k_vecs) {
        scores = push(scores, dot(q, k_vecs[i]) / scale)
    }
    let weights = softmax(scores)

    var result = []
    for j in 0..len(q) {
        result = push(result, 0.0)
    }
    for i in 0..len(v_vecs) {
        let w = weights[i]
        let v = v_vecs[i]
        var new_result = []
        for j in 0..len(q) {
            new_result = push(new_result, result[j] + v[j] * w)
        }
        result = new_result
    }
    return result
}

fn main() {
    println("=== Mini Transformer ===")
    let tokens = [1, 3, 5, 2]

    // Embed tokens
    var x_seq = []
    for i in 0..4 {
        let t = tokens[i] * 1.0
        x_seq = push(x_seq, [t * 0.1, t * 0.2, t * 0.3, t * 0.4])
    }

    // Attention with softmax
    let q = x_seq[3]
    let out = single_head_attn(q, x_seq, x_seq, 4)
    println(format("Attention output: {}", to_string(out)))
}
```

See `examples/transformer.vx` in the repo for the full multi-head version with FFN and layer norm.

## Crypto Wallet

Generate keys, sign messages, verify signatures:

```vortex
fn main() {
    // Generate keypair
    let sk = bigint_from_hex("DEADBEEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEFDEADBEEF")
    let G = secp256k1_generator()
    let pubkey = scalar_mul(sk, G)
    println(format("Public key X: {}", to_hex(point_x(pubkey))))

    // ECDSA sign and verify
    let message = "Hello from Vortex wallet!"
    let sig = ecdsa_sign("DEADBEEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEFDEADBEEF", message)
    let valid = ecdsa_verify(pubkey, message, sig)
    println(format("ECDSA valid: {}", to_string(valid)))
    assert(valid, "ECDSA verification failed!")

    // SHA-256 transaction hash
    let tx_hash = sha256("send:42:to:recipient")
    println(format("TX hash: {}", tx_hash))
}
```

## Continuous Learning Server

A model that learns while serving inference:

```vortex
fn main() {
    let model = continuous_learner_new([4, 8, 2])

    // Train on incoming data
    var i = 0
    while i < 100 {
        let loss = continuous_learner_learn(model, [1.0, 0.0, 1.0, 0.0], [1.0, 0.0], 0.01)
        if i % 20 == 0 {
            println(format("Step {}: loss = {}", to_string(i), to_string(loss)))
        }
        i = i + 1
    }

    // Inference
    let prediction = continuous_learner_infer(model, [1.0, 0.0, 1.0, 0.0])
    println(format("Prediction: {}", to_string(prediction)))
}
```

## Symbolic Math

```vortex
fn main() {
    println(symbolic_eval("gcd", [48, 18]))        // 6
    println(symbolic_eval("is_prime", [97]))        // true
    println(symbolic_eval("is_prime", [100]))       // false

    // Math builtins
    println(sqrt(144.0))       // 12.0
    println(sin(PI / 2.0))    // 1.0
    println(pow(2.0, 10.0))   // 1024.0
}
```

## More Examples

The `examples/` directory in the repository contains many more examples:

| File | Description |
|------|-------------|
| `hello_vortex.vx` | Complete tour of Vortex features |
| `train_xor.vx` | XOR neural network training |
| `transformer.vx` | Full multi-head transformer inference |
| `crypto_wallet.vx` | ECDSA/Schnorr wallet operations |
| `train_transformer.vx` | Transformer training with tensor autodiff |
| `spike_ssm_hybrid.vx` | Spike-SSM architecture demo |
| `liquid_neural_ode.vx` | Liquid neural networks with ODE solvers |
| `quantized_inference.vx` | Quantized model inference |
| `verifiable_inference.vx` | ZK proofs for model inference |
| `moe_router.vx` | Mixture-of-experts routing |
| `forward_forward.vx` | Forward-forward learning algorithm |
| `spiking_network.vx` | Neuromorphic spiking networks |
| `multiscale_reasoning.vx` | Multiscale reasoning architecture |
| `self_improving_server.vx` | Self-modifying neural network server |

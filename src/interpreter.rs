use crate::ast::*;
use crate::crypto;
use std::collections::HashMap;
use std::fmt;

/// Runtime value in the Vortex interpreter
#[derive(Debug, Clone)]
pub enum Value {
    Int(i128),
    Float(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    Tuple(Vec<Value>),
    /// Big integer (256-bit), stored as hex string internally
    BigInt(crypto::BigUint256),
    /// Field element (modular arithmetic)
    FieldElem(crypto::FieldElement),
    /// Elliptic curve point
    ECPoint(crypto::ECPoint),
    /// Void / unit
    Void,
    /// A struct instance
    Struct {
        name: String,
        fields: HashMap<String, Value>,
    },
    /// An enum variant value
    EnumVariant {
        enum_name: String,
        variant: String,
        fields: Vec<Value>,
    },
    /// A function value (closure)
    Function {
        name: String,
        params: Vec<String>,
        body: Block,
    },
    /// Return signal (used internally)
    Return(Box<Value>),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::String(s) => write!(f, "{}", s),
            Value::Array(elems) => {
                write!(f, "[")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", e)?;
                }
                write!(f, "]")
            }
            Value::Tuple(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", e)?;
                }
                write!(f, ")")
            }
            Value::Struct { name, fields } => {
                write!(f, "{} {{ ", name)?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, " }}")
            }
            Value::EnumVariant { enum_name, variant, fields } => {
                write!(f, "{}::{}", enum_name, variant)?;
                if !fields.is_empty() {
                    write!(f, "(")?;
                    for (i, v) in fields.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}", v)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
            Value::BigInt(n) => write!(f, "0x{}", n),
            Value::FieldElem(fe) => write!(f, "{}", fe),
            Value::ECPoint(p) => write!(f, "{}", p),
            Value::Void => write!(f, "()"),
            Value::Function { name, .. } => write!(f, "<fn {}>", name),
            Value::Return(v) => write!(f, "{}", v),
        }
    }
}

/// Environment for the interpreter
struct Env {
    scopes: Vec<HashMap<String, Value>>,
    functions: HashMap<String, FnDef>,
    /// Struct definitions: name -> field names (in order)
    struct_defs: HashMap<String, Vec<String>>,
    output: Vec<String>,
}

#[derive(Clone)]
enum FnDef {
    User {
        params: Vec<String>,
        body: Block,
    },
    Builtin(fn(&mut Env, Vec<Value>) -> Result<Value, String>),
}

impl Env {
    fn new() -> Self {
        let mut env = Self {
            scopes: vec![HashMap::new()],
            functions: HashMap::new(),
            struct_defs: HashMap::new(),
            output: Vec::new(),
        };
        env.register_builtins();
        env
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define(&mut self, name: &str, val: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), val);
        }
    }

    fn get(&self, name: &str) -> Option<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(name) {
                return Some(val.clone());
            }
        }
        None
    }

    fn set(&mut self, name: &str, val: Value) -> bool {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), val);
                return true;
            }
        }
        false
    }

    fn register_builtins(&mut self) {
        // print(value)
        self.functions.insert("print".to_string(), FnDef::Builtin(builtin_print));
        self.functions.insert("println".to_string(), FnDef::Builtin(builtin_println));

        // Crypto builtins
        self.functions.insert("secp256k1_generator".to_string(), FnDef::Builtin(builtin_secp256k1_generator));
        self.functions.insert("scalar_mul".to_string(), FnDef::Builtin(builtin_scalar_mul));
        self.functions.insert("point_add".to_string(), FnDef::Builtin(builtin_point_add));
        self.functions.insert("field_from_hex".to_string(), FnDef::Builtin(builtin_field_from_hex));
        self.functions.insert("field_inv".to_string(), FnDef::Builtin(builtin_field_inv));
        self.functions.insert("field_mul".to_string(), FnDef::Builtin(builtin_field_mul));
        self.functions.insert("field_add".to_string(), FnDef::Builtin(builtin_field_add));
        self.functions.insert("point_x".to_string(), FnDef::Builtin(builtin_point_x));
        self.functions.insert("point_y".to_string(), FnDef::Builtin(builtin_point_y));
        self.functions.insert("bigint_from_hex".to_string(), FnDef::Builtin(builtin_bigint_from_hex));
        self.functions.insert("to_hex".to_string(), FnDef::Builtin(builtin_to_hex));

        // Extended field arithmetic builtins
        self.functions.insert("field_new".to_string(), FnDef::Builtin(builtin_field_new));
        self.functions.insert("field_sub".to_string(), FnDef::Builtin(builtin_field_sub));
        self.functions.insert("field_pow".to_string(), FnDef::Builtin(builtin_field_pow));
        self.functions.insert("field_neg".to_string(), FnDef::Builtin(builtin_field_neg));
        self.functions.insert("field_zero".to_string(), FnDef::Builtin(builtin_field_zero));
        self.functions.insert("field_one".to_string(), FnDef::Builtin(builtin_field_one));
        self.functions.insert("field_eq".to_string(), FnDef::Builtin(builtin_field_eq));
        self.functions.insert("field_prime".to_string(), FnDef::Builtin(builtin_field_prime));

        // Scalar (mod N) builtins
        self.functions.insert("scalar_new".to_string(), FnDef::Builtin(builtin_scalar_new));
        self.functions.insert("scalar_inv".to_string(), FnDef::Builtin(builtin_scalar_inv));
        self.functions.insert("scalar_neg".to_string(), FnDef::Builtin(builtin_scalar_neg));

        // Point operations
        self.functions.insert("point_negate".to_string(), FnDef::Builtin(builtin_point_negate));
        self.functions.insert("point_sub".to_string(), FnDef::Builtin(builtin_point_sub));
        self.functions.insert("point_validate".to_string(), FnDef::Builtin(builtin_point_validate));
        self.functions.insert("point_compress".to_string(), FnDef::Builtin(builtin_point_compress));
        self.functions.insert("point_decompress".to_string(), FnDef::Builtin(builtin_point_decompress));

        // SHA-256
        self.functions.insert("sha256".to_string(), FnDef::Builtin(builtin_sha256));
        self.functions.insert("sha256d".to_string(), FnDef::Builtin(builtin_sha256d));

        // ECDSA
        self.functions.insert("ecdsa_sign".to_string(), FnDef::Builtin(builtin_ecdsa_sign));
        self.functions.insert("ecdsa_verify".to_string(), FnDef::Builtin(builtin_ecdsa_verify));

        // Schnorr (BIP-340)
        self.functions.insert("schnorr_sign".to_string(), FnDef::Builtin(builtin_schnorr_sign));
        self.functions.insert("schnorr_verify".to_string(), FnDef::Builtin(builtin_schnorr_verify));

        // String / utility builtins
        self.functions.insert("len".to_string(), FnDef::Builtin(builtin_len));
        self.functions.insert("to_string".to_string(), FnDef::Builtin(builtin_to_string));
        self.functions.insert("format".to_string(), FnDef::Builtin(builtin_format));
        self.functions.insert("push".to_string(), FnDef::Builtin(builtin_push));
        self.functions.insert("assert".to_string(), FnDef::Builtin(builtin_assert));
        self.functions.insert("assert_eq".to_string(), FnDef::Builtin(builtin_assert_eq));
    }
}

// --- Builtin functions ---

fn builtin_print(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let s: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
    let output = s.join(" ");
    print!("{}", output);
    env.output.push(output);
    Ok(Value::Void)
}

fn builtin_println(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let s: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
    let output = s.join(" ");
    println!("{}", output);
    env.output.push(output);
    Ok(Value::Void)
}

fn builtin_secp256k1_generator(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    Ok(Value::ECPoint(crypto::secp256k1_generator()))
}

fn builtin_scalar_mul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("scalar_mul expects 2 arguments: (scalar, point)".to_string());
    }
    let scalar = match &args[0] {
        Value::FieldElem(fe) => fe.value.clone(),
        Value::BigInt(n) => n.clone(),
        _ => return Err("scalar_mul: first argument must be a field element or bigint".to_string()),
    };
    let point = match &args[1] {
        Value::ECPoint(p) => p.clone(),
        _ => return Err("scalar_mul: second argument must be an EC point".to_string()),
    };
    Ok(Value::ECPoint(crypto::scalar_mul(&scalar, &point)))
}

fn builtin_point_add(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("point_add expects 2 arguments".to_string());
    }
    let p1 = match &args[0] {
        Value::ECPoint(p) => p.clone(),
        _ => return Err("point_add: arguments must be EC points".to_string()),
    };
    let p2 = match &args[1] {
        Value::ECPoint(p) => p.clone(),
        _ => return Err("point_add: arguments must be EC points".to_string()),
    };
    Ok(Value::ECPoint(crypto::point_add(&p1, &p2)))
}

fn builtin_field_from_hex(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("field_from_hex expects 1 argument".to_string());
    }
    let hex = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("field_from_hex: argument must be a string".to_string()),
    };
    let n = crypto::BigUint256::from_hex(&hex)
        .ok_or_else(|| format!("invalid hex: {}", hex))?;
    Ok(Value::FieldElem(crypto::FieldElement::new(n, crypto::secp256k1_field_prime())))
}

fn builtin_field_inv(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("field_inv expects 1 argument".to_string());
    }
    match &args[0] {
        Value::FieldElem(fe) => Ok(Value::FieldElem(fe.inv())),
        _ => Err("field_inv: argument must be a field element".to_string()),
    }
}

fn builtin_field_mul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("field_mul expects 2 arguments".to_string());
    }
    match (&args[0], &args[1]) {
        (Value::FieldElem(a), Value::FieldElem(b)) => Ok(Value::FieldElem(a.mul(b))),
        _ => Err("field_mul: arguments must be field elements".to_string()),
    }
}

fn builtin_field_add(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("field_add expects 2 arguments".to_string());
    }
    match (&args[0], &args[1]) {
        (Value::FieldElem(a), Value::FieldElem(b)) => Ok(Value::FieldElem(a.add(b))),
        _ => Err("field_add: arguments must be field elements".to_string()),
    }
}

fn builtin_field_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("field_new expects 2 arguments: (value_hex, modulus_hex) or (int, modulus_hex)".to_string());
    }
    let modulus = match &args[1] {
        Value::String(s) => crypto::BigUint256::from_hex(s)
            .ok_or_else(|| format!("invalid modulus hex: {}", s))?,
        Value::FieldElem(fe) => fe.modulus.clone(),
        _ => return Err("field_new: second argument must be a hex string or field element".to_string()),
    };
    let value = match &args[0] {
        Value::String(s) => crypto::BigUint256::from_hex(s)
            .ok_or_else(|| format!("invalid value hex: {}", s))?,
        Value::Int(n) => {
            let hex = format!("{:x}", n);
            crypto::BigUint256::from_hex(&hex)
                .ok_or_else(|| format!("cannot convert {} to bigint", n))?
        }
        Value::BigInt(n) => n.clone(),
        _ => return Err("field_new: first argument must be a hex string, integer, or bigint".to_string()),
    };
    Ok(Value::FieldElem(crypto::FieldElement::new(value, modulus)))
}

fn builtin_field_sub(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("field_sub expects 2 arguments".to_string());
    }
    match (&args[0], &args[1]) {
        (Value::FieldElem(a), Value::FieldElem(b)) => Ok(Value::FieldElem(a.sub(b))),
        _ => Err("field_sub: arguments must be field elements".to_string()),
    }
}

fn builtin_field_pow(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("field_pow expects 2 arguments: (field_elem, exponent)".to_string());
    }
    let fe = match &args[0] {
        Value::FieldElem(fe) => fe,
        _ => return Err("field_pow: first argument must be a field element".to_string()),
    };
    let exp = match &args[1] {
        Value::BigInt(n) => n.clone(),
        Value::Int(n) => {
            let hex = format!("{:x}", n);
            crypto::BigUint256::from_hex(&hex)
                .ok_or_else(|| format!("cannot convert {} to bigint", n))?
        }
        _ => return Err("field_pow: second argument must be a bigint or integer".to_string()),
    };
    Ok(Value::FieldElem(fe.pow(&exp)))
}

fn builtin_field_neg(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("field_neg expects 1 argument".to_string());
    }
    match &args[0] {
        Value::FieldElem(fe) => Ok(Value::FieldElem(fe.neg())),
        _ => Err("field_neg: argument must be a field element".to_string()),
    }
}

fn builtin_field_zero(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("field_zero expects 1 argument: modulus_hex".to_string());
    }
    let modulus = match &args[0] {
        Value::String(s) => crypto::BigUint256::from_hex(s)
            .ok_or_else(|| format!("invalid modulus hex: {}", s))?,
        Value::FieldElem(fe) => fe.modulus.clone(),
        _ => return Err("field_zero: argument must be a hex string or field element".to_string()),
    };
    Ok(Value::FieldElem(crypto::FieldElement::zero(modulus)))
}

fn builtin_field_one(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("field_one expects 1 argument: modulus_hex".to_string());
    }
    let modulus = match &args[0] {
        Value::String(s) => crypto::BigUint256::from_hex(s)
            .ok_or_else(|| format!("invalid modulus hex: {}", s))?,
        Value::FieldElem(fe) => fe.modulus.clone(),
        _ => return Err("field_one: argument must be a hex string or field element".to_string()),
    };
    Ok(Value::FieldElem(crypto::FieldElement::one(modulus)))
}

fn builtin_field_eq(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("field_eq expects 2 arguments".to_string());
    }
    match (&args[0], &args[1]) {
        (Value::FieldElem(a), Value::FieldElem(b)) => Ok(Value::Bool(a == b)),
        _ => Err("field_eq: arguments must be field elements".to_string()),
    }
}

fn builtin_field_prime(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("field_prime expects 1 argument: prime name (e.g., \"secp256k1\", \"bn254\", \"bls12_381\")".to_string());
    }
    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("field_prime: argument must be a string".to_string()),
    };
    let prime = match name.as_str() {
        "secp256k1" | "secp256k1_field" => crypto::secp256k1_field_prime(),
        "secp256k1_order" | "secp256k1_n" => crypto::secp256k1_order(),
        "bn254" | "bn254_field" => {
            // BN254 (alt_bn128) field prime: 21888242871839275222246405745257275088696311157297823662689037894645226208583
            crypto::BigUint256::from_hex("30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47").unwrap()
        }
        "bls12_381" | "bls12_381_field" => {
            // BLS12-381 field prime
            crypto::BigUint256::from_hex("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab").unwrap()
        }
        _ => return Err(format!("unknown prime name: {}. Options: secp256k1, secp256k1_order, bn254, bls12_381", name)),
    };
    Ok(Value::String(format!("0x{}", prime)))
}

// --- Scalar (mod N) builtins ---

fn builtin_scalar_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("scalar_new expects 1 argument: hex string or integer".to_string());
    }
    let value = match &args[0] {
        Value::String(s) => crypto::BigUint256::from_hex(s)
            .ok_or_else(|| format!("invalid hex: {}", s))?,
        Value::Int(n) => {
            let hex = format!("{:x}", n);
            crypto::BigUint256::from_hex(&hex)
                .ok_or_else(|| format!("cannot convert {} to bigint", n))?
        }
        Value::BigInt(n) => n.clone(),
        _ => return Err("scalar_new: argument must be a hex string or integer".to_string()),
    };
    let scalar = crypto::Scalar::new(value);
    // Store as FieldElem with curve order as modulus
    Ok(Value::FieldElem(crypto::FieldElement::new(scalar.value, crypto::secp256k1_order())))
}

fn builtin_scalar_inv(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("scalar_inv expects 1 argument".to_string()); }
    match &args[0] {
        Value::FieldElem(fe) => {
            let n = crypto::secp256k1_order();
            Ok(Value::FieldElem(crypto::FieldElement {
                value: crypto::mod_inv(&fe.value, &n),
                modulus: n,
            }))
        }
        _ => Err("scalar_inv: argument must be a scalar (field element mod n)".to_string()),
    }
}

fn builtin_scalar_neg(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("scalar_neg expects 1 argument".to_string()); }
    match &args[0] {
        Value::FieldElem(fe) => {
            let n = crypto::secp256k1_order();
            Ok(Value::FieldElem(crypto::FieldElement {
                value: crypto::mod_sub(&crypto::BigUint256::ZERO, &fe.value, &n),
                modulus: n,
            }))
        }
        _ => Err("scalar_neg: argument must be a scalar".to_string()),
    }
}

// --- Point operation builtins ---

fn builtin_point_negate(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("point_negate expects 1 argument".to_string()); }
    match &args[0] {
        Value::ECPoint(p) => Ok(Value::ECPoint(crypto::point_negate(p))),
        _ => Err("point_negate: argument must be an EC point".to_string()),
    }
}

fn builtin_point_sub(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("point_sub expects 2 arguments".to_string()); }
    match (&args[0], &args[1]) {
        (Value::ECPoint(p), Value::ECPoint(q)) => Ok(Value::ECPoint(crypto::point_sub(p, q))),
        _ => Err("point_sub: arguments must be EC points".to_string()),
    }
}

fn builtin_point_validate(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("point_validate expects 1 argument".to_string()); }
    match &args[0] {
        Value::ECPoint(p) => Ok(Value::Bool(crypto::validate_point(p))),
        _ => Err("point_validate: argument must be an EC point".to_string()),
    }
}

fn builtin_point_compress(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("point_compress expects 1 argument".to_string()); }
    match &args[0] {
        Value::ECPoint(p) => {
            let bytes = crypto::point_compress(p);
            let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
            Ok(Value::String(hex))
        }
        _ => Err("point_compress: argument must be an EC point".to_string()),
    }
}

fn builtin_point_decompress(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("point_decompress expects 1 argument (hex string)".to_string()); }
    match &args[0] {
        Value::String(hex) => {
            let bytes: Vec<u8> = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i+2], 16).unwrap_or(0))
                .collect();
            match crypto::point_decompress(&bytes) {
                Some(pt) => Ok(Value::ECPoint(pt)),
                None => Err("point_decompress: invalid compressed point".to_string()),
            }
        }
        _ => Err("point_decompress: argument must be a hex string".to_string()),
    }
}

// --- SHA-256 builtins ---

fn builtin_sha256(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("sha256 expects 1 argument".to_string()); }
    let data = match &args[0] {
        Value::String(s) => s.as_bytes().to_vec(),
        _ => return Err("sha256: argument must be a string".to_string()),
    };
    let hash = crypto::sha256(&data);
    let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
    Ok(Value::String(hex))
}

fn builtin_sha256d(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("sha256d expects 1 argument".to_string()); }
    let data = match &args[0] {
        Value::String(s) => s.as_bytes().to_vec(),
        _ => return Err("sha256d: argument must be a string".to_string()),
    };
    let hash = crypto::sha256d(&data);
    let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
    Ok(Value::String(hex))
}

// --- ECDSA builtins ---

fn builtin_ecdsa_sign(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("ecdsa_sign expects 2 arguments: (privkey_hex, message_string)".to_string());
    }
    let privkey = match &args[0] {
        Value::String(s) => crypto::BigUint256::from_hex(s)
            .ok_or_else(|| format!("invalid privkey hex: {}", s))?,
        Value::BigInt(n) => n.clone(),
        Value::FieldElem(fe) => fe.value.clone(),
        _ => return Err("ecdsa_sign: privkey must be a hex string or bigint".to_string()),
    };
    let message = match &args[1] {
        Value::String(s) => s.as_bytes().to_vec(),
        _ => return Err("ecdsa_sign: message must be a string".to_string()),
    };
    match crypto::ecdsa_sign_deterministic(&privkey, &message) {
        Some(sig) => {
            let mut fields = HashMap::new();
            fields.insert("r".to_string(), Value::String(format!("0x{}", sig.r)));
            fields.insert("s".to_string(), Value::String(format!("0x{}", sig.s)));
            Ok(Value::Struct { name: "ECDSASignature".to_string(), fields })
        }
        None => Err("ecdsa_sign: signing failed".to_string()),
    }
}

fn builtin_ecdsa_verify(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("ecdsa_verify expects 3 arguments: (pubkey, message, signature)".to_string());
    }
    let pubkey = match &args[0] {
        Value::ECPoint(p) => p.clone(),
        _ => return Err("ecdsa_verify: first argument must be an EC point".to_string()),
    };
    let message = match &args[1] {
        Value::String(s) => s.as_bytes().to_vec(),
        _ => return Err("ecdsa_verify: message must be a string".to_string()),
    };
    let sig = match &args[2] {
        Value::Struct { fields, .. } => {
            let r_str = match fields.get("r") {
                Some(Value::String(s)) => s.clone(),
                _ => return Err("ecdsa_verify: signature must have 'r' field".to_string()),
            };
            let s_str = match fields.get("s") {
                Some(Value::String(s)) => s.clone(),
                _ => return Err("ecdsa_verify: signature must have 's' field".to_string()),
            };
            crypto::ECDSASignature {
                r: crypto::BigUint256::from_hex(&r_str).ok_or("invalid r")?,
                s: crypto::BigUint256::from_hex(&s_str).ok_or("invalid s")?,
            }
        }
        _ => return Err("ecdsa_verify: signature must be a struct with r, s fields".to_string()),
    };
    let message_hash = crypto::sha256_to_bigint(&message);
    Ok(Value::Bool(crypto::ecdsa_verify(&pubkey, &message_hash, &sig)))
}

// --- Schnorr builtins ---

fn builtin_schnorr_sign(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("schnorr_sign expects 2 arguments: (privkey_hex, message_string)".to_string());
    }
    let privkey = match &args[0] {
        Value::String(s) => crypto::BigUint256::from_hex(s)
            .ok_or_else(|| format!("invalid privkey hex: {}", s))?,
        Value::BigInt(n) => n.clone(),
        Value::FieldElem(fe) => fe.value.clone(),
        _ => return Err("schnorr_sign: privkey must be a hex string or bigint".to_string()),
    };
    let message = match &args[1] {
        Value::String(s) => s.as_bytes().to_vec(),
        _ => return Err("schnorr_sign: message must be a string".to_string()),
    };
    match crypto::schnorr_sign(&privkey, &message) {
        Some(sig) => {
            let mut fields = HashMap::new();
            fields.insert("rx".to_string(), Value::String(format!("0x{}", sig.rx)));
            fields.insert("s".to_string(), Value::String(format!("0x{}", sig.s)));
            Ok(Value::Struct { name: "SchnorrSignature".to_string(), fields })
        }
        None => Err("schnorr_sign: signing failed".to_string()),
    }
}

fn builtin_schnorr_verify(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("schnorr_verify expects 3 arguments: (pubkey_x_hex, message, signature)".to_string());
    }
    let pubkey_x = match &args[0] {
        Value::String(s) => crypto::BigUint256::from_hex(s)
            .ok_or_else(|| format!("invalid pubkey hex: {}", s))?,
        Value::BigInt(n) => n.clone(),
        _ => return Err("schnorr_verify: pubkey_x must be a hex string".to_string()),
    };
    let message = match &args[1] {
        Value::String(s) => s.as_bytes().to_vec(),
        _ => return Err("schnorr_verify: message must be a string".to_string()),
    };
    let sig = match &args[2] {
        Value::Struct { fields, .. } => {
            let rx_str = match fields.get("rx") {
                Some(Value::String(s)) => s.clone(),
                _ => return Err("schnorr_verify: signature must have 'rx' field".to_string()),
            };
            let s_str = match fields.get("s") {
                Some(Value::String(s)) => s.clone(),
                _ => return Err("schnorr_verify: signature must have 's' field".to_string()),
            };
            crypto::SchnorrSignature {
                rx: crypto::BigUint256::from_hex(&rx_str).ok_or("invalid rx")?,
                s: crypto::BigUint256::from_hex(&s_str).ok_or("invalid s")?,
            }
        }
        _ => return Err("schnorr_verify: signature must be a struct with rx, s fields".to_string()),
    };
    Ok(Value::Bool(crypto::schnorr_verify(&pubkey_x, &message, &sig)))
}

fn builtin_point_x(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("point_x expects 1 argument".to_string()); }
    match &args[0] {
        Value::ECPoint(p) => {
            let (x, _) = crypto::point_to_affine(p);
            Ok(Value::BigInt(x))
        }
        _ => Err("point_x: argument must be an EC point".to_string()),
    }
}

fn builtin_point_y(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("point_y expects 1 argument".to_string()); }
    match &args[0] {
        Value::ECPoint(p) => {
            let (_, y) = crypto::point_to_affine(p);
            Ok(Value::BigInt(y))
        }
        _ => Err("point_y: argument must be an EC point".to_string()),
    }
}

fn builtin_bigint_from_hex(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("bigint_from_hex expects 1 argument".to_string()); }
    match &args[0] {
        Value::String(s) => {
            let n = crypto::BigUint256::from_hex(s)
                .ok_or_else(|| format!("invalid hex: {}", s))?;
            Ok(Value::BigInt(n))
        }
        _ => Err("bigint_from_hex: argument must be a string".to_string()),
    }
}

fn builtin_to_hex(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("to_hex expects 1 argument".to_string()); }
    match &args[0] {
        Value::BigInt(n) => Ok(Value::String(format!("0x{}", n))),
        Value::FieldElem(fe) => Ok(Value::String(format!("0x{}", fe.value))),
        Value::Int(n) => Ok(Value::String(format!("0x{:x}", n))),
        _ => Err("to_hex: argument must be a numeric type".to_string()),
    }
}

fn builtin_len(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("len expects 1 argument".to_string()); }
    match &args[0] {
        Value::Array(arr) => Ok(Value::Int(arr.len() as i128)),
        Value::String(s) => Ok(Value::Int(s.len() as i128)),
        _ => Err(format!("len: unsupported type {}", args[0])),
    }
}

fn builtin_to_string(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("to_string expects 1 argument".to_string()); }
    Ok(Value::String(format!("{}", args[0])))
}

fn builtin_format(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() { return Err("format expects at least 1 argument".to_string()); }
    let template = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("format: first argument must be a string".to_string()),
    };
    // Simple {} placeholder replacement
    let mut result = String::new();
    let mut arg_idx = 1;
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' && chars.peek() == Some(&'}') {
            chars.next(); // consume }
            if arg_idx < args.len() {
                result.push_str(&format!("{}", args[arg_idx]));
                arg_idx += 1;
            } else {
                result.push_str("{}");
            }
        } else {
            result.push(c);
        }
    }
    Ok(Value::String(result))
}

fn builtin_push(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("push expects 2 arguments (array, value)".to_string()); }
    match &args[0] {
        Value::Array(arr) => {
            let mut new_arr = arr.clone();
            new_arr.push(args[1].clone());
            Ok(Value::Array(new_arr))
        }
        _ => Err("push: first argument must be an array".to_string()),
    }
}

fn builtin_assert(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() { return Err("assert expects at least 1 argument".to_string()); }
    let cond = value_to_bool(&args[0])?;
    if !cond {
        let msg = if args.len() > 1 {
            format!("{}", args[1])
        } else {
            "assertion failed".to_string()
        };
        return Err(msg);
    }
    Ok(Value::Void)
}

fn builtin_assert_eq(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 { return Err("assert_eq expects 2 arguments".to_string()); }
    if !values_equal(&args[0], &args[1]) {
        return Err(format!("assertion failed: {} != {}", args[0], args[1]));
    }
    Ok(Value::Void)
}

// --- Interpreter entry point ---

pub fn interpret(program: &Program) -> Result<Vec<String>, String> {
    let mut env = Env::new();

    // First pass: register all functions and kernels
    for item in &program.items {
        match &item.kind {
            ItemKind::Function(func) => {
                let params: Vec<String> = func.params.iter().map(|p| p.name.name.clone()).collect();
                env.functions.insert(
                    func.name.name.clone(),
                    FnDef::User {
                        params,
                        body: func.body.clone(),
                    },
                );
            }
            ItemKind::Kernel(kernel) => {
                let params: Vec<String> = kernel.params.iter().map(|p| p.name.name.clone()).collect();
                env.functions.insert(
                    kernel.name.name.clone(),
                    FnDef::User {
                        params,
                        body: kernel.body.clone(),
                    },
                );
            }
            ItemKind::Const(c) => {
                let val = eval_expr(&mut env, &c.value)?;
                env.define(&c.name.name, val);
            }
            ItemKind::Struct(s) => {
                let field_names: Vec<String> = s.fields.iter().map(|f| f.name.name.clone()).collect();
                env.struct_defs.insert(s.name.name.clone(), field_names);
            }
            ItemKind::Enum(e) => {
                let enum_name = e.name.name.clone();
                for variant in &e.variants {
                    let vn = variant.name.name.clone();
                    let en = enum_name.clone();
                    match &variant.kind {
                        crate::ast::EnumVariantKind::Unit => {
                            let val = Value::EnumVariant {
                                enum_name: en.clone(),
                                variant: vn.clone(),
                                fields: Vec::new(),
                            };
                            env.define(&vn, val.clone());
                            env.define(&format!("{}::{}", en, vn), val);
                        }
                        crate::ast::EnumVariantKind::Tuple(types) => {
                            // Store arity info for constructor dispatch
                            env.struct_defs.insert(
                                format!("__enum_{}_{}", en, vn),
                                (0..types.len()).map(|i| format!("_{}", i)).collect(),
                            );
                        }
                        crate::ast::EnumVariantKind::Struct(_) => {}
                    }
                }
            }
            ItemKind::Impl(impl_block) => {
                // Get the target type name
                let type_name = match &impl_block.target.kind {
                    crate::ast::TypeExprKind::Named(id) => id.name.clone(),
                    _ => continue,
                };
                // Register each method as TypeName::method_name
                for method_item in &impl_block.methods {
                    if let ItemKind::Function(func) = &method_item.kind {
                        let params: Vec<String> = func.params.iter().map(|p| p.name.name.clone()).collect();
                        let qualified_name = format!("{}::{}", type_name, func.name.name);
                        env.functions.insert(
                            qualified_name,
                            FnDef::User {
                                params,
                                body: func.body.clone(),
                            },
                        );
                    }
                }
            }
            _ => {}
        }
    }

    // Find and call main()
    if let Some(func) = env.functions.get("main").cloned() {
        match func {
            FnDef::User { body, .. } => {
                eval_block(&mut env, &body)?;
            }
            _ => return Err("main must be a user function".to_string()),
        }
    } else {
        return Err("no main() function found".to_string());
    }

    Ok(env.output)
}

fn eval_block(env: &mut Env, block: &Block) -> Result<Value, String> {
    for stmt in &block.stmts {
        let val = eval_stmt(env, stmt)?;
        if let Value::Return(v) = val {
            return Ok(Value::Return(v));
        }
    }
    if let Some(expr) = &block.expr {
        eval_expr(env, expr)
    } else {
        Ok(Value::Void)
    }
}

fn eval_stmt(env: &mut Env, stmt: &Stmt) -> Result<Value, String> {
    match &stmt.kind {
        StmtKind::Let { name, value, .. } => {
            let val = eval_expr(env, value)?;
            env.define(&name.name, val);
            Ok(Value::Void)
        }
        StmtKind::Var { name, value, .. } => {
            let val = eval_expr(env, value)?;
            env.define(&name.name, val);
            Ok(Value::Void)
        }
        StmtKind::Return(Some(expr)) => {
            let val = eval_expr(env, expr)?;
            Ok(Value::Return(Box::new(val)))
        }
        StmtKind::Return(None) => Ok(Value::Return(Box::new(Value::Void))),
        StmtKind::Expr(expr) => eval_expr(env, expr),
        StmtKind::Assign { target, op, value } => {
            let rhs = eval_expr(env, value)?;
            let name = match &target.kind {
                ExprKind::Ident(id) => id.name.clone(),
                ExprKind::Index { base, indices } => {
                    // Array index assignment
                    if let ExprKind::Ident(id) = &base.kind {
                        let idx = eval_expr(env, &indices[0])?;
                        let idx = value_to_int(&idx)? as usize;
                        if let Some(Value::Array(mut arr)) = env.get(&id.name) {
                            let new_val = match op {
                                AssignOp::Assign => rhs,
                                AssignOp::AddAssign => value_add(&arr[idx], &rhs)?,
                                AssignOp::SubAssign => value_sub(&arr[idx], &rhs)?,
                                AssignOp::MulAssign => value_mul(&arr[idx], &rhs)?,
                                _ => return Err("unsupported assign op for array".to_string()),
                            };
                            arr[idx] = new_val;
                            env.set(&id.name, Value::Array(arr));
                            return Ok(Value::Void);
                        }
                    }
                    return Err("unsupported assignment target".to_string());
                }
                ExprKind::FieldAccess { base, field } => {
                    // Struct field assignment: s.field = value
                    if let ExprKind::Ident(id) = &base.kind {
                        if let Some(Value::Struct { name, mut fields }) = env.get(&id.name) {
                            let old = fields.get(&field.name).cloned().unwrap_or(Value::Void);
                            let new_val = match op {
                                AssignOp::Assign => rhs,
                                AssignOp::AddAssign => value_add(&old, &rhs)?,
                                AssignOp::SubAssign => value_sub(&old, &rhs)?,
                                AssignOp::MulAssign => value_mul(&old, &rhs)?,
                                _ => return Err("unsupported assign op for struct field".to_string()),
                            };
                            fields.insert(field.name.clone(), new_val);
                            env.set(&id.name, Value::Struct { name, fields });
                            return Ok(Value::Void);
                        }
                    }
                    return Err("unsupported field assignment target".to_string());
                }
                _ => return Err("unsupported assignment target".to_string()),
            };

            let current = env.get(&name).unwrap_or(Value::Int(0));
            let new_val = match op {
                AssignOp::Assign => rhs,
                AssignOp::AddAssign => value_add(&current, &rhs)?,
                AssignOp::SubAssign => value_sub(&current, &rhs)?,
                AssignOp::MulAssign => value_mul(&current, &rhs)?,
                AssignOp::DivAssign => value_div(&current, &rhs)?,
                AssignOp::MatMulAssign => return Err("@= not yet supported in interpreter".to_string()),
            };
            env.set(&name, new_val);
            Ok(Value::Void)
        }
        StmtKind::For { var, iter, body } => {
            let iter_val = eval_expr(env, iter)?;
            match iter_val {
                Value::Array(elems) => {
                    env.push_scope();
                    for elem in elems {
                        env.define(&var.name, elem);
                        let result = eval_block(env, body)?;
                        if let Value::Return(_) = &result {
                            env.pop_scope();
                            return Ok(result);
                        }
                    }
                    env.pop_scope();
                }
                _ => {
                    // Assume it's a range-like thing, try to extract start..end
                    return Err("for loop requires an iterable (array or range)".to_string());
                }
            }
            Ok(Value::Void)
        }
        StmtKind::While { cond, body } => {
            loop {
                let c = eval_expr(env, cond)?;
                if !value_to_bool(&c)? {
                    break;
                }
                env.push_scope();
                let result = eval_block(env, body)?;
                env.pop_scope();
                if let Value::Return(_) = &result {
                    return Ok(result);
                }
            }
            Ok(Value::Void)
        }
        StmtKind::Break => Err("break outside loop".to_string()),
        StmtKind::Continue => Err("continue outside loop".to_string()),
    }
}

fn eval_expr(env: &mut Env, expr: &Expr) -> Result<Value, String> {
    match &expr.kind {
        ExprKind::IntLiteral(n) => Ok(Value::Int(*n as i128)),
        ExprKind::FloatLiteral(n) => Ok(Value::Float(*n)),
        ExprKind::StringLiteral(s) => Ok(Value::String(s.clone())),
        ExprKind::BoolLiteral(b) => Ok(Value::Bool(*b)),

        ExprKind::Ident(id) => {
            env.get(&id.name).ok_or_else(|| format!("undefined variable: {}", id.name))
        }

        ExprKind::Binary { lhs, op, rhs } => {
            let l = eval_expr(env, lhs)?;
            let r = eval_expr(env, rhs)?;
            eval_binop(&l, *op, &r)
        }

        ExprKind::Unary { op, expr: inner } => {
            let v = eval_expr(env, inner)?;
            match op {
                UnaryOp::Neg => match v {
                    Value::Int(n) => Ok(Value::Int(-n)),
                    Value::Float(n) => Ok(Value::Float(-n)),
                    Value::FieldElem(fe) => Ok(Value::FieldElem(fe.neg())),
                    _ => Err("cannot negate this type".to_string()),
                },
                UnaryOp::Not => match v {
                    Value::Bool(b) => Ok(Value::Bool(!b)),
                    _ => Err("cannot apply ! to this type".to_string()),
                },
                UnaryOp::BitNot => match v {
                    Value::Int(n) => Ok(Value::Int(!n)),
                    _ => Err("cannot apply ~ to this type".to_string()),
                },
            }
        }

        ExprKind::Call { func, args } => {
            // Check for method call: base.method(args)
            if let ExprKind::FieldAccess { base, field } = &func.kind {
                let base_val = eval_expr(env, base)?;
                let method_name = &field.name;

                // Try to find Type::method in the function registry
                let type_name = match &base_val {
                    Value::Struct { name, .. } => Some(name.clone()),
                    _ => None,
                };

                if let Some(ref tn) = type_name {
                    let qualified = format!("{}::{}", tn, method_name);
                    if let Some(func_def) = env.functions.get(&qualified).cloned() {
                        let mut arg_vals = vec![base_val];
                        for a in args {
                            arg_vals.push(eval_expr(env, a)?);
                        }
                        return match func_def {
                            FnDef::Builtin(f) => f(env, arg_vals),
                            FnDef::User { params, body } => {
                                env.push_scope();
                                for (param, val) in params.iter().zip(arg_vals.iter()) {
                                    env.define(param, val.clone());
                                }
                                let result = eval_block(env, &body)?;
                                env.pop_scope();
                                match result {
                                    Value::Return(v) => Ok(*v),
                                    other => Ok(other),
                                }
                            }
                        };
                    }
                }

                // Not a method call on a struct — treat as a plain function name
                // (e.g., module.func style)
                let func_name = format!("{}", base).replace("(", "").replace(")", "") + "." + method_name;
                let arg_vals: Vec<Value> = args.iter()
                    .map(|a| eval_expr(env, a))
                    .collect::<Result<Vec<_>, _>>()?;

                let func_def = env.functions.get(&func_name).cloned()
                    .ok_or_else(|| format!("undefined function: {}", func_name))?;

                return match func_def {
                    FnDef::Builtin(f) => f(env, arg_vals),
                    FnDef::User { params, body } => {
                        env.push_scope();
                        for (param, val) in params.iter().zip(arg_vals.iter()) {
                            env.define(param, val.clone());
                        }
                        let result = eval_block(env, &body)?;
                        env.pop_scope();
                        match result {
                            Value::Return(v) => Ok(*v),
                            other => Ok(other),
                        }
                    }
                };
            }

            let arg_vals: Vec<Value> = args.iter()
                .map(|a| eval_expr(env, a))
                .collect::<Result<Vec<_>, _>>()?;

            // Get function name
            let func_name = match &func.kind {
                ExprKind::Ident(id) => id.name.clone(),
                _ => return Err("not a callable expression".to_string()),
            };

            // Check if this is an enum variant constructor
            for (key, _) in env.struct_defs.iter() {
                if key.starts_with("__enum_") && key.ends_with(&format!("_{}", func_name)) {
                    let parts: Vec<&str> = key.strip_prefix("__enum_").unwrap().rsplitn(2, '_').collect();
                    if parts.len() == 2 {
                        let enum_name = parts[1].to_string();
                        return Ok(Value::EnumVariant {
                            enum_name,
                            variant: func_name,
                            fields: arg_vals,
                        });
                    }
                }
            }

            let func_def = env.functions.get(&func_name).cloned()
                .ok_or_else(|| format!("undefined function: {}", func_name))?;

            match func_def {
                FnDef::Builtin(f) => f(env, arg_vals),
                FnDef::User { params, body } => {
                    env.push_scope();
                    for (param, val) in params.iter().zip(arg_vals.iter()) {
                        env.define(param, val.clone());
                    }
                    let result = eval_block(env, &body)?;
                    env.pop_scope();
                    match result {
                        Value::Return(v) => Ok(*v),
                        other => Ok(other),
                    }
                }
            }
        }

        ExprKind::MatMul { lhs, rhs } => {
            let _l = eval_expr(env, lhs)?;
            let _r = eval_expr(env, rhs)?;
            Err("matrix multiply not yet supported in interpreter".to_string())
        }

        ExprKind::FieldAccess { base, field } => {
            let base_val = eval_expr(env, base)?;
            match &base_val {
                Value::Array(arr) => match field.name.as_str() {
                    "len" => Ok(Value::Int(arr.len() as i128)),
                    _ => Err(format!("array has no field .{}", field.name)),
                },
                Value::String(s) => match field.name.as_str() {
                    "len" => Ok(Value::Int(s.len() as i128)),
                    _ => Err(format!("string has no field .{}", field.name)),
                },
                Value::Struct { fields, .. } => {
                    fields.get(&field.name).cloned()
                        .ok_or_else(|| format!("struct has no field .{}", field.name))
                }
                Value::Tuple(elems) => {
                    // Support .0, .1, etc. for tuples
                    if let Ok(idx) = field.name.parse::<usize>() {
                        elems.get(idx).cloned()
                            .ok_or_else(|| format!("tuple index {} out of bounds", idx))
                    } else {
                        Err(format!("tuple has no field .{}", field.name))
                    }
                }
                _ => Err(format!("cannot access field .{} on {}", field.name, base_val)),
            }
        }

        ExprKind::Index { base, indices } => {
            let base_val = eval_expr(env, base)?;
            let idx = eval_expr(env, &indices[0])?;
            let idx = value_to_int(&idx)? as usize;
            match base_val {
                Value::Array(arr) => {
                    arr.get(idx).cloned().ok_or_else(|| format!("index {} out of bounds (len {})", idx, arr.len()))
                }
                _ => Err("cannot index this type".to_string()),
            }
        }

        ExprKind::Block(block) => {
            env.push_scope();
            let result = eval_block(env, block)?;
            env.pop_scope();
            Ok(result)
        }

        ExprKind::If { cond, then_block, else_block } => {
            let c = eval_expr(env, cond)?;
            if value_to_bool(&c)? {
                env.push_scope();
                let r = eval_block(env, then_block)?;
                env.pop_scope();
                Ok(r)
            } else if let Some(else_block) = else_block {
                env.push_scope();
                let r = eval_block(env, else_block)?;
                env.pop_scope();
                Ok(r)
            } else {
                Ok(Value::Void)
            }
        }

        ExprKind::Range { start, end } => {
            let s = value_to_int(&eval_expr(env, start)?)?;
            let e = value_to_int(&eval_expr(env, end)?)?;
            let elems: Vec<Value> = (s..e).map(Value::Int).collect();
            Ok(Value::Array(elems))
        }

        ExprKind::ArrayLiteral(elems) => {
            let vals: Vec<Value> = elems.iter()
                .map(|e| eval_expr(env, e))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Value::Array(vals))
        }

        ExprKind::Cast { expr: inner, .. } => {
            // For now, just evaluate the expression
            eval_expr(env, inner)
        }

        ExprKind::TypeCall { .. } => {
            Err("type method calls not yet supported in interpreter".to_string())
        }

        ExprKind::StructLiteral { name, fields } => {
            let mut field_map = HashMap::new();
            for (fname, fexpr) in fields {
                let val = eval_expr(env, fexpr)?;
                field_map.insert(fname.name.clone(), val);
            }
            Ok(Value::Struct {
                name: name.name.clone(),
                fields: field_map,
            })
        }

        ExprKind::Match { expr: match_expr, arms } => {
            let val = eval_expr(env, match_expr)?;
            for arm in arms {
                if pattern_matches(&arm.pattern, &val) {
                    env.push_scope();
                    bind_pattern(env, &arm.pattern, &val);
                    let result = eval_expr(env, &arm.body)?;
                    env.pop_scope();
                    return Ok(result);
                }
            }
            Err(format!("no matching arm for value: {}", val))
        }
    }
}

fn pattern_matches(pattern: &Pattern, value: &Value) -> bool {
    match pattern {
        Pattern::Wildcard => true,
        Pattern::Ident(id) => {
            // If the value is an enum variant and the ident matches a variant name,
            // treat this as a variant match (not a binding)
            if let Value::EnumVariant { variant, fields, .. } = value {
                if id.name == *variant && fields.is_empty() {
                    return true; // Unit variant match
                }
                // If the ident doesn't match any variant name, it's a binding
                // (but we can't distinguish perfectly without type info, so
                // we check: if the ident starts with uppercase, treat as variant match)
                if id.name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    return id.name == *variant;
                }
            }
            true // Regular identifier binding
        }
        Pattern::Literal(expr) => {
            // Compare literal values
            match &expr.kind {
                ExprKind::IntLiteral(n) => {
                    if let Value::Int(v) = value { *v == *n as i128 } else { false }
                }
                ExprKind::FloatLiteral(n) => {
                    if let Value::Float(v) = value { *v == *n } else { false }
                }
                ExprKind::StringLiteral(s) => {
                    if let Value::String(v) = value { v == s } else { false }
                }
                ExprKind::BoolLiteral(b) => {
                    if let Value::Bool(v) = value { *v == *b } else { false }
                }
                _ => false,
            }
        }
        Pattern::Variant { name, fields } => {
            if let Value::EnumVariant { variant, fields: val_fields, .. } = value {
                if variant == &name.name {
                    // Match inner patterns
                    if fields.is_empty() {
                        return true;
                    }
                    if fields.len() != val_fields.len() {
                        return false;
                    }
                    return fields.iter().zip(val_fields.iter())
                        .all(|(pat, val)| pattern_matches(pat, val));
                }
            }
            // Also check against unit variant identifiers
            if let Value::EnumVariant { variant, fields: val_fields, .. } = value {
                if variant == &name.name && fields.is_empty() && val_fields.is_empty() {
                    return true;
                }
            }
            false
        }
    }
}

fn bind_pattern(env: &mut Env, pattern: &Pattern, value: &Value) {
    match pattern {
        Pattern::Ident(id) => {
            // Don't bind uppercase idents that are enum variant matches
            if let Value::EnumVariant { variant, .. } = value {
                if id.name == *variant {
                    return; // This was a variant match, not a binding
                }
            }
            env.define(&id.name, value.clone());
        }
        Pattern::Variant { name: _, fields } => {
            if let Value::EnumVariant { fields: val_fields, .. } = value {
                for (pat, val) in fields.iter().zip(val_fields.iter()) {
                    bind_pattern(env, pat, val);
                }
            }
        }
        Pattern::Wildcard | Pattern::Literal(_) => {}
    }
}

fn eval_binop(lhs: &Value, op: BinOp, rhs: &Value) -> Result<Value, String> {
    match op {
        BinOp::Add => value_add(lhs, rhs),
        BinOp::Sub => value_sub(lhs, rhs),
        BinOp::Mul => value_mul(lhs, rhs),
        BinOp::Div => value_div(lhs, rhs),
        BinOp::Mod => {
            match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a % b)),
                _ => Err("% requires integers".to_string()),
            }
        }
        BinOp::Pow => {
            match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.pow(*b as u32))),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a.powf(*b))),
                (Value::FieldElem(a), Value::Int(b)) => {
                    let exp_hex = format!("{:x}", b);
                    let exp = crypto::BigUint256::from_hex(&exp_hex).unwrap_or(crypto::BigUint256::ZERO);
                    Ok(Value::FieldElem(a.pow(&exp)))
                }
                _ => Err("** requires numeric types".to_string()),
            }
        }
        BinOp::Eq => Ok(Value::Bool(values_equal(lhs, rhs))),
        BinOp::NotEq => Ok(Value::Bool(!values_equal(lhs, rhs))),
        BinOp::Lt => value_cmp(lhs, rhs, |a, b| a < b, |a, b| a < b),
        BinOp::Gt => value_cmp(lhs, rhs, |a, b| a > b, |a, b| a > b),
        BinOp::LtEq => value_cmp(lhs, rhs, |a, b| a <= b, |a, b| a <= b),
        BinOp::GtEq => value_cmp(lhs, rhs, |a, b| a >= b, |a, b| a >= b),
        BinOp::And => {
            match (lhs, rhs) {
                (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a && *b)),
                _ => Err("&& requires booleans".to_string()),
            }
        }
        BinOp::Or => {
            match (lhs, rhs) {
                (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a || *b)),
                _ => Err("|| requires booleans".to_string()),
            }
        }
        BinOp::BitAnd => {
            match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a & b)),
                _ => Err("& requires integers".to_string()),
            }
        }
        BinOp::BitOr => {
            match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a | b)),
                _ => Err("| requires integers".to_string()),
            }
        }
        BinOp::BitXor => {
            match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a ^ b)),
                _ => Err("^ requires integers".to_string()),
            }
        }
        BinOp::Shl => {
            match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a << (*b as u32))),
                _ => Err("<< requires integers".to_string()),
            }
        }
        BinOp::Shr => {
            match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a >> (*b as u32))),
                _ => Err(">> requires integers".to_string()),
            }
        }
        BinOp::ElemMul | BinOp::ElemDiv => {
            Err("elementwise ops not yet supported in interpreter".to_string())
        }
    }
}

fn value_add(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x + y)),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x + y)),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Float(*x as f64 + y)),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x + *y as f64)),
        (Value::String(x), Value::String(y)) => Ok(Value::String(format!("{}{}", x, y))),
        (Value::FieldElem(x), Value::FieldElem(y)) => Ok(Value::FieldElem(x.add(y))),
        _ => Err(format!("cannot add {} and {}", a, b)),
    }
}

fn value_sub(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x - y)),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x - y)),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Float(*x as f64 - y)),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x - *y as f64)),
        (Value::FieldElem(x), Value::FieldElem(y)) => Ok(Value::FieldElem(x.sub(y))),
        _ => Err(format!("cannot subtract {} and {}", a, b)),
    }
}

fn value_mul(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x * y)),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x * y)),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Float(*x as f64 * y)),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x * *y as f64)),
        (Value::FieldElem(x), Value::FieldElem(y)) => Ok(Value::FieldElem(x.mul(y))),
        _ => Err(format!("cannot multiply {} and {}", a, b)),
    }
}

fn value_div(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => {
            if *y == 0 { return Err("division by zero".to_string()); }
            Ok(Value::Int(x / y))
        }
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x / y)),
        (Value::FieldElem(x), Value::FieldElem(y)) => {
            if y.is_zero() { return Err("division by zero in field".to_string()); }
            Ok(Value::FieldElem(x.mul(&y.inv())))
        }
        _ => Err(format!("cannot divide {} and {}", a, b)),
    }
}

fn value_to_int(v: &Value) -> Result<i128, String> {
    match v {
        Value::Int(n) => Ok(*n),
        Value::Float(n) => Ok(*n as i128),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        _ => Err(format!("cannot convert {} to integer", v)),
    }
}

fn value_to_bool(v: &Value) -> Result<bool, String> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::Int(n) => Ok(*n != 0),
        _ => Err(format!("cannot convert {} to boolean", v)),
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::FieldElem(x), Value::FieldElem(y)) => x == y,
        _ => false,
    }
}

fn value_cmp(
    a: &Value,
    b: &Value,
    int_op: fn(i128, i128) -> bool,
    float_op: fn(f64, f64) -> bool,
) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Bool(int_op(*x, *y))),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Bool(float_op(*x, *y))),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Bool(float_op(*x as f64, *y))),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Bool(float_op(*x, *y as f64))),
        _ => Err(format!("cannot compare {} and {}", a, b)),
    }
}

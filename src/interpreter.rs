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
            let arg_vals: Vec<Value> = args.iter()
                .map(|a| eval_expr(env, a))
                .collect::<Result<Vec<_>, _>>()?;

            // Get function name
            let func_name = match &func.kind {
                ExprKind::Ident(id) => id.name.clone(),
                ExprKind::FieldAccess { base, field } => {
                    // Handle module.function calls like crypto.foo
                    format!("{}", base).replace("(", "").replace(")", "") + "." + &field.name
                }
                _ => return Err("not a callable expression".to_string()),
            };

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
        Pattern::Ident(_) => true, // Ident always matches (it's a binding)
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
        Pattern::Variant { .. } => {
            // Enum variants not fully implemented yet
            false
        }
    }
}

fn bind_pattern(env: &mut Env, pattern: &Pattern, value: &Value) {
    match pattern {
        Pattern::Ident(id) => {
            env.define(&id.name, value.clone());
        }
        Pattern::Wildcard | Pattern::Literal(_) | Pattern::Variant { .. } => {}
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
        _ => Err(format!("cannot add {} and {}", a, b)),
    }
}

fn value_sub(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x - y)),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x - y)),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Float(*x as f64 - y)),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x - *y as f64)),
        _ => Err(format!("cannot subtract {} and {}", a, b)),
    }
}

fn value_mul(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x * y)),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x * y)),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Float(*x as f64 * y)),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x * *y as f64)),
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

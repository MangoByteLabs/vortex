use crate::lexer::Span;
use std::fmt;

/// A complete Vortex program (one source file)
#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
}

/// Top-level item in a program
#[derive(Debug, Clone)]
pub struct Item {
    pub kind: ItemKind,
    pub span: Span,
    pub is_pub: bool,
}

#[derive(Debug, Clone)]
pub enum ItemKind {
    Function(Function),
    Kernel(Kernel),
    Struct(StructDef),
    Enum(EnumDef),
    Trait(TraitDef),
    Impl(ImplBlock),
    Import(ImportDecl),
    Const(ConstDecl),
    TypeAlias(TypeAlias),
}

/// A regular function
#[derive(Debug, Clone)]
pub struct Function {
    pub name: Ident,
    pub generics: Vec<GenericParam>,
    pub params: Vec<Param>,
    pub ret_type: Option<TypeExpr>,
    pub where_clause: Vec<WhereClause>,
    pub body: Block,
    /// Annotations like @scan(...), @recurrent
    pub annotations: Vec<Annotation>,
}

/// A GPU kernel function
#[derive(Debug, Clone)]
pub struct Kernel {
    pub name: Ident,
    pub schedule: Option<ScheduleAnnotation>,
    pub generics: Vec<GenericParam>,
    pub params: Vec<Param>,
    pub ret_type: Option<TypeExpr>,
    pub where_clause: Vec<WhereClause>,
    pub body: Block,
    /// Annotations like @scan(...), @recurrent
    pub annotations: Vec<Annotation>,
}

/// Schedule annotation like @schedule(tile_m=128, num_warps=4)
#[derive(Debug, Clone)]
pub struct ScheduleAnnotation {
    pub params: Vec<(Ident, Expr)>,
    pub span: Span,
}

/// An annotation on a function or kernel (e.g., @scan, @recurrent, @schedule)
#[derive(Debug, Clone)]
pub enum Annotation {
    /// @schedule(tile_m=128, num_warps=4)
    Schedule(ScheduleAnnotation),
    /// @scan(mode = "parallel") or @scan with no params
    Scan(Option<Vec<(Ident, Expr)>>),
    /// @recurrent - marks for sequential recurrent execution
    Recurrent,
}

/// Struct definition
#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: Ident,
    pub generics: Vec<GenericParam>,
    pub fields: Vec<Field>,
}

/// A struct field
#[derive(Debug, Clone)]
pub struct Field {
    pub name: Ident,
    pub ty: TypeExpr,
    pub default: Option<Expr>,
    pub is_pub: bool,
    pub span: Span,
}

/// Enum definition
#[derive(Debug, Clone)]
pub struct EnumDef {
    pub name: Ident,
    pub generics: Vec<GenericParam>,
    pub variants: Vec<EnumVariant>,
}

/// An enum variant
#[derive(Debug, Clone)]
pub struct EnumVariant {
    pub name: Ident,
    pub kind: EnumVariantKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum EnumVariantKind {
    /// Unit variant: None
    Unit,
    /// Tuple variant: Some(T)
    Tuple(Vec<TypeExpr>),
    /// Struct variant: Point { x: f64, y: f64 }
    Struct(Vec<Field>),
}

/// Trait definition
#[derive(Debug, Clone)]
pub struct TraitDef {
    pub name: Ident,
    pub supertraits: Vec<TypeExpr>,
    pub methods: Vec<TraitMethod>,
}

/// A method in a trait definition
#[derive(Debug, Clone)]
pub struct TraitMethod {
    pub name: Ident,
    pub params: Vec<Param>,
    pub ret_type: Option<TypeExpr>,
    pub body: Option<Block>,
    pub span: Span,
}

/// Impl block
#[derive(Debug, Clone)]
pub struct ImplBlock {
    pub trait_name: Option<TypeExpr>,
    pub target: TypeExpr,
    pub methods: Vec<Item>,
}

/// Import declaration
#[derive(Debug, Clone)]
pub struct ImportDecl {
    pub path: Vec<Ident>,
    pub items: ImportItems,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ImportItems {
    /// import foo.bar { A, B, C }
    Named(Vec<ImportItem>),
    /// import foo.bar { * }
    Wildcard,
}

#[derive(Debug, Clone)]
pub struct ImportItem {
    pub name: Ident,
    pub alias: Option<Ident>,
}

/// Constant declaration
#[derive(Debug, Clone)]
pub struct ConstDecl {
    pub name: Ident,
    pub ty: Option<TypeExpr>,
    pub value: Expr,
}

/// Type alias
#[derive(Debug, Clone)]
pub struct TypeAlias {
    pub name: Ident,
    pub generics: Vec<GenericParam>,
    pub value: TypeExpr,
}

// --- Generics ---

#[derive(Debug, Clone)]
pub struct GenericParam {
    pub name: Ident,
    pub kind: GenericParamKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum GenericParamKind {
    /// Type parameter: T
    Type { bounds: Vec<TypeExpr> },
    /// Const parameter: const N: usize
    Const { ty: TypeExpr },
}

#[derive(Debug, Clone)]
pub struct WhereClause {
    pub ty: TypeExpr,
    pub bounds: Vec<TypeExpr>,
    pub span: Span,
}

// --- Parameters ---

#[derive(Debug, Clone)]
pub struct Param {
    pub name: Ident,
    pub ty: TypeExpr,
    pub default: Option<Expr>,
    pub span: Span,
}

// --- Statements ---

#[derive(Debug, Clone)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum StmtKind {
    /// let x: T = expr
    Let {
        name: Ident,
        ty: Option<TypeExpr>,
        value: Expr,
    },
    /// var x: T = expr (mutable binding)
    Var {
        name: Ident,
        ty: Option<TypeExpr>,
        value: Expr,
    },
    /// return expr
    Return(Option<Expr>),
    /// expr (as statement, e.g. function call)
    Expr(Expr),
    /// Assignment: lhs = rhs (or +=, -=, etc.)
    Assign {
        target: Expr,
        op: AssignOp,
        value: Expr,
    },
    /// for i in start..end { body }
    For {
        var: Ident,
        iter: Expr,
        body: Block,
    },
    /// while cond { body }
    While {
        cond: Expr,
        body: Block,
    },
    /// break
    Break,
    /// continue
    Continue,
    /// dispatch index -> [target1, target2, ...](args)
    Dispatch {
        index: Box<Expr>,
        targets: Vec<Ident>,
        args: Vec<Expr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignOp {
    Assign,   // =
    AddAssign, // +=
    SubAssign, // -=
    MulAssign, // *=
    DivAssign, // /=
    MatMulAssign, // @=
}

// --- Expressions ---

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    /// Integer literal: 42, 0xFF
    IntLiteral(u128),
    /// Float literal: 3.14
    FloatLiteral(f64),
    /// String literal: "hello"
    StringLiteral(String),
    /// Boolean literal: true, false
    BoolLiteral(bool),
    /// Identifier: x, foo
    Ident(Ident),
    /// Binary operation: a + b
    Binary {
        lhs: Box<Expr>,
        op: BinOp,
        rhs: Box<Expr>,
    },
    /// Unary operation: -x, !b
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    /// Function/method call: foo(a, b)
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
    },
    /// Matrix multiply: a @ b
    MatMul {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// Field/method access: a.b
    FieldAccess {
        base: Box<Expr>,
        field: Ident,
    },
    /// Indexing: a[i] or a[i, j]
    Index {
        base: Box<Expr>,
        indices: Vec<Expr>,
    },
    /// Block expression: { stmts; expr }
    Block(Block),
    /// If expression: if cond { then } else { else_ }
    If {
        cond: Box<Expr>,
        then_block: Block,
        else_block: Option<Block>,
    },
    /// Range: start..end
    Range {
        start: Box<Expr>,
        end: Box<Expr>,
    },
    /// Array/tensor literal: [1, 2, 3]
    ArrayLiteral(Vec<Expr>),
    /// Type instantiation with generics: Tensor.zeros<f32, [N]>()
    TypeCall {
        ty: TypeExpr,
        method: Ident,
        args: Vec<Expr>,
    },
    /// Cast: x as f32
    Cast {
        expr: Box<Expr>,
        ty: TypeExpr,
    },
    /// Struct literal: Point { x: 1.0, y: 2.0 }
    StructLiteral {
        name: Ident,
        fields: Vec<(Ident, Expr)>,
    },
    /// Match expression
    Match {
        expr: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    /// Closure/lambda: |x, y| x + y
    Closure {
        params: Vec<Param>,
        body: Box<Expr>,
    },
    /// Try operator: expr?
    Try(Box<Expr>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,    // +
    Sub,    // -
    Mul,    // *
    Div,    // /
    Mod,    // %
    Pow,    // **
    Eq,     // ==
    NotEq,  // !=
    Lt,     // <
    Gt,     // >
    LtEq,   // <=
    GtEq,   // >=
    And,    // &&
    Or,     // ||
    BitAnd, // &
    BitOr,  // |
    BitXor, // ^
    Shl,    // <<
    Shr,    // >>
    ElemMul, // .*
    ElemDiv, // ./
}

impl BinOp {
    pub fn precedence(self) -> u8 {
        match self {
            BinOp::Or => 1,
            BinOp::And => 2,
            BinOp::BitOr => 3,
            BinOp::BitXor => 4,
            BinOp::BitAnd => 5,
            BinOp::Eq | BinOp::NotEq => 6,
            BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq => 7,
            BinOp::Shl | BinOp::Shr => 8,
            BinOp::Add | BinOp::Sub => 9,
            BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::ElemMul | BinOp::ElemDiv => 10,
            BinOp::Pow => 11,
        }
    }

    pub fn is_right_assoc(self) -> bool {
        matches!(self, BinOp::Pow)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,    // -
    Not,    // !
    BitNot, // ~
}

// --- Types ---

#[derive(Debug, Clone)]
pub struct TypeExpr {
    pub kind: TypeExprKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum TypeExprKind {
    /// Named type: f32, u64, bool, MyStruct
    Named(Ident),
    /// Generic type: Tensor<f32, [N, M]>
    Generic {
        name: Ident,
        args: Vec<TypeArg>,
    },
    /// Array type: [f32; 256]
    Array {
        elem: Box<TypeExpr>,
        size: Box<Expr>,
    },
    /// Reference type: &T or &mut T
    Ref {
        mutable: bool,
        inner: Box<TypeExpr>,
    },
    /// Tuple type: (f32, u32)
    Tuple(Vec<TypeExpr>),
    /// Shape literal used in Tensor<f32, [M, N]>
    Shape(Vec<ShapeDim>),
    /// Function type: fn(T, U) -> V
    Fn {
        params: Vec<TypeExpr>,
        ret: Box<TypeExpr>,
    },
    /// Sparse type: Sparse<T>
    Sparse(Box<TypeExpr>),
    /// SparseIndex type: SparseIndex[B, k]
    SparseIndex { batch: Box<Expr>, k: Box<Expr> },
}

/// A type argument can be a type or a value (for const generics / shapes)
#[derive(Debug, Clone)]
pub enum TypeArg {
    Type(TypeExpr),
    Expr(Expr),
    Shape(Vec<ShapeDim>),
}

/// A dimension in a shape
#[derive(Debug, Clone)]
pub enum ShapeDim {
    /// Concrete size: 4096
    Lit(u64),
    /// Symbolic size: N, M
    Ident(Ident),
    /// Computed size: N * 2, 3 * D
    Expr(Expr),
    /// Dynamic/unknown: ?
    Dynamic,
}

/// A match arm: pattern => body
#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Pattern {
    /// Wildcard: _
    Wildcard,
    /// Literal: 42, "hello", true
    Literal(Expr),
    /// Identifier binding: x
    Ident(Ident),
    /// Enum variant: Some(x), None
    Variant { name: Ident, fields: Vec<Pattern> },
}

// --- Block ---

#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    /// The trailing expression (implicit return)
    pub expr: Option<Box<Expr>>,
    pub span: Span,
}

// --- Identifier ---

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

impl Ident {
    pub fn new(name: String, span: Span) -> Self {
        Self { name, span }
    }
}

// --- Display implementations for pretty printing ---

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, item) in self.items.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{}", item)?;
        }
        Ok(())
    }
}

impl fmt::Display for Item {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_pub {
            write!(f, "pub ")?;
        }
        match &self.kind {
            ItemKind::Function(func) => write!(f, "{}", func),
            ItemKind::Kernel(kernel) => write!(f, "{}", kernel),
            ItemKind::Struct(s) => write!(f, "{}", s),
            ItemKind::Enum(e) => write!(f, "{}", e),
            ItemKind::Trait(t) => write!(f, "{}", t),
            ItemKind::Impl(i) => write!(f, "{}", i),
            ItemKind::Import(i) => write!(f, "{}", i),
            ItemKind::Const(c) => write!(f, "{}", c),
            ItemKind::TypeAlias(t) => write!(f, "{}", t),
        }
    }
}

impl fmt::Display for Annotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Annotation::Schedule(sched) => write!(f, "{}", sched),
            Annotation::Scan(Some(params)) => {
                write!(f, "@scan(")?;
                for (i, (name, val)) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{} = {}", name.name, val)?;
                }
                write!(f, ")")
            }
            Annotation::Scan(None) => write!(f, "@scan"),
            Annotation::Recurrent => write!(f, "@recurrent"),
        }
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for ann in &self.annotations {
            writeln!(f, "{}", ann)?;
        }
        write!(f, "fn {}", self.name.name)?;
        if !self.generics.is_empty() {
            write!(f, "<")?;
            for (i, g) in self.generics.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", g)?;
            }
            write!(f, ">")?;
        }
        write!(f, "(")?;
        for (i, p) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", p)?;
        }
        write!(f, ")")?;
        if let Some(ret) = &self.ret_type {
            write!(f, " -> {}", ret)?;
        }
        write!(f, " {}", self.body)
    }
}

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(sched) = &self.schedule {
            writeln!(f, "{}", sched)?;
        }
        write!(f, "kernel {}", self.name.name)?;
        if !self.generics.is_empty() {
            write!(f, "<")?;
            for (i, g) in self.generics.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", g)?;
            }
            write!(f, ">")?;
        }
        write!(f, "(")?;
        for (i, p) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", p)?;
        }
        write!(f, ")")?;
        if let Some(ret) = &self.ret_type {
            write!(f, " -> {}", ret)?;
        }
        write!(f, " {}", self.body)
    }
}

impl fmt::Display for StructDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "struct {}", self.name.name)?;
        if !self.generics.is_empty() {
            write!(f, "<")?;
            for (i, g) in self.generics.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", g)?;
            }
            write!(f, ">")?;
        }
        writeln!(f, " {{")?;
        for field in &self.fields {
            if field.is_pub {
                write!(f, "    pub ")?;
            } else {
                write!(f, "    ")?;
            }
            write!(f, "{}: {}", field.name.name, field.ty)?;
            if let Some(default) = &field.default {
                write!(f, " = {}", default)?;
            }
            writeln!(f, ",")?;
        }
        write!(f, "}}")
    }
}

impl fmt::Display for EnumDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "enum {}", self.name.name)?;
        if !self.generics.is_empty() {
            write!(f, "<")?;
            for (i, g) in self.generics.iter().enumerate() {
                if i > 0 { write!(f, ", ")?; }
                write!(f, "{}", g)?;
            }
            write!(f, ">")?;
        }
        writeln!(f, " {{")?;
        for v in &self.variants {
            write!(f, "    {}", v.name.name)?;
            match &v.kind {
                EnumVariantKind::Unit => {}
                EnumVariantKind::Tuple(types) => {
                    write!(f, "(")?;
                    for (i, t) in types.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}", t)?;
                    }
                    write!(f, ")")?;
                }
                EnumVariantKind::Struct(fields) => {
                    write!(f, " {{ ")?;
                    for (i, field) in fields.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}: {}", field.name.name, field.ty)?;
                    }
                    write!(f, " }}")?;
                }
            }
            writeln!(f, ",")?;
        }
        write!(f, "}}")
    }
}

impl fmt::Display for TraitDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "trait {}", self.name.name)?;
        if !self.supertraits.is_empty() {
            write!(f, ": ")?;
            for (i, s) in self.supertraits.iter().enumerate() {
                if i > 0 {
                    write!(f, " + ")?;
                }
                write!(f, "{}", s)?;
            }
        }
        writeln!(f, " {{")?;
        for method in &self.methods {
            write!(f, "    fn {}(", method.name.name)?;
            for (i, p) in method.params.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", p)?;
            }
            write!(f, ")")?;
            if let Some(ret) = &method.ret_type {
                write!(f, " -> {}", ret)?;
            }
            writeln!(f)?;
        }
        write!(f, "}}")
    }
}

impl fmt::Display for ImplBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "impl ")?;
        if let Some(trait_name) = &self.trait_name {
            write!(f, "{} for ", trait_name)?;
        }
        writeln!(f, "{} {{", self.target)?;
        for method in &self.methods {
            write!(f, "    {}", method)?;
            writeln!(f)?;
        }
        write!(f, "}}")
    }
}

impl fmt::Display for ImportDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "import ")?;
        for (i, p) in self.path.iter().enumerate() {
            if i > 0 {
                write!(f, ".")?;
            }
            write!(f, "{}", p.name)?;
        }
        match &self.items {
            ImportItems::Named(items) => {
                write!(f, " {{ ")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item.name.name)?;
                    if let Some(alias) = &item.alias {
                        write!(f, " as {}", alias.name)?;
                    }
                }
                write!(f, " }}")
            }
            ImportItems::Wildcard => write!(f, " {{ * }}"),
        }
    }
}

impl fmt::Display for ConstDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "const {}", self.name.name)?;
        if let Some(ty) = &self.ty {
            write!(f, ": {}", ty)?;
        }
        write!(f, " = {}", self.value)
    }
}

impl fmt::Display for TypeAlias {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "type {}", self.name.name)?;
        if !self.generics.is_empty() {
            write!(f, "<")?;
            for (i, g) in self.generics.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", g)?;
            }
            write!(f, ">")?;
        }
        write!(f, " = {}", self.value)
    }
}

impl fmt::Display for GenericParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            GenericParamKind::Type { bounds } => {
                write!(f, "{}", self.name.name)?;
                if !bounds.is_empty() {
                    write!(f, ": ")?;
                    for (i, b) in bounds.iter().enumerate() {
                        if i > 0 {
                            write!(f, " + ")?;
                        }
                        write!(f, "{}", b)?;
                    }
                }
                Ok(())
            }
            GenericParamKind::Const { ty } => {
                write!(f, "const {}: {}", self.name.name, ty)
            }
        }
    }
}

impl fmt::Display for Param {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name.name, self.ty)?;
        if let Some(default) = &self.default {
            write!(f, " = {}", default)?;
        }
        Ok(())
    }
}

impl fmt::Display for ScheduleAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@schedule(")?;
        for (i, (name, val)) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{} = {}", name.name, val)?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            StmtKind::Let { name, ty, value } => {
                write!(f, "let {}", name.name)?;
                if let Some(ty) = ty {
                    write!(f, ": {}", ty)?;
                }
                write!(f, " = {}", value)
            }
            StmtKind::Var { name, ty, value } => {
                write!(f, "var {}", name.name)?;
                if let Some(ty) = ty {
                    write!(f, ": {}", ty)?;
                }
                write!(f, " = {}", value)
            }
            StmtKind::Return(Some(expr)) => write!(f, "return {}", expr),
            StmtKind::Return(None) => write!(f, "return"),
            StmtKind::Expr(expr) => write!(f, "{}", expr),
            StmtKind::Assign { target, op, value } => {
                write!(f, "{} {} {}", target, op, value)
            }
            StmtKind::For { var, iter, body } => {
                write!(f, "for {} in {} {}", var.name, iter, body)
            }
            StmtKind::While { cond, body } => {
                write!(f, "while {} {}", cond, body)
            }
            StmtKind::Break => write!(f, "break"),
            StmtKind::Continue => write!(f, "continue"),
            StmtKind::Dispatch { index, targets, args } => {
                write!(f, "dispatch {} -> [", index)?;
                for (i, t) in targets.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", t.name)?;
                }
                write!(f, "](")?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", a)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for AssignOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssignOp::Assign => write!(f, "="),
            AssignOp::AddAssign => write!(f, "+="),
            AssignOp::SubAssign => write!(f, "-="),
            AssignOp::MulAssign => write!(f, "*="),
            AssignOp::DivAssign => write!(f, "/="),
            AssignOp::MatMulAssign => write!(f, "@="),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ExprKind::IntLiteral(n) => write!(f, "{}", n),
            ExprKind::FloatLiteral(n) => write!(f, "{}", n),
            ExprKind::StringLiteral(s) => write!(f, "\"{}\"", s),
            ExprKind::BoolLiteral(b) => write!(f, "{}", b),
            ExprKind::Ident(id) => write!(f, "{}", id.name),
            ExprKind::Binary { lhs, op, rhs } => {
                write!(f, "({} {} {})", lhs, op, rhs)
            }
            ExprKind::Unary { op, expr } => write!(f, "({}{})", op, expr),
            ExprKind::Call { func, args } => {
                write!(f, "{}(", func)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            ExprKind::MatMul { lhs, rhs } => {
                write!(f, "({} @ {})", lhs, rhs)
            }
            ExprKind::FieldAccess { base, field } => {
                write!(f, "{}.{}", base, field.name)
            }
            ExprKind::Index { base, indices } => {
                write!(f, "{}[", base)?;
                for (i, idx) in indices.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", idx)?;
                }
                write!(f, "]")
            }
            ExprKind::Block(block) => write!(f, "{}", block),
            ExprKind::If {
                cond,
                then_block,
                else_block,
            } => {
                write!(f, "if {} {}", cond, then_block)?;
                if let Some(else_block) = else_block {
                    write!(f, " else {}", else_block)?;
                }
                Ok(())
            }
            ExprKind::Range { start, end } => write!(f, "{}..{}", start, end),
            ExprKind::ArrayLiteral(elems) => {
                write!(f, "[")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", e)?;
                }
                write!(f, "]")
            }
            ExprKind::TypeCall { ty, method, args } => {
                write!(f, "{}.{}(", ty, method.name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            ExprKind::Cast { expr, ty } => write!(f, "({} as {})", expr, ty),
            ExprKind::StructLiteral { name, fields } => {
                write!(f, "{} {{ ", name.name)?;
                for (i, (fname, fval)) in fields.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {}", fname.name, fval)?;
                }
                write!(f, " }}")
            }
            ExprKind::Match { expr, arms } => {
                writeln!(f, "match {} {{", expr)?;
                for arm in arms {
                    writeln!(f, "    {} => {},", arm.pattern, arm.body)?;
                }
                write!(f, "}}")
            }
            ExprKind::Closure { params, body } => {
                write!(f, "|")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", p.name.name)?;
                }
                write!(f, "| {}", body)
            }
            ExprKind::Try(inner) => write!(f, "{}?", inner),
        }
    }
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Mod => write!(f, "%"),
            BinOp::Pow => write!(f, "**"),
            BinOp::Eq => write!(f, "=="),
            BinOp::NotEq => write!(f, "!="),
            BinOp::Lt => write!(f, "<"),
            BinOp::Gt => write!(f, ">"),
            BinOp::LtEq => write!(f, "<="),
            BinOp::GtEq => write!(f, ">="),
            BinOp::And => write!(f, "&&"),
            BinOp::Or => write!(f, "||"),
            BinOp::BitAnd => write!(f, "&"),
            BinOp::BitOr => write!(f, "|"),
            BinOp::BitXor => write!(f, "^"),
            BinOp::Shl => write!(f, "<<"),
            BinOp::Shr => write!(f, ">>"),
            BinOp::ElemMul => write!(f, ".*"),
            BinOp::ElemDiv => write!(f, "./"),
        }
    }
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Not => write!(f, "!"),
            UnaryOp::BitNot => write!(f, "~"),
        }
    }
}

impl fmt::Display for TypeExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            TypeExprKind::Named(name) => write!(f, "{}", name.name),
            TypeExprKind::Generic { name, args } => {
                write!(f, "{}<", name.name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ">")
            }
            TypeExprKind::Array { elem, size } => write!(f, "[{}; {}]", elem, size),
            TypeExprKind::Ref { mutable, inner } => {
                if *mutable {
                    write!(f, "&mut {}", inner)
                } else {
                    write!(f, "&{}", inner)
                }
            }
            TypeExprKind::Tuple(types) => {
                write!(f, "(")?;
                for (i, t) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", t)?;
                }
                write!(f, ")")
            }
            TypeExprKind::Shape(dims) => {
                write!(f, "[")?;
                for (i, d) in dims.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", d)?;
                }
                write!(f, "]")
            }
            TypeExprKind::Fn { params, ret } => {
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
            TypeExprKind::Sparse(inner) => write!(f, "Sparse<{}>", inner),
            TypeExprKind::SparseIndex { batch, k } => write!(f, "SparseIndex[{}, {}]", batch, k),
        }
    }
}

impl fmt::Display for TypeArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeArg::Type(ty) => write!(f, "{}", ty),
            TypeArg::Expr(expr) => write!(f, "{}", expr),
            TypeArg::Shape(dims) => {
                write!(f, "[")?;
                for (i, d) in dims.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", d)?;
                }
                write!(f, "]")
            }
        }
    }
}

impl fmt::Display for ShapeDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeDim::Lit(n) => write!(f, "{}", n),
            ShapeDim::Ident(id) => write!(f, "{}", id.name),
            ShapeDim::Expr(e) => write!(f, "{}", e),
            ShapeDim::Dynamic => write!(f, "?"),
        }
    }
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pattern::Wildcard => write!(f, "_"),
            Pattern::Literal(expr) => write!(f, "{}", expr),
            Pattern::Ident(id) => write!(f, "{}", id.name),
            Pattern::Variant { name, fields } => {
                write!(f, "{}", name.name)?;
                if !fields.is_empty() {
                    write!(f, "(")?;
                    for (i, p) in fields.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}", p)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{{")?;
        for stmt in &self.stmts {
            writeln!(f, "    {}", stmt)?;
        }
        if let Some(expr) = &self.expr {
            writeln!(f, "    {}", expr)?;
        }
        write!(f, "}}")
    }
}

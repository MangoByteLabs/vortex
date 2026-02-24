use logos::Logos;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

#[derive(Logos, Debug, Clone, Copy, PartialEq, Eq)]
#[logos(skip r"[ \t\r]+")]
#[logos(skip r"//[^\n]*")]
#[logos(skip r"/\*([^*]|\*[^/])*\*/")]
pub enum TokenKind {
    // Keywords - declarations
    #[token("fn")]
    Fn,
    #[token("kernel")]
    Kernel,
    #[token("struct")]
    Struct,
    #[token("enum")]
    Enum,
    #[token("trait")]
    Trait,
    #[token("impl")]
    Impl,
    #[token("type")]
    Type,
    #[token("const")]
    Const,
    #[token("pub")]
    Pub,
    #[token("import")]
    Import,
    #[token("from")]
    From,

    // Keywords - bindings
    #[token("let")]
    Let,
    #[token("var")]
    Var,
    #[token("mut")]
    Mut,

    // Keywords - control flow
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("match")]
    Match,
    #[token("for")]
    For,
    #[token("in")]
    In,
    #[token("while")]
    While,
    #[token("loop")]
    Loop,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("return")]
    Return,

    // Keywords - types & values
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("where")]
    Where,
    #[token("as")]
    As,

    // Keywords - GPU specific
    #[token("comptime")]
    Comptime,
    #[token("pipeline")]
    Pipeline,
    #[token("stage")]
    Stage,

    // Keywords - sparse / dispatch
    #[token("Sparse")]
    Sparse,
    #[token("SparseIndex")]
    SparseIndex,
    #[token("dispatch")]
    Dispatch,
    #[token("field")]
    Field,
    #[token("diff")]
    Diff,
    #[token("unique")]
    Unique,
    #[token("live")]
    Live,
    #[token("fuse")]
    Fuse,
    #[token("gpu")]
    Gpu,
    #[token("parallel")]
    Parallel,
    #[token("train")]
    Train,
    #[token("deterministic")]
    Deterministic,
    #[token("autocast")]
    Autocast,
    #[token("speculate")]
    Speculate,
    #[token("stream")]
    Stream,
    #[token("reward")]
    Reward,
    #[token("topology")]
    Topology,
    #[token("evolve")]
    Evolve,
    #[token("mmap")]
    Mmap,
    #[token("explain")]
    Explain,
    #[token("cache")]
    Cache,
    #[token("quantize")]
    Quantize,
    #[token("safe")]
    Safe,
    #[token("consensus")]
    Consensus,
    #[token("symbolic")]
    Symbolic,
    #[token("temporal")]
    Temporal,
    #[token("federated")]
    Federated,
    #[token("sandbox")]
    Sandbox,
    #[token("compress")]
    Compress,
    #[token("alignment")]
    Alignment,
    #[token("metacognition")]
    Metacognition,
    #[token("theorem")]
    Theorem,
    #[token("continual")]
    Continual,
    #[token("multimodal")]
    Multimodal,
    #[token("world_model")]
    WorldModel,
    #[token("self_improve")]
    SelfImprove,
    #[token("intention")]
    Intention,
    #[token("memory")]
    Memory,
    #[token("attention")]
    Attention,
    #[token("curriculum")]
    Curriculum,
    #[token("ensemble")]
    Ensemble,
    #[token("adversarial")]
    Adversarial,
    #[token("transfer")]
    Transfer,
    #[token("sparse")]
    SparseScope,
    #[token("async_infer")]
    AsyncInfer,
    #[token("profile")]
    Profile,
    #[token("contract")]
    Contract,

    // Arithmetic operators
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("**")]
    StarStar,

    // Matrix multiply
    #[token("@")]
    At,

    // Elementwise operators
    #[token(".*")]
    DotStar,
    #[token("./")]
    DotSlash,

    // Comparison operators
    #[token("==")]
    EqEq,
    #[token("!=")]
    BangEq,
    #[token("<=")]
    LessEq,
    #[token(">=")]
    GreaterEq,
    // Note: < and > are handled specially due to generic syntax
    // We tokenize them as Lt/Gt and disambiguate in the parser

    // Logical operators
    #[token("&&")]
    AmpAmp,
    #[token("||")]
    PipePipe,
    #[token("!")]
    Bang,

    // Bitwise operators
    #[token("&")]
    Amp,
    #[token("|")]
    Pipe,
    #[token("^")]
    Caret,
    #[token("~")]
    Tilde,
    #[token("<<")]
    LessLess,
    #[token(">>")]
    GreaterGreater,

    // Assignment operators
    #[token("=")]
    Eq,
    #[token("+=")]
    PlusEq,
    #[token("-=")]
    MinusEq,
    #[token("*=")]
    StarEq,
    #[token("/=")]
    SlashEq,
    #[token("@=")]
    AtEq,

    // Delimiters
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,

    // Punctuation
    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token(";")]
    Semicolon,
    #[token(".")]
    Dot,
    #[token("..")]
    DotDot,
    #[token("->")]
    Arrow,
    #[token("=>")]
    FatArrow,
    #[token("?")]
    Question,
    #[token("#")]
    Hash,

    // Newlines (significant for statement termination)
    #[token("\n")]
    Newline,

    // Literals
    #[regex(r"0x[0-9a-fA-F][0-9a-fA-F_]*n", priority = 5)]
    HexBigIntLiteral,

    #[regex(r"[0-9][0-9_]*n", priority = 4)]
    BigIntLiteral,

    #[regex(r"0x[0-9a-fA-F][0-9a-fA-F_]*", priority = 3)]
    HexLiteral,

    #[regex(r"[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9]+)?", priority = 3)]
    FloatLiteral,

    #[regex(r"[0-9][0-9_]*", priority = 2)]
    IntLiteral,

    #[regex(r#""([^"\\]|\\.)*""#)]
    StringLiteral,

    // Identifiers
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", priority = 1)]
    Ident,

    // End of file
    Eof,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Fn => write!(f, "fn"),
            TokenKind::Kernel => write!(f, "kernel"),
            TokenKind::Struct => write!(f, "struct"),
            TokenKind::Enum => write!(f, "enum"),
            TokenKind::Trait => write!(f, "trait"),
            TokenKind::Impl => write!(f, "impl"),
            TokenKind::Type => write!(f, "type"),
            TokenKind::Const => write!(f, "const"),
            TokenKind::Pub => write!(f, "pub"),
            TokenKind::Import => write!(f, "import"),
            TokenKind::From => write!(f, "from"),
            TokenKind::Let => write!(f, "let"),
            TokenKind::Var => write!(f, "var"),
            TokenKind::Mut => write!(f, "mut"),
            TokenKind::If => write!(f, "if"),
            TokenKind::Else => write!(f, "else"),
            TokenKind::Match => write!(f, "match"),
            TokenKind::For => write!(f, "for"),
            TokenKind::In => write!(f, "in"),
            TokenKind::While => write!(f, "while"),
            TokenKind::Loop => write!(f, "loop"),
            TokenKind::Break => write!(f, "break"),
            TokenKind::Continue => write!(f, "continue"),
            TokenKind::Return => write!(f, "return"),
            TokenKind::True => write!(f, "true"),
            TokenKind::False => write!(f, "false"),
            TokenKind::Where => write!(f, "where"),
            TokenKind::As => write!(f, "as"),
            TokenKind::Comptime => write!(f, "comptime"),
            TokenKind::Pipeline => write!(f, "pipeline"),
            TokenKind::Stage => write!(f, "stage"),
            TokenKind::Sparse => write!(f, "Sparse"),
            TokenKind::SparseIndex => write!(f, "SparseIndex"),
            TokenKind::Dispatch => write!(f, "dispatch"),
            TokenKind::Field => write!(f, "field"),
            TokenKind::Diff => write!(f, "diff"),
            TokenKind::Unique => write!(f, "unique"),
            TokenKind::Live => write!(f, "live"),
            TokenKind::Fuse => write!(f, "fuse"),
            TokenKind::Gpu => write!(f, "gpu"),
            TokenKind::Parallel => write!(f, "parallel"),
            TokenKind::Train => write!(f, "train"),
            TokenKind::Deterministic => write!(f, "deterministic"),
            TokenKind::Autocast => write!(f, "autocast"),
            TokenKind::Speculate => write!(f, "speculate"),
            TokenKind::Stream => write!(f, "stream"),
            TokenKind::Reward => write!(f, "reward"),
            TokenKind::Topology => write!(f, "topology"),
            TokenKind::Evolve => write!(f, "evolve"),
            TokenKind::Mmap => write!(f, "mmap"),
            TokenKind::Explain => write!(f, "explain"),
            TokenKind::Cache => write!(f, "cache"),
            TokenKind::Quantize => write!(f, "quantize"),
            TokenKind::Safe => write!(f, "safe"),
            TokenKind::Consensus => write!(f, "consensus"),
            TokenKind::Symbolic => write!(f, "symbolic"),
            TokenKind::Temporal => write!(f, "temporal"),
            TokenKind::Federated => write!(f, "federated"),
            TokenKind::Sandbox => write!(f, "sandbox"),
            TokenKind::Compress => write!(f, "compress"),
            TokenKind::Alignment => write!(f, "alignment"),
            TokenKind::Metacognition => write!(f, "metacognition"),
            TokenKind::Theorem => write!(f, "theorem"),
            TokenKind::Continual => write!(f, "continual"),
            TokenKind::Multimodal => write!(f, "multimodal"),
            TokenKind::WorldModel => write!(f, "world_model"),
            TokenKind::SelfImprove => write!(f, "self_improve"),
            TokenKind::Intention => write!(f, "intention"),
            TokenKind::Memory => write!(f, "memory"),
            TokenKind::Attention => write!(f, "attention"),
            TokenKind::Curriculum => write!(f, "curriculum"),
            TokenKind::Ensemble => write!(f, "ensemble"),
            TokenKind::Adversarial => write!(f, "adversarial"),
            TokenKind::Transfer => write!(f, "transfer"),
            TokenKind::SparseScope => write!(f, "sparse"),
            TokenKind::AsyncInfer => write!(f, "async_infer"),
            TokenKind::Profile => write!(f, "profile"),
            TokenKind::Contract => write!(f, "contract"),
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Star => write!(f, "*"),
            TokenKind::Slash => write!(f, "/"),
            TokenKind::Percent => write!(f, "%"),
            TokenKind::StarStar => write!(f, "**"),
            TokenKind::At => write!(f, "@"),
            TokenKind::DotStar => write!(f, ".*"),
            TokenKind::DotSlash => write!(f, "./"),
            TokenKind::EqEq => write!(f, "=="),
            TokenKind::BangEq => write!(f, "!="),
            TokenKind::LessEq => write!(f, "<="),
            TokenKind::GreaterEq => write!(f, ">="),
            TokenKind::AmpAmp => write!(f, "&&"),
            TokenKind::PipePipe => write!(f, "||"),
            TokenKind::Bang => write!(f, "!"),
            TokenKind::Amp => write!(f, "&"),
            TokenKind::Pipe => write!(f, "|"),
            TokenKind::Caret => write!(f, "^"),
            TokenKind::Tilde => write!(f, "~"),
            TokenKind::LessLess => write!(f, "<<"),
            TokenKind::GreaterGreater => write!(f, ">>"),
            TokenKind::Eq => write!(f, "="),
            TokenKind::PlusEq => write!(f, "+="),
            TokenKind::MinusEq => write!(f, "-="),
            TokenKind::StarEq => write!(f, "*="),
            TokenKind::SlashEq => write!(f, "/="),
            TokenKind::AtEq => write!(f, "@="),
            TokenKind::LParen => write!(f, "("),
            TokenKind::RParen => write!(f, ")"),
            TokenKind::LBracket => write!(f, "["),
            TokenKind::RBracket => write!(f, "]"),
            TokenKind::LBrace => write!(f, "{{"),
            TokenKind::RBrace => write!(f, "}}"),
            TokenKind::Lt => write!(f, "<"),
            TokenKind::Gt => write!(f, ">"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::Dot => write!(f, "."),
            TokenKind::DotDot => write!(f, ".."),
            TokenKind::Arrow => write!(f, "->"),
            TokenKind::FatArrow => write!(f, "=>"),
            TokenKind::Question => write!(f, "?"),
            TokenKind::Hash => write!(f, "#"),
            TokenKind::Newline => write!(f, "\\n"),
            TokenKind::HexBigIntLiteral => write!(f, "hex bigint literal"),
            TokenKind::BigIntLiteral => write!(f, "bigint literal"),
            TokenKind::HexLiteral => write!(f, "hex literal"),
            TokenKind::FloatLiteral => write!(f, "float literal"),
            TokenKind::IntLiteral => write!(f, "int literal"),
            TokenKind::StringLiteral => write!(f, "string literal"),
            TokenKind::Ident => write!(f, "identifier"),
            TokenKind::Eof => write!(f, "end of file"),
        }
    }
}

pub fn lex(source: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut lexer = TokenKind::lexer(source);

    while let Some(result) = lexer.next() {
        let span = lexer.span();
        let text = lexer.slice().to_string();
        match result {
            Ok(kind) => {
                tokens.push(Token {
                    kind,
                    span: Span::new(span.start, span.end),
                    text,
                });
            }
            Err(()) => {
                // Skip unknown characters but could collect errors
                tokens.push(Token {
                    kind: TokenKind::Ident, // fallback
                    span: Span::new(span.start, span.end),
                    text,
                });
            }
        }
    }

    tokens.push(Token {
        kind: TokenKind::Eof,
        span: Span::new(source.len(), source.len()),
        text: String::new(),
    });

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let tokens = lex("fn main() { }");
        let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind).collect();
        assert_eq!(
            kinds,
            vec![
                TokenKind::Fn,
                TokenKind::Ident,
                TokenKind::LParen,
                TokenKind::RParen,
                TokenKind::LBrace,
                TokenKind::RBrace,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_kernel_declaration() {
        let tokens = lex("kernel vector_add(a: Tensor<f32, [N]>) -> Tensor<f32, [N]>");
        let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind).collect();
        assert_eq!(
            kinds,
            vec![
                TokenKind::Kernel,
                TokenKind::Ident, // vector_add
                TokenKind::LParen,
                TokenKind::Ident, // a
                TokenKind::Colon,
                TokenKind::Ident, // Tensor
                TokenKind::Lt,
                TokenKind::Ident, // f32
                TokenKind::Comma,
                TokenKind::LBracket,
                TokenKind::Ident, // N
                TokenKind::RBracket,
                TokenKind::Gt,
                TokenKind::RParen,
                TokenKind::Arrow,
                TokenKind::Ident, // Tensor
                TokenKind::Lt,
                TokenKind::Ident, // f32
                TokenKind::Comma,
                TokenKind::LBracket,
                TokenKind::Ident, // N
                TokenKind::RBracket,
                TokenKind::Gt,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_operators() {
        let tokens = lex("a + b * c @ d == e && f");
        let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind).collect();
        assert_eq!(
            kinds,
            vec![
                TokenKind::Ident,
                TokenKind::Plus,
                TokenKind::Ident,
                TokenKind::Star,
                TokenKind::Ident,
                TokenKind::At,
                TokenKind::Ident,
                TokenKind::EqEq,
                TokenKind::Ident,
                TokenKind::AmpAmp,
                TokenKind::Ident,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_literals() {
        let tokens = lex("42 3.14 0xFF true false \"hello\"");
        let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind).collect();
        assert_eq!(
            kinds,
            vec![
                TokenKind::IntLiteral,
                TokenKind::FloatLiteral,
                TokenKind::HexLiteral,
                TokenKind::True,
                TokenKind::False,
                TokenKind::StringLiteral,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_comments_skipped() {
        let tokens = lex("a // this is a comment\nb");
        let kinds: Vec<TokenKind> = tokens
            .iter()
            .filter(|t| t.kind != TokenKind::Newline)
            .map(|t| t.kind)
            .collect();
        assert_eq!(kinds, vec![TokenKind::Ident, TokenKind::Ident, TokenKind::Eof]);
    }

    #[test]
    fn test_range_and_arrow() {
        let tokens = lex("0..N -> f32");
        let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind).collect();
        assert_eq!(
            kinds,
            vec![
                TokenKind::IntLiteral,
                TokenKind::DotDot,
                TokenKind::Ident,
                TokenKind::Arrow,
                TokenKind::Ident,
                TokenKind::Eof,
            ]
        );
    }
}

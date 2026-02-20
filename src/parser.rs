use crate::ast::*;
use crate::lexer::{Span, Token, TokenKind};
use codespan_reporting::diagnostic::{Diagnostic, Label};

type FileId = usize;

pub fn parse(
    tokens: Vec<Token>,
    file_id: FileId,
) -> Result<Program, Vec<Diagnostic<FileId>>> {
    let mut parser = Parser::new(tokens, file_id);
    parser.parse_program()
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    file_id: FileId,
    errors: Vec<Diagnostic<FileId>>,
}

impl Parser {
    fn new(tokens: Vec<Token>, file_id: FileId) -> Self {
        Self {
            tokens,
            pos: 0,
            file_id,
            errors: Vec::new(),
        }
    }

    // --- Token access helpers ---

    fn peek(&self) -> &Token {
        self.skip_newlines_peek()
    }

    fn peek_kind(&self) -> TokenKind {
        self.peek().kind
    }

    fn skip_newlines_peek(&self) -> &Token {
        let mut pos = self.pos;
        while pos < self.tokens.len() && self.tokens[pos].kind == TokenKind::Newline {
            pos += 1;
        }
        if pos < self.tokens.len() {
            &self.tokens[pos]
        } else {
            self.tokens.last().unwrap()
        }
    }

    fn advance(&mut self) -> Token {
        self.skip_newlines();
        let tok = self.tokens[self.pos].clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        tok
    }

    fn skip_newlines(&mut self) {
        while self.pos < self.tokens.len() && self.tokens[self.pos].kind == TokenKind::Newline {
            self.pos += 1;
        }
    }

    fn expect(&mut self, kind: TokenKind) -> Result<Token, ()> {
        let tok = self.peek().clone();
        if tok.kind == kind {
            Ok(self.advance())
        } else {
            self.error(
                tok.span,
                format!("expected `{}`, found `{}`", kind, tok.kind),
            );
            Err(())
        }
    }

    fn check(&self, kind: TokenKind) -> bool {
        self.peek_kind() == kind
    }

    fn eat(&mut self, kind: TokenKind) -> Option<Token> {
        if self.check(kind) {
            Some(self.advance())
        } else {
            None
        }
    }

    fn at_end(&self) -> bool {
        self.peek_kind() == TokenKind::Eof
    }

    fn span(&self) -> Span {
        self.peek().span
    }

    fn error(&mut self, span: Span, msg: String) {
        self.errors.push(
            Diagnostic::error()
                .with_message(&msg)
                .with_labels(vec![Label::primary(self.file_id, span.start..span.end)]),
        );
    }

    // --- Parse entry point ---

    fn parse_program(&mut self) -> Result<Program, Vec<Diagnostic<FileId>>> {
        let mut items = Vec::new();

        while !self.at_end() {
            match self.parse_item() {
                Ok(item) => items.push(item),
                Err(()) => {
                    // Skip to next likely item start for error recovery
                    self.recover_to_item();
                }
            }
        }

        if self.errors.is_empty() {
            Ok(Program { items })
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    fn recover_to_item(&mut self) {
        loop {
            match self.peek_kind() {
                TokenKind::Fn
                | TokenKind::Kernel
                | TokenKind::Struct
                | TokenKind::Trait
                | TokenKind::Impl
                | TokenKind::Import
                | TokenKind::Pub
                | TokenKind::Const
                | TokenKind::Type
                | TokenKind::Eof => break,
                _ => {
                    self.advance();
                }
            }
        }
    }

    // --- Top-level items ---

    fn parse_item(&mut self) -> Result<Item, ()> {
        let start = self.span();
        let is_pub = self.eat(TokenKind::Pub).is_some();

        let kind = match self.peek_kind() {
            TokenKind::Fn => ItemKind::Function(self.parse_function()?),
            TokenKind::Kernel => ItemKind::Kernel(self.parse_kernel()?),
            TokenKind::Struct => ItemKind::Struct(self.parse_struct()?),
            TokenKind::Trait => ItemKind::Trait(self.parse_trait()?),
            TokenKind::Impl => ItemKind::Impl(self.parse_impl()?),
            TokenKind::Import => ItemKind::Import(self.parse_import()?),
            TokenKind::Const => ItemKind::Const(self.parse_const()?),
            TokenKind::Type => ItemKind::TypeAlias(self.parse_type_alias()?),
            _ => {
                let tok = self.peek().clone();
                self.error(
                    tok.span,
                    format!("expected item declaration, found `{}`", tok.kind),
                );
                return Err(());
            }
        };

        let end_span = self.tokens[self.pos.saturating_sub(1)].span;

        Ok(Item {
            kind,
            span: start.merge(end_span),
            is_pub,
        })
    }

    // --- Function parsing ---

    fn parse_function(&mut self) -> Result<Function, ()> {
        self.expect(TokenKind::Fn)?;
        let name = self.parse_ident()?;
        let generics = self.parse_optional_generics()?;
        self.expect(TokenKind::LParen)?;
        let params = self.parse_params()?;
        self.expect(TokenKind::RParen)?;

        let ret_type = if self.eat(TokenKind::Arrow).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };

        let where_clause = self.parse_optional_where()?;
        let body = self.parse_block()?;

        Ok(Function {
            name,
            generics,
            params,
            ret_type,
            where_clause,
            body,
        })
    }

    // --- Kernel parsing ---

    fn parse_kernel(&mut self) -> Result<Kernel, ()> {
        self.expect(TokenKind::Kernel)?;
        let name = self.parse_ident()?;
        let generics = self.parse_optional_generics()?;
        self.expect(TokenKind::LParen)?;
        let params = self.parse_params()?;
        self.expect(TokenKind::RParen)?;

        let ret_type = if self.eat(TokenKind::Arrow).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };

        let where_clause = self.parse_optional_where()?;
        let body = self.parse_block()?;

        Ok(Kernel {
            name,
            schedule: None,
            generics,
            params,
            ret_type,
            where_clause,
            body,
        })
    }

    // --- Struct parsing ---

    fn parse_struct(&mut self) -> Result<StructDef, ()> {
        self.expect(TokenKind::Struct)?;
        let name = self.parse_ident()?;
        let generics = self.parse_optional_generics()?;
        self.expect(TokenKind::LBrace)?;

        let mut fields = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.at_end() {
            let start = self.span();
            let is_pub = self.eat(TokenKind::Pub).is_some();
            let field_name = self.parse_ident()?;
            self.expect(TokenKind::Colon)?;
            let ty = self.parse_type()?;
            let default = if self.eat(TokenKind::Eq).is_some() {
                Some(self.parse_expr()?)
            } else {
                None
            };
            let end = self.span();
            self.eat(TokenKind::Comma);
            fields.push(Field {
                name: field_name,
                ty,
                default,
                is_pub,
                span: start.merge(end),
            });
        }
        self.expect(TokenKind::RBrace)?;

        Ok(StructDef {
            name,
            generics,
            fields,
        })
    }

    // --- Trait parsing ---

    fn parse_trait(&mut self) -> Result<TraitDef, ()> {
        self.expect(TokenKind::Trait)?;
        let name = self.parse_ident()?;

        let supertraits = if self.eat(TokenKind::Colon).is_some() {
            let mut traits = vec![self.parse_type()?];
            while self.eat(TokenKind::Plus).is_some() {
                traits.push(self.parse_type()?);
            }
            traits
        } else {
            Vec::new()
        };

        self.expect(TokenKind::LBrace)?;
        let mut methods = Vec::new();

        while !self.check(TokenKind::RBrace) && !self.at_end() {
            let start = self.span();
            self.expect(TokenKind::Fn)?;
            let method_name = self.parse_ident()?;
            self.expect(TokenKind::LParen)?;
            let params = self.parse_params()?;
            self.expect(TokenKind::RParen)?;

            let ret_type = if self.eat(TokenKind::Arrow).is_some() {
                Some(self.parse_type()?)
            } else {
                None
            };

            let body = if self.check(TokenKind::LBrace) {
                Some(self.parse_block()?)
            } else {
                None
            };

            let end = self.span();
            methods.push(TraitMethod {
                name: method_name,
                params,
                ret_type,
                body,
                span: start.merge(end),
            });
        }
        self.expect(TokenKind::RBrace)?;

        Ok(TraitDef {
            name,
            supertraits,
            methods,
        })
    }

    // --- Impl parsing ---

    fn parse_impl(&mut self) -> Result<ImplBlock, ()> {
        self.expect(TokenKind::Impl)?;
        let first_type = self.parse_type()?;

        let (trait_name, target) = if self.eat(TokenKind::For).is_some() {
            let target = self.parse_type()?;
            (Some(first_type), target)
        } else {
            (None, first_type)
        };

        self.expect(TokenKind::LBrace)?;
        let mut methods = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.at_end() {
            methods.push(self.parse_item()?);
        }
        self.expect(TokenKind::RBrace)?;

        Ok(ImplBlock {
            trait_name,
            target,
            methods,
        })
    }

    // --- Import parsing ---

    fn parse_import(&mut self) -> Result<ImportDecl, ()> {
        let start = self.span();
        self.expect(TokenKind::Import)?;

        let mut path = vec![self.parse_ident()?];
        while self.eat(TokenKind::Dot).is_some() {
            if self.check(TokenKind::LBrace) {
                break;
            }
            path.push(self.parse_ident()?);
        }

        let items = if self.eat(TokenKind::LBrace).is_some() {
            if self.eat(TokenKind::Star).is_some() {
                self.expect(TokenKind::RBrace)?;
                ImportItems::Wildcard
            } else {
                let mut items = Vec::new();
                loop {
                    let name = self.parse_ident()?;
                    let alias = if self.eat(TokenKind::As).is_some() {
                        Some(self.parse_ident()?)
                    } else {
                        None
                    };
                    items.push(ImportItem { name, alias });
                    if !self.eat(TokenKind::Comma).is_some() {
                        break;
                    }
                }
                self.expect(TokenKind::RBrace)?;
                ImportItems::Named(items)
            }
        } else {
            ImportItems::Named(Vec::new())
        };

        let end = self.span();
        Ok(ImportDecl {
            path,
            items,
            span: start.merge(end),
        })
    }

    // --- Const parsing ---

    fn parse_const(&mut self) -> Result<ConstDecl, ()> {
        self.expect(TokenKind::Const)?;
        let name = self.parse_ident()?;
        let ty = if self.eat(TokenKind::Colon).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(TokenKind::Eq)?;
        let value = self.parse_expr()?;

        Ok(ConstDecl { name, ty, value })
    }

    // --- Type alias parsing ---

    fn parse_type_alias(&mut self) -> Result<TypeAlias, ()> {
        self.expect(TokenKind::Type)?;
        let name = self.parse_ident()?;
        let generics = self.parse_optional_generics()?;
        self.expect(TokenKind::Eq)?;
        let value = self.parse_type()?;

        Ok(TypeAlias {
            name,
            generics,
            value,
        })
    }

    // --- Generics ---

    fn parse_optional_generics(&mut self) -> Result<Vec<GenericParam>, ()> {
        if !self.check(TokenKind::Lt) {
            return Ok(Vec::new());
        }
        self.advance(); // consume <

        let mut params = Vec::new();
        loop {
            if self.check(TokenKind::Gt) {
                break;
            }

            let start = self.span();

            // Check for const generics: const N: usize
            if self.eat(TokenKind::Const).is_some() {
                let name = self.parse_ident()?;
                self.expect(TokenKind::Colon)?;
                let ty = self.parse_type()?;
                params.push(GenericParam {
                    name,
                    kind: GenericParamKind::Const { ty },
                    span: start.merge(self.span()),
                });
            } else {
                let name = self.parse_ident()?;
                let bounds = if self.eat(TokenKind::Colon).is_some() {
                    let mut bounds = vec![self.parse_type()?];
                    while self.eat(TokenKind::Plus).is_some() {
                        bounds.push(self.parse_type()?);
                    }
                    bounds
                } else {
                    Vec::new()
                };
                params.push(GenericParam {
                    name,
                    kind: GenericParamKind::Type { bounds },
                    span: start.merge(self.span()),
                });
            }

            if !self.eat(TokenKind::Comma).is_some() {
                break;
            }
        }
        self.expect(TokenKind::Gt)?;
        Ok(params)
    }

    // --- Where clauses ---

    fn parse_optional_where(&mut self) -> Result<Vec<WhereClause>, ()> {
        if !self.eat(TokenKind::Where).is_some() {
            return Ok(Vec::new());
        }

        let mut clauses = Vec::new();
        loop {
            if self.check(TokenKind::LBrace) {
                break;
            }
            let start = self.span();
            let ty = self.parse_type()?;
            self.expect(TokenKind::Colon)?;
            let mut bounds = vec![self.parse_type()?];
            while self.eat(TokenKind::Plus).is_some() {
                bounds.push(self.parse_type()?);
            }
            let end = self.span();
            clauses.push(WhereClause {
                ty,
                bounds,
                span: start.merge(end),
            });
            if !self.eat(TokenKind::Comma).is_some() {
                break;
            }
        }
        Ok(clauses)
    }

    // --- Parameters ---

    fn parse_params(&mut self) -> Result<Vec<Param>, ()> {
        let mut params = Vec::new();
        while !self.check(TokenKind::RParen) && !self.at_end() {
            let start = self.span();
            let name = self.parse_ident()?;
            self.expect(TokenKind::Colon)?;
            let ty = self.parse_type()?;
            let default = if self.eat(TokenKind::Eq).is_some() {
                Some(self.parse_expr()?)
            } else {
                None
            };
            let end = self.span();
            params.push(Param {
                name,
                ty,
                default,
                span: start.merge(end),
            });
            if !self.eat(TokenKind::Comma).is_some() {
                break;
            }
        }
        Ok(params)
    }

    // --- Type parsing ---

    fn parse_type(&mut self) -> Result<TypeExpr, ()> {
        let start = self.span();

        // Reference types: &T, &mut T
        if self.eat(TokenKind::Amp).is_some() {
            let mutable = self.eat(TokenKind::Mut).is_some();
            let inner = self.parse_type()?;
            let end = inner.span;
            return Ok(TypeExpr {
                kind: TypeExprKind::Ref {
                    mutable,
                    inner: Box::new(inner),
                },
                span: start.merge(end),
            });
        }

        // Tuple types: (T, U)
        if self.check(TokenKind::LParen) {
            self.advance();
            let mut types = Vec::new();
            while !self.check(TokenKind::RParen) && !self.at_end() {
                types.push(self.parse_type()?);
                if !self.eat(TokenKind::Comma).is_some() {
                    break;
                }
            }
            let end = self.expect(TokenKind::RParen)?.span;
            return Ok(TypeExpr {
                kind: TypeExprKind::Tuple(types),
                span: start.merge(end),
            });
        }

        // Array types: [T; N]
        if self.check(TokenKind::LBracket) {
            return self.parse_array_or_shape_type();
        }

        // fn types: fn(T, U) -> V
        if self.check(TokenKind::Fn) {
            self.advance();
            self.expect(TokenKind::LParen)?;
            let mut params = Vec::new();
            while !self.check(TokenKind::RParen) && !self.at_end() {
                params.push(self.parse_type()?);
                if !self.eat(TokenKind::Comma).is_some() {
                    break;
                }
            }
            self.expect(TokenKind::RParen)?;
            self.expect(TokenKind::Arrow)?;
            let ret = self.parse_type()?;
            let end = ret.span;
            return Ok(TypeExpr {
                kind: TypeExprKind::Fn {
                    params,
                    ret: Box::new(ret),
                },
                span: start.merge(end),
            });
        }

        // Named or generic type
        let name = self.parse_ident()?;

        if self.check(TokenKind::Lt) {
            // Generic type: Tensor<f32, [N, M]>
            self.advance();
            let mut args = Vec::new();
            loop {
                if self.check(TokenKind::Gt) {
                    break;
                }
                if self.check(TokenKind::LBracket) {
                    // Shape argument
                    let dims = self.parse_shape_dims()?;
                    args.push(TypeArg::Shape(dims));
                } else {
                    // Try to parse as a type
                    let ty = self.parse_type()?;
                    args.push(TypeArg::Type(ty));
                }
                if !self.eat(TokenKind::Comma).is_some() {
                    break;
                }
            }
            let end = self.expect(TokenKind::Gt)?.span;
            Ok(TypeExpr {
                kind: TypeExprKind::Generic { name, args },
                span: start.merge(end),
            })
        } else {
            let end = name.span;
            Ok(TypeExpr {
                kind: TypeExprKind::Named(name),
                span: start.merge(end),
            })
        }
    }

    fn parse_array_or_shape_type(&mut self) -> Result<TypeExpr, ()> {
        let start = self.span();
        self.expect(TokenKind::LBracket)?;

        // Check if this is [T; N] (array) or [N, M] (shape)
        // We use a heuristic: if first element is a type name followed by `;`, it's an array
        let first_tok = self.peek().clone();
        if first_tok.kind == TokenKind::Ident {
            // Look ahead: if there's a `;` after the type, it's an array
            let saved_pos = self.pos;
            let _ = self.advance();
            if self.check(TokenKind::Semicolon) {
                // It's [T; N] array
                self.pos = saved_pos;
                let elem = self.parse_type()?;
                self.expect(TokenKind::Semicolon)?;
                let size = self.parse_expr()?;
                let end = self.expect(TokenKind::RBracket)?.span;
                return Ok(TypeExpr {
                    kind: TypeExprKind::Array {
                        elem: Box::new(elem),
                        size: Box::new(size),
                    },
                    span: start.merge(end),
                });
            }
            // Restore and parse as shape
            self.pos = saved_pos;
        }

        // Parse as shape: [N, M, K]
        let dims = self.parse_shape_dims_inner()?;
        let end = self.expect(TokenKind::RBracket)?.span;
        Ok(TypeExpr {
            kind: TypeExprKind::Shape(dims),
            span: start.merge(end),
        })
    }

    fn parse_shape_dims(&mut self) -> Result<Vec<ShapeDim>, ()> {
        self.expect(TokenKind::LBracket)?;
        let dims = self.parse_shape_dims_inner()?;
        self.expect(TokenKind::RBracket)?;
        Ok(dims)
    }

    fn parse_shape_dims_inner(&mut self) -> Result<Vec<ShapeDim>, ()> {
        let mut dims = Vec::new();
        while !self.check(TokenKind::RBracket) && !self.at_end() {
            let dim = match self.peek_kind() {
                TokenKind::IntLiteral => {
                    let tok = self.advance();
                    let n: u64 = tok.text.replace('_', "").parse().unwrap_or(0);
                    ShapeDim::Lit(n)
                }
                TokenKind::Question => {
                    self.advance();
                    ShapeDim::Dynamic
                }
                TokenKind::Ident => {
                    let id = self.parse_ident()?;
                    ShapeDim::Ident(id)
                }
                _ => {
                    let tok = self.peek().clone();
                    self.error(tok.span, format!("expected shape dimension, found `{}`", tok.kind));
                    return Err(());
                }
            };
            dims.push(dim);
            if !self.eat(TokenKind::Comma).is_some() {
                break;
            }
        }
        Ok(dims)
    }

    // --- Block parsing ---

    fn parse_block(&mut self) -> Result<Block, ()> {
        let start = self.span();
        self.expect(TokenKind::LBrace)?;

        let mut stmts = Vec::new();
        let mut trailing_expr = None;

        while !self.check(TokenKind::RBrace) && !self.at_end() {
            // Try to parse a statement
            match self.parse_stmt() {
                Ok(stmt) => {
                    stmts.push(stmt);
                }
                Err(()) => {
                    // Try to recover
                    self.recover_to_stmt();
                }
            }
        }

        // Check if the last statement is an expression (implicit return)
        if let Some(last) = stmts.last() {
            if let StmtKind::Expr(_) = &last.kind {
                if let Some(stmt) = stmts.pop() {
                    if let StmtKind::Expr(expr) = stmt.kind {
                        trailing_expr = Some(Box::new(expr));
                    }
                }
            }
        }

        let end = self.expect(TokenKind::RBrace)?.span;

        Ok(Block {
            stmts,
            expr: trailing_expr,
            span: start.merge(end),
        })
    }

    fn recover_to_stmt(&mut self) {
        loop {
            match self.peek_kind() {
                TokenKind::Let
                | TokenKind::Var
                | TokenKind::Return
                | TokenKind::For
                | TokenKind::While
                | TokenKind::If
                | TokenKind::RBrace
                | TokenKind::Eof => break,
                _ => {
                    self.advance();
                }
            }
        }
    }

    // --- Statement parsing ---

    fn parse_stmt(&mut self) -> Result<Stmt, ()> {
        let start = self.span();

        match self.peek_kind() {
            TokenKind::Let => {
                self.advance();
                let name = self.parse_ident()?;
                let ty = if self.eat(TokenKind::Colon).is_some() {
                    Some(self.parse_type()?)
                } else {
                    None
                };
                self.expect(TokenKind::Eq)?;
                let value = self.parse_expr()?;
                Ok(Stmt {
                    span: start.merge(value.span),
                    kind: StmtKind::Let { name, ty, value },
                })
            }
            TokenKind::Var => {
                self.advance();
                let name = self.parse_ident()?;
                let ty = if self.eat(TokenKind::Colon).is_some() {
                    Some(self.parse_type()?)
                } else {
                    None
                };
                self.expect(TokenKind::Eq)?;
                let value = self.parse_expr()?;
                Ok(Stmt {
                    span: start.merge(value.span),
                    kind: StmtKind::Var { name, ty, value },
                })
            }
            TokenKind::Return => {
                self.advance();
                let value = if self.check(TokenKind::RBrace) || self.check(TokenKind::Newline) || self.at_end() {
                    None
                } else {
                    Some(self.parse_expr()?)
                };
                let end = value.as_ref().map(|e| e.span).unwrap_or(start);
                Ok(Stmt {
                    span: start.merge(end),
                    kind: StmtKind::Return(value),
                })
            }
            TokenKind::For => self.parse_for_stmt(),
            TokenKind::While => self.parse_while_stmt(),
            TokenKind::Break => {
                self.advance();
                Ok(Stmt {
                    span: start,
                    kind: StmtKind::Break,
                })
            }
            TokenKind::Continue => {
                self.advance();
                Ok(Stmt {
                    span: start,
                    kind: StmtKind::Continue,
                })
            }
            _ => {
                // Expression statement, possibly followed by assignment
                let expr = self.parse_expr()?;

                // Check for assignment operators
                let assign_op = match self.peek_kind() {
                    TokenKind::Eq => Some(AssignOp::Assign),
                    TokenKind::PlusEq => Some(AssignOp::AddAssign),
                    TokenKind::MinusEq => Some(AssignOp::SubAssign),
                    TokenKind::StarEq => Some(AssignOp::MulAssign),
                    TokenKind::SlashEq => Some(AssignOp::DivAssign),
                    TokenKind::AtEq => Some(AssignOp::MatMulAssign),
                    _ => None,
                };

                if let Some(op) = assign_op {
                    self.advance();
                    let value = self.parse_expr()?;
                    Ok(Stmt {
                        span: start.merge(value.span),
                        kind: StmtKind::Assign {
                            target: expr,
                            op,
                            value,
                        },
                    })
                } else {
                    Ok(Stmt {
                        span: expr.span,
                        kind: StmtKind::Expr(expr),
                    })
                }
            }
        }
    }

    fn parse_for_stmt(&mut self) -> Result<Stmt, ()> {
        let start = self.span();
        self.expect(TokenKind::For)?;
        let var = self.parse_ident()?;
        self.expect(TokenKind::In)?;
        let iter = self.parse_expr()?;
        let body = self.parse_block()?;
        let end = body.span;
        Ok(Stmt {
            span: start.merge(end),
            kind: StmtKind::For { var, iter, body },
        })
    }

    fn parse_while_stmt(&mut self) -> Result<Stmt, ()> {
        let start = self.span();
        self.expect(TokenKind::While)?;
        let cond = self.parse_expr()?;
        let body = self.parse_block()?;
        let end = body.span;
        Ok(Stmt {
            span: start.merge(end),
            kind: StmtKind::While { cond, body },
        })
    }

    // --- Expression parsing (precedence climbing) ---

    fn parse_expr(&mut self) -> Result<Expr, ()> {
        self.parse_expr_bp(0)
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr, ()> {
        let mut lhs = self.parse_unary()?;

        loop {
            // Check for binary operator
            let op = match self.peek_kind() {
                TokenKind::Plus => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                TokenKind::Star => BinOp::Mul,
                TokenKind::Slash => BinOp::Div,
                TokenKind::Percent => BinOp::Mod,
                TokenKind::StarStar => BinOp::Pow,
                TokenKind::EqEq => BinOp::Eq,
                TokenKind::BangEq => BinOp::NotEq,
                TokenKind::Lt => BinOp::Lt,
                TokenKind::Gt => BinOp::Gt,
                TokenKind::LessEq => BinOp::LtEq,
                TokenKind::GreaterEq => BinOp::GtEq,
                TokenKind::AmpAmp => BinOp::And,
                TokenKind::PipePipe => BinOp::Or,
                TokenKind::Amp => BinOp::BitAnd,
                TokenKind::Pipe => BinOp::BitOr,
                TokenKind::Caret => BinOp::BitXor,
                TokenKind::LessLess => BinOp::Shl,
                TokenKind::GreaterGreater => BinOp::Shr,
                TokenKind::DotStar => BinOp::ElemMul,
                TokenKind::DotSlash => BinOp::ElemDiv,
                TokenKind::DotDot => {
                    // Range expression
                    let prec: u8 = 0; // lowest precedence
                    if prec < min_bp {
                        break;
                    }
                    self.advance();
                    let rhs = self.parse_expr_bp(prec + 1)?;
                    let span = lhs.span.merge(rhs.span);
                    lhs = Expr {
                        kind: ExprKind::Range {
                            start: Box::new(lhs),
                            end: Box::new(rhs),
                        },
                        span,
                    };
                    continue;
                }
                TokenKind::At => {
                    // Matrix multiply — treated as an operator with precedence 10 (same as mul)
                    let prec: u8 = 10;
                    if prec < min_bp {
                        break;
                    }
                    self.advance();
                    let rhs = self.parse_expr_bp(prec + 1)?;
                    let span = lhs.span.merge(rhs.span);
                    lhs = Expr {
                        kind: ExprKind::MatMul {
                            lhs: Box::new(lhs),
                            rhs: Box::new(rhs),
                        },
                        span,
                    };
                    continue;
                }
                TokenKind::As => {
                    // Cast expression
                    self.advance();
                    let ty = self.parse_type()?;
                    let span = lhs.span.merge(ty.span);
                    lhs = Expr {
                        kind: ExprKind::Cast {
                            expr: Box::new(lhs),
                            ty,
                        },
                        span,
                    };
                    continue;
                }
                _ => break,
            };

            let prec = op.precedence();
            if prec < min_bp {
                break;
            }

            self.advance(); // consume operator

            let next_min = if op.is_right_assoc() { prec } else { prec + 1 };
            let rhs = self.parse_expr_bp(next_min)?;
            let span = lhs.span.merge(rhs.span);

            lhs = Expr {
                kind: ExprKind::Binary {
                    lhs: Box::new(lhs),
                    op,
                    rhs: Box::new(rhs),
                },
                span,
            };
        }

        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, ()> {
        let start = self.span();
        match self.peek_kind() {
            TokenKind::Minus => {
                self.advance();
                let expr = self.parse_unary()?;
                let span = start.merge(expr.span);
                Ok(Expr {
                    kind: ExprKind::Unary {
                        op: UnaryOp::Neg,
                        expr: Box::new(expr),
                    },
                    span,
                })
            }
            TokenKind::Bang => {
                self.advance();
                let expr = self.parse_unary()?;
                let span = start.merge(expr.span);
                Ok(Expr {
                    kind: ExprKind::Unary {
                        op: UnaryOp::Not,
                        expr: Box::new(expr),
                    },
                    span,
                })
            }
            TokenKind::Tilde => {
                self.advance();
                let expr = self.parse_unary()?;
                let span = start.merge(expr.span);
                Ok(Expr {
                    kind: ExprKind::Unary {
                        op: UnaryOp::BitNot,
                        expr: Box::new(expr),
                    },
                    span,
                })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, ()> {
        let mut expr = self.parse_primary()?;

        loop {
            match self.peek_kind() {
                TokenKind::LParen => {
                    // Function call
                    self.advance();
                    let mut args = Vec::new();
                    while !self.check(TokenKind::RParen) && !self.at_end() {
                        args.push(self.parse_expr()?);
                        if !self.eat(TokenKind::Comma).is_some() {
                            break;
                        }
                    }
                    let end = self.expect(TokenKind::RParen)?.span;
                    expr = Expr {
                        span: expr.span.merge(end),
                        kind: ExprKind::Call {
                            func: Box::new(expr),
                            args,
                        },
                    };
                }
                TokenKind::LBracket => {
                    // Indexing
                    self.advance();
                    let mut indices = Vec::new();
                    while !self.check(TokenKind::RBracket) && !self.at_end() {
                        indices.push(self.parse_expr()?);
                        if !self.eat(TokenKind::Comma).is_some() {
                            break;
                        }
                    }
                    let end = self.expect(TokenKind::RBracket)?.span;
                    expr = Expr {
                        span: expr.span.merge(end),
                        kind: ExprKind::Index {
                            base: Box::new(expr),
                            indices,
                        },
                    };
                }
                TokenKind::Dot => {
                    // Field access
                    self.advance();
                    let field = self.parse_ident()?;
                    let span = expr.span.merge(field.span);
                    expr = Expr {
                        span,
                        kind: ExprKind::FieldAccess {
                            base: Box::new(expr),
                            field,
                        },
                    };
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, ()> {
        let tok = self.peek().clone();
        let start = tok.span;

        match tok.kind {
            TokenKind::IntLiteral => {
                self.advance();
                let value: u128 = tok.text.replace('_', "").parse().unwrap_or(0);
                Ok(Expr {
                    kind: ExprKind::IntLiteral(value),
                    span: start,
                })
            }
            TokenKind::HexLiteral => {
                self.advance();
                let hex_str = tok.text.replace('_', "");
                let value = u128::from_str_radix(&hex_str[2..], 16).unwrap_or(0);
                Ok(Expr {
                    kind: ExprKind::IntLiteral(value),
                    span: start,
                })
            }
            TokenKind::FloatLiteral => {
                self.advance();
                let value: f64 = tok.text.replace('_', "").parse().unwrap_or(0.0);
                Ok(Expr {
                    kind: ExprKind::FloatLiteral(value),
                    span: start,
                })
            }
            TokenKind::StringLiteral => {
                self.advance();
                // Strip quotes
                let s = tok.text[1..tok.text.len() - 1].to_string();
                Ok(Expr {
                    kind: ExprKind::StringLiteral(s),
                    span: start,
                })
            }
            TokenKind::True => {
                self.advance();
                Ok(Expr {
                    kind: ExprKind::BoolLiteral(true),
                    span: start,
                })
            }
            TokenKind::False => {
                self.advance();
                Ok(Expr {
                    kind: ExprKind::BoolLiteral(false),
                    span: start,
                })
            }
            TokenKind::Ident => {
                let ident = self.parse_ident()?;
                // Check for struct literal: Name { field: value, ... }
                // Only if the { is on the same line (to distinguish from blocks)
                if self.check(TokenKind::LBrace) {
                    // Look ahead to see if this is Name { field: value } (struct)
                    // vs just an identifier before a block
                    let saved_pos = self.pos;
                    self.advance(); // consume {
                    // If next token is Ident followed by :, it's a struct literal
                    if self.check(TokenKind::Ident) {
                        let peek_pos = self.pos;
                        let _ = self.advance();
                        if self.check(TokenKind::Colon) {
                            // It's a struct literal — rewind and parse properly
                            self.pos = saved_pos;
                            return self.parse_struct_literal(ident);
                        }
                        self.pos = peek_pos;
                    }
                    // Check for empty struct: Name {}
                    if self.check(TokenKind::RBrace) {
                        let end = self.advance().span;
                        return Ok(Expr {
                            kind: ExprKind::StructLiteral {
                                name: ident,
                                fields: Vec::new(),
                            },
                            span: start.merge(end),
                        });
                    }
                    // Not a struct literal, restore position
                    self.pos = saved_pos;
                }
                Ok(Expr {
                    span: ident.span,
                    kind: ExprKind::Ident(ident),
                })
            }
            TokenKind::LParen => {
                // Parenthesized expression or tuple
                self.advance();
                let first = self.parse_expr()?;
                if self.check(TokenKind::Comma) {
                    // Tuple literal: (a, b, c)
                    let mut elems = vec![first];
                    while self.eat(TokenKind::Comma).is_some() {
                        if self.check(TokenKind::RParen) {
                            break; // trailing comma
                        }
                        elems.push(self.parse_expr()?);
                    }
                    let end = self.expect(TokenKind::RParen)?.span;
                    Ok(Expr {
                        kind: ExprKind::ArrayLiteral(elems), // reuse ArrayLiteral for tuples for now
                        span: start.merge(end),
                    })
                } else {
                    self.expect(TokenKind::RParen)?;
                    Ok(first)
                }
            }
            TokenKind::LBracket => {
                // Array literal: [1, 2, 3]
                self.advance();
                let mut elems = Vec::new();
                while !self.check(TokenKind::RBracket) && !self.at_end() {
                    elems.push(self.parse_expr()?);
                    if !self.eat(TokenKind::Comma).is_some() {
                        break;
                    }
                }
                let end = self.expect(TokenKind::RBracket)?.span;
                Ok(Expr {
                    kind: ExprKind::ArrayLiteral(elems),
                    span: start.merge(end),
                })
            }
            TokenKind::LBrace => {
                // Block expression
                let block = self.parse_block()?;
                let span = block.span;
                Ok(Expr {
                    kind: ExprKind::Block(block),
                    span,
                })
            }
            TokenKind::If => self.parse_if_expr(),
            TokenKind::Match => self.parse_match_expr(),
            _ => {
                self.error(start, format!("expected expression, found `{}`", tok.kind));
                Err(())
            }
        }
    }

    fn parse_struct_literal(&mut self, name: Ident) -> Result<Expr, ()> {
        let start = name.span;
        self.expect(TokenKind::LBrace)?;

        let mut fields = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.at_end() {
            let field_name = self.parse_ident()?;
            self.expect(TokenKind::Colon)?;
            let value = self.parse_expr()?;
            fields.push((field_name, value));
            if !self.eat(TokenKind::Comma).is_some() {
                break;
            }
        }

        let end = self.expect(TokenKind::RBrace)?.span;
        Ok(Expr {
            kind: ExprKind::StructLiteral { name, fields },
            span: start.merge(end),
        })
    }

    fn parse_match_expr(&mut self) -> Result<Expr, ()> {
        let start = self.span();
        self.expect(TokenKind::Match)?;
        let expr = self.parse_expr()?;
        self.expect(TokenKind::LBrace)?;

        let mut arms = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.at_end() {
            let arm_start = self.span();
            let pattern = self.parse_pattern()?;
            self.expect(TokenKind::FatArrow)?;
            let body = self.parse_expr()?;
            let arm_end = body.span;
            arms.push(MatchArm {
                pattern,
                body,
                span: arm_start.merge(arm_end),
            });
            // Allow optional comma between arms
            self.eat(TokenKind::Comma);
        }

        let end = self.expect(TokenKind::RBrace)?.span;
        Ok(Expr {
            kind: ExprKind::Match {
                expr: Box::new(expr),
                arms,
            },
            span: start.merge(end),
        })
    }

    fn parse_pattern(&mut self) -> Result<Pattern, ()> {
        match self.peek_kind() {
            TokenKind::Ident => {
                let ident = self.parse_ident()?;
                if ident.name == "_" {
                    return Ok(Pattern::Wildcard);
                }
                // Check for variant pattern: Name(fields...)
                if self.check(TokenKind::LParen) {
                    self.advance();
                    let mut fields = Vec::new();
                    while !self.check(TokenKind::RParen) && !self.at_end() {
                        fields.push(self.parse_pattern()?);
                        if !self.eat(TokenKind::Comma).is_some() {
                            break;
                        }
                    }
                    self.expect(TokenKind::RParen)?;
                    Ok(Pattern::Variant { name: ident, fields })
                } else {
                    Ok(Pattern::Ident(ident))
                }
            }
            TokenKind::IntLiteral | TokenKind::HexLiteral => {
                let expr = self.parse_primary()?;
                Ok(Pattern::Literal(expr))
            }
            TokenKind::StringLiteral => {
                let expr = self.parse_primary()?;
                Ok(Pattern::Literal(expr))
            }
            TokenKind::True | TokenKind::False => {
                let expr = self.parse_primary()?;
                Ok(Pattern::Literal(expr))
            }
            _ => {
                let tok = self.peek().clone();
                self.error(tok.span, format!("expected pattern, found `{}`", tok.kind));
                Err(())
            }
        }
    }

    fn parse_if_expr(&mut self) -> Result<Expr, ()> {
        let start = self.span();
        self.expect(TokenKind::If)?;
        let cond = self.parse_expr()?;
        let then_block = self.parse_block()?;
        let else_block = if self.eat(TokenKind::Else).is_some() {
            if self.check(TokenKind::If) {
                // else if => wrap in a block with single if expression
                let if_expr = self.parse_if_expr()?;
                let span = if_expr.span;
                Some(Block {
                    stmts: Vec::new(),
                    expr: Some(Box::new(if_expr)),
                    span,
                })
            } else {
                Some(self.parse_block()?)
            }
        } else {
            None
        };

        let end = else_block
            .as_ref()
            .map(|b| b.span)
            .unwrap_or(then_block.span);

        Ok(Expr {
            kind: ExprKind::If {
                cond: Box::new(cond),
                then_block,
                else_block,
            },
            span: start.merge(end),
        })
    }

    // --- Identifier ---

    fn parse_ident(&mut self) -> Result<Ident, ()> {
        let tok = self.peek().clone();
        // Accept actual identifiers and also keywords that might be used as identifiers in some contexts
        if tok.kind == TokenKind::Ident {
            self.advance();
            Ok(Ident::new(tok.text, tok.span))
        } else {
            self.error(
                tok.span,
                format!("expected identifier, found `{}`", tok.kind),
            );
            Err(())
        }
    }
}

// Add + operator for Span (used in merge context)
impl std::ops::Add for Span {
    type Output = Span;
    fn add(self, rhs: Self) -> Self {
        self.merge(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer;

    fn parse_ok(source: &str) -> Program {
        let tokens = lexer::lex(source);
        parse(tokens, 0).expect("parse failed")
    }

    #[test]
    fn test_parse_simple_function() {
        let program = parse_ok("fn add(a: f32, b: f32) -> f32 { return a + b }");
        assert_eq!(program.items.len(), 1);
        assert!(matches!(program.items[0].kind, ItemKind::Function(_)));
    }

    #[test]
    fn test_parse_kernel() {
        let program = parse_ok(
            "kernel vector_add(a: Tensor<f32, [N]>, b: Tensor<f32, [N]>) -> Tensor<f32, [N]> {
                return a + b
            }",
        );
        assert_eq!(program.items.len(), 1);
        assert!(matches!(program.items[0].kind, ItemKind::Kernel(_)));
    }

    #[test]
    fn test_parse_struct() {
        let program = parse_ok(
            "struct Point {
                x: f32,
                y: f32,
            }",
        );
        assert_eq!(program.items.len(), 1);
        if let ItemKind::Struct(s) = &program.items[0].kind {
            assert_eq!(s.name.name, "Point");
            assert_eq!(s.fields.len(), 2);
        } else {
            panic!("expected struct");
        }
    }

    #[test]
    fn test_parse_trait() {
        let program = parse_ok(
            "trait Numeric {
                fn zero() -> Self
                fn one() -> Self
            }",
        );
        assert_eq!(program.items.len(), 1);
        if let ItemKind::Trait(t) = &program.items[0].kind {
            assert_eq!(t.name.name, "Numeric");
            assert_eq!(t.methods.len(), 2);
        } else {
            panic!("expected trait");
        }
    }

    #[test]
    fn test_parse_import() {
        let program = parse_ok("import crypto.fields.bn254 { Fp, Fr }");
        assert_eq!(program.items.len(), 1);
        if let ItemKind::Import(i) = &program.items[0].kind {
            assert_eq!(i.path.len(), 3);
            if let ImportItems::Named(items) = &i.items {
                assert_eq!(items.len(), 2);
            }
        } else {
            panic!("expected import");
        }
    }

    #[test]
    fn test_parse_let_and_var() {
        let program = parse_ok(
            "fn test() {
                let x: f32 = 1.0
                var y = x + 2.0
            }",
        );
        if let ItemKind::Function(f) = &program.items[0].kind {
            // let is a stmt, var is also parsed as a stmt. The last stmt (var)
            // is an expression statement (StmtKind::Var), which is pulled as trailing expr.
            assert!(f.body.stmts.len() >= 1);
        }
    }

    #[test]
    fn test_parse_for_loop() {
        let program = parse_ok(
            "fn test() {
                for i in 0..10 {
                    let x = i
                }
            }",
        );
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_parse_if_else() {
        let program = parse_ok(
            "fn test(x: f32) -> f32 {
                if x > 0.0 { x } else { -x }
            }",
        );
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_parse_generic_function() {
        let program = parse_ok(
            "fn dot<T: Numeric, const N: usize>(a: Tensor<T, [N]>, b: Tensor<T, [N]>) -> T {
                return a + b
            }",
        );
        if let ItemKind::Function(f) = &program.items[0].kind {
            assert_eq!(f.generics.len(), 2);
        }
    }

    #[test]
    fn test_parse_matmul_operator() {
        let program = parse_ok(
            "fn test(a: Tensor<f32, [M, K]>, b: Tensor<f32, [K, N]>) -> Tensor<f32, [M, N]> {
                return a @ b
            }",
        );
        if let ItemKind::Function(f) = &program.items[0].kind {
            if let Some(expr) = &f.body.expr {
                assert!(matches!(expr.kind, ExprKind::MatMul { .. }));
            }
        }
    }

    #[test]
    fn test_parse_multiple_items() {
        let program = parse_ok(
            "
            import std.tensor { Tensor }

            fn add(a: f32, b: f32) -> f32 {
                return a + b
            }

            kernel vec_add(a: Tensor<f32, [N]>, b: Tensor<f32, [N]>) -> Tensor<f32, [N]> {
                return a + b
            }
            ",
        );
        assert_eq!(program.items.len(), 3);
    }

    #[test]
    fn test_parse_const_decl() {
        let program = parse_ok("const PI: f64 = 3.14159");
        assert_eq!(program.items.len(), 1);
        assert!(matches!(program.items[0].kind, ItemKind::Const(_)));
    }

    #[test]
    fn test_parse_type_alias() {
        let program = parse_ok("type Fr = Field<BN254>");
        assert_eq!(program.items.len(), 1);
        assert!(matches!(program.items[0].kind, ItemKind::TypeAlias(_)));
    }
}

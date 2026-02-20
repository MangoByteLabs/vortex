//! Module system for Vortex.
//!
//! Handles import resolution, file loading, and symbol management.
//! Supports: import path.to.module { Symbol1, Symbol2 }
//!           import path.to.module { * }

use crate::ast::*;
use crate::lexer;
use crate::parser;
use std::collections::HashMap;
use std::path::PathBuf;

/// A resolved module with its AST
#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub path: PathBuf,
    pub program: Program,
}

/// Module resolver that manages loaded modules
pub struct ModuleResolver {
    /// Root directory for resolving imports
    root: PathBuf,
    /// Standard library path
    stdlib_path: Option<PathBuf>,
    /// Loaded modules cache: module path string -> Module
    modules: HashMap<String, Module>,
    /// Search paths for module resolution
    search_paths: Vec<PathBuf>,
}

impl ModuleResolver {
    pub fn new(root: PathBuf) -> Self {
        let mut search_paths = vec![root.clone()];

        // Check for standard library location
        let stdlib = root.join("stdlib");
        let stdlib_path = if stdlib.exists() {
            search_paths.push(stdlib.clone());
            Some(stdlib)
        } else {
            None
        };

        Self {
            root,
            stdlib_path,
            modules: HashMap::new(),
            search_paths,
        }
    }

    /// Add a search path for module resolution
    pub fn add_search_path(&mut self, path: PathBuf) {
        self.search_paths.push(path);
    }

    /// Resolve an import path (e.g., ["crypto", "field"]) to a file
    pub fn resolve_path(&self, import_path: &[String]) -> Option<PathBuf> {
        for base in &self.search_paths {
            // Try: base/crypto/field.vx
            let mut file_path = base.clone();
            for segment in import_path {
                file_path = file_path.join(segment);
            }
            let vx_path = file_path.with_extension("vx");
            if vx_path.exists() {
                return Some(vx_path);
            }

            // Try: base/crypto/field/mod.vx
            let mod_path = file_path.join("mod.vx");
            if mod_path.exists() {
                return Some(mod_path);
            }
        }
        None
    }

    /// Load and parse a module
    pub fn load_module(&mut self, import_path: &[String]) -> Result<Module, String> {
        let path_key = import_path.join(".");

        // Check cache
        if let Some(module) = self.modules.get(&path_key) {
            return Ok(module.clone());
        }

        // Resolve file path
        let file_path = self.resolve_path(import_path)
            .ok_or_else(|| format!(
                "cannot find module '{}' (searched: {})",
                path_key,
                self.search_paths.iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))?;

        // Read and parse
        let source = std::fs::read_to_string(&file_path)
            .map_err(|e| format!("cannot read {}: {}", file_path.display(), e))?;

        let tokens = lexer::lex(&source);
        let program = parser::parse(tokens, 0)
            .map_err(|diags| {
                format!("parse errors in {}: {} error(s)",
                    file_path.display(),
                    diags.len())
            })?;

        let module = Module {
            name: path_key.clone(),
            path: file_path,
            program,
        };

        self.modules.insert(path_key, module.clone());
        Ok(module)
    }

    /// Resolve all imports in a program and return the merged program
    pub fn resolve_imports(&mut self, program: &Program) -> Result<Program, String> {
        let mut merged_items: Vec<Item> = Vec::new();

        for item in &program.items {
            if let ItemKind::Import(import) = &item.kind {
                let path: Vec<String> = import.path.iter()
                    .map(|id| id.name.clone())
                    .collect();

                match self.load_module(&path) {
                    Ok(module) => {
                        match &import.items {
                            ImportItems::Wildcard => {
                                // Import all public items
                                for mod_item in &module.program.items {
                                    if mod_item.is_pub {
                                        merged_items.push(mod_item.clone());
                                    }
                                }
                            }
                            ImportItems::Named(names) => {
                                // Import only named items
                                for import_item in names {
                                    let target_name = &import_item.name.name;
                                    for mod_item in &module.program.items {
                                        let item_name = get_item_name(&mod_item.kind);
                                        if item_name.as_deref() == Some(target_name) {
                                            merged_items.push(mod_item.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        // Non-fatal: just warn and continue
                        eprintln!("warning: {}", e);
                    }
                }
            } else {
                merged_items.push(item.clone());
            }
        }

        Ok(Program { items: merged_items })
    }
}

/// Get the name of a top-level item
fn get_item_name(kind: &ItemKind) -> Option<String> {
    match kind {
        ItemKind::Function(f) => Some(f.name.name.clone()),
        ItemKind::Kernel(k) => Some(k.name.name.clone()),
        ItemKind::Struct(s) => Some(s.name.name.clone()),
        ItemKind::Enum(e) => Some(e.name.name.clone()),
        ItemKind::Trait(t) => Some(t.name.name.clone()),
        ItemKind::Const(c) => Some(c.name.name.clone()),
        ItemKind::TypeAlias(t) => Some(t.name.name.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_resolve_path() {
        let dir = std::env::temp_dir().join("vortex_test_modules");
        let _ = fs::create_dir_all(dir.join("crypto"));
        fs::write(dir.join("crypto/field.vx"), "pub fn test_field() -> i64 { return 1 }").unwrap();

        let resolver = ModuleResolver::new(dir.clone());
        let path = resolver.resolve_path(&["crypto".to_string(), "field".to_string()]);
        assert!(path.is_some());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_load_module() {
        let dir = std::env::temp_dir().join("vortex_test_load");
        let _ = fs::create_dir_all(dir.join("math"));
        fs::write(dir.join("math/utils.vx"), "pub fn add(a: i64, b: i64) -> i64 { return a + b }").unwrap();

        let mut resolver = ModuleResolver::new(dir.clone());
        let module = resolver.load_module(&["math".to_string(), "utils".to_string()]);
        assert!(module.is_ok());
        let m = module.unwrap();
        assert_eq!(m.program.items.len(), 1);

        let _ = fs::remove_dir_all(dir);
    }
}

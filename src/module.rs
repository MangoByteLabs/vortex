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
    loading_stack: Vec<String>,
    pub namespaces: HashMap<String, Vec<String>>,
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
            loading_stack: Vec::new(),
            namespaces: HashMap::new(),
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

        // Check for circular dependencies
        if self.loading_stack.contains(&path_key) {
            return Err(format!(
                "circular dependency detected: {} -> {}",
                self.loading_stack.join(" -> "),
                path_key
            ));
        }
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

        self.loading_stack.push(path_key.clone());

        // Read and parse
        let source = std::fs::read_to_string(&file_path)
            .map_err(|e| { self.loading_stack.pop(); format!("cannot read {}: {}", file_path.display(), e) })?;

        let tokens = lexer::lex(&source);
        let program = parser::parse(tokens, 0)
            .map_err(|diags| {
                self.loading_stack.pop();
                format!("parse errors in {}: {} error(s)",
                    file_path.display(),
                    diags.len())
            })?;

        // Recursively resolve imports within this module
        let resolved_program = self.resolve_imports_inner(&program)
            .map_err(|e| { self.loading_stack.pop(); e })?;

        let module = Module {
            name: path_key.clone(),
            path: file_path,
            program: resolved_program,
        };

        self.loading_stack.pop();
        self.modules.insert(path_key, module.clone());
        Ok(module)
    }

    /// Resolve all imports in a program and return the merged program
    pub fn resolve_imports(&mut self, program: &Program) -> Result<Program, String> {
        self.loading_stack.clear();
        self.resolve_imports_inner(program)
    }

    /// Internal import resolution
    fn resolve_imports_inner(&mut self, program: &Program) -> Result<Program, String> {
        let mut merged_items: Vec<Item> = Vec::new();

        for item in &program.items {
            if let ItemKind::Import(import) = &item.kind {
                let path: Vec<String> = import.path.iter()
                    .map(|id| id.name.clone())
                    .collect();

                let module_name = path.last().cloned().unwrap_or_default();
                let is_reexport = item.is_pub;

                match self.load_module(&path) {
                    Ok(module) => {
                        let mut namespace_items = Vec::new();
                        match &import.items {
                            ImportItems::Wildcard => {
                                // Import all public items
                                for mod_item in &module.program.items {
                                    if mod_item.is_pub {
                                        let mut imported = mod_item.clone();
                                        imported.is_pub = is_reexport;
                                        if let Some(name) = get_item_name(&imported.kind) {
                                            namespace_items.push(name);
                                        }
                                        merged_items.push(imported);
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
                                            let mut imported = mod_item.clone();
                                            imported.is_pub = is_reexport;
                                            namespace_items.push(target_name.clone());
                                            merged_items.push(imported);
                                        }
                                    }
                                }
                            }
                        }
                        // Register namespace
                        self.namespaces.entry(module_name.clone()).or_default().extend(namespace_items);
                    }
                    Err(e) => {
                        if e.contains("circular dependency") {
                            return Err(e);
                        }
                        // Non-fatal for other errors: just warn and continue
                        eprintln!("warning: {}", e);
                    }
                }
            } else {
                merged_items.push(item.clone());
            }
        }

        Ok(Program { items: merged_items })
    }

    /// Get namespace registry
    pub fn get_namespaces(&self) -> &HashMap<String, Vec<String>> {
        &self.namespaces
    }
}

/// Rename an item (used for import aliases)
fn rename_item(kind: &mut ItemKind, new_name: &str) {
    match kind {
        ItemKind::Function(f) => f.name.name = new_name.to_string(),
        ItemKind::Kernel(k) => k.name.name = new_name.to_string(),
        ItemKind::Struct(s) => s.name.name = new_name.to_string(),
        ItemKind::Enum(e) => e.name.name = new_name.to_string(),
        ItemKind::Trait(t) => t.name.name = new_name.to_string(),
        ItemKind::Const(c) => c.name.name = new_name.to_string(),
        ItemKind::TypeAlias(t) => t.name.name = new_name.to_string(),
        _ => {}
    }
}

/// Get the name of a top-level item
pub fn get_item_name(kind: &ItemKind) -> Option<String> {
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


    #[test]
    fn test_circular_dependency_detection() {
        let dir = std::env::temp_dir().join("vortex_test_circular");
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::create_dir_all(&dir);
        fs::write(dir.join("a.vx"), "import b { * }\npub fn from_a() -> i64 { return 1 }").unwrap();
        fs::write(dir.join("b.vx"), "import a { * }\npub fn from_b() -> i64 { return 2 }").unwrap();
        let mut resolver = ModuleResolver::new(dir.clone());
        let result = resolver.load_module(&["a".to_string()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("circular dependency"), "expected circular dependency error, got: {}", err);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_qualified_access() {
        let dir = std::env::temp_dir().join("vortex_test_qualified");
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::create_dir_all(&dir);
        fs::write(dir.join("mymod.vx"), "pub fn helper() -> i64 { return 42 }").unwrap();
        fs::write(dir.join("main.vx"), "import mymod { helper }\nfn main() { let x = helper() }").unwrap();
        let mut resolver = ModuleResolver::new(dir.clone());
        let tokens = crate::lexer::lex(&std::fs::read_to_string(dir.join("main.vx")).unwrap());
        let program = crate::parser::parse(tokens, 0).unwrap();
        let _resolved = resolver.resolve_imports(&program).unwrap();
        assert!(resolver.namespaces.contains_key("mymod"));
        assert!(resolver.namespaces["mymod"].contains(&"helper".to_string()));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_stdlib_resolution() {
        let dir = std::env::temp_dir().join("vortex_test_stdlib");
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::create_dir_all(dir.join("stdlib"));
        fs::write(dir.join("stdlib/math.vx"), "pub fn abs(x: i64) -> i64 {\n  if x < 0 { return 0 - x }\n  return x\n}").unwrap();
        let resolver = ModuleResolver::new(dir.clone());
        let path = resolver.resolve_path(&["math".to_string()]);
        assert!(path.is_some(), "stdlib/math.vx should be resolvable");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_reexport() {
        let dir = std::env::temp_dir().join("vortex_test_reexport");
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::create_dir_all(&dir);
        fs::write(dir.join("inner.vx"), "pub fn secret() -> i64 { return 99 }").unwrap();
        fs::write(dir.join("outer.vx"), "pub import inner { secret }").unwrap();
        let mut resolver = ModuleResolver::new(dir.clone());
        let module = resolver.load_module(&["outer".to_string()]);
        assert!(module.is_ok(), "outer module should load");
        let m = module.unwrap();
        assert!(m.program.items.iter().any(|item| item.is_pub && get_item_name(&item.kind).as_deref() == Some("secret")));
        let _ = fs::remove_dir_all(dir);
    }
}

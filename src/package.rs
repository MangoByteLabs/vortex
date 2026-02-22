use std::fs;
use std::path::{Path, PathBuf};

/// A Vortex package descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct VortexPackage {
    pub name: String,
    pub version: String,
    pub description: String,
    pub dependencies: Vec<(String, String)>, // (name, source) where source is git url or path
    pub entry_point: String,                  // main .vx file
}

impl Default for VortexPackage {
    fn default() -> Self {
        VortexPackage {
            name: String::new(),
            version: "0.1.0".to_string(),
            description: String::new(),
            dependencies: Vec::new(),
            entry_point: "main.vx".to_string(),
        }
    }
}

/// Parse a simple TOML-like manifest from a `vortex.toml` file.
///
/// Supported format:
/// ```toml
/// [package]
/// name = "my-project"
/// version = "0.1.0"
/// description = "A cool project"
/// entry_point = "src/main.vx"
///
/// [dependencies]
/// std = { path = "stdlib" }
/// crypto = { git = "https://github.com/example/crypto-vx.git" }
/// ```
pub fn parse_manifest(content: &str) -> Result<VortexPackage, String> {
    let mut pkg = VortexPackage::default();
    let mut section = String::new();

    for (line_num, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Section header
        if line.starts_with('[') && line.ends_with(']') {
            section = line[1..line.len() - 1].trim().to_string();
            continue;
        }

        // Key = value
        let eq_pos = match line.find('=') {
            Some(p) => p,
            None => {
                return Err(format!(
                    "Line {}: expected key = value, got: {}",
                    line_num + 1,
                    line
                ))
            }
        };

        let key = line[..eq_pos].trim();
        let value_raw = line[eq_pos + 1..].trim();

        match section.as_str() {
            "package" => {
                let value = unquote(value_raw)?;
                match key {
                    "name" => pkg.name = value,
                    "version" => pkg.version = value,
                    "description" => pkg.description = value,
                    "entry_point" => pkg.entry_point = value,
                    _ => {} // ignore unknown keys
                }
            }
            "dependencies" => {
                // Parse inline table: name = { path = "..." } or name = { git = "..." }
                let source = parse_dep_value(value_raw)
                    .map_err(|e| format!("Line {}: {}", line_num + 1, e))?;
                pkg.dependencies.push((key.to_string(), source));
            }
            _ => {} // ignore unknown sections
        }
    }

    if pkg.name.is_empty() {
        return Err("Missing required field: [package] name".to_string());
    }

    Ok(pkg)
}

/// Remove surrounding quotes from a string value.
fn unquote(s: &str) -> Result<String, String> {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        Ok(s[1..s.len() - 1].to_string())
    } else {
        // Allow unquoted values
        Ok(s.to_string())
    }
}

/// Parse a dependency value like `{ path = "stdlib" }` or `{ git = "https://..." }`.
fn parse_dep_value(raw: &str) -> Result<String, String> {
    let s = raw.trim();
    if s.starts_with('{') && s.ends_with('}') {
        let inner = s[1..s.len() - 1].trim();
        if let Some(eq) = inner.find('=') {
            let dep_key = inner[..eq].trim();
            let dep_val = unquote(inner[eq + 1..].trim())?;
            match dep_key {
                "path" | "git" => Ok(format!("{}={}", dep_key, dep_val)),
                _ => Err(format!("Unknown dependency type: {}", dep_key)),
            }
        } else {
            Err("Expected key = value inside dependency block".to_string())
        }
    } else {
        // Treat as a quoted string (simple version constraint or path)
        unquote(s)
    }
}

/// Resolves and installs packages.
pub struct PackageResolver {
    packages_dir: PathBuf,
}

impl PackageResolver {
    pub fn new() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        PackageResolver {
            packages_dir: PathBuf::from(home).join(".vortex").join("packages"),
        }
    }

    /// Create the packages directory if it doesn't exist.
    fn ensure_packages_dir(&self) -> Result<(), String> {
        fs::create_dir_all(&self.packages_dir)
            .map_err(|e| format!("Failed to create packages directory: {}", e))
    }

    /// Resolve a dependency to a local filesystem path.
    pub fn resolve_dependency(&self, name: &str, source: &str) -> Result<String, String> {
        if let Some(path) = source.strip_prefix("path=") {
            // Local path dependency -- resolve relative to current dir
            let resolved = PathBuf::from(path);
            if resolved.exists() {
                Ok(resolved.to_string_lossy().to_string())
            } else {
                Ok(path.to_string()) // return as-is, may be resolved later
            }
        } else if let Some(url) = source.strip_prefix("git=") {
            // Git dependency -- clone into packages dir
            self.ensure_packages_dir()?;
            let target = self.packages_dir.join(name);
            if target.exists() {
                Ok(target.to_string_lossy().to_string())
            } else {
                self.git_clone(url, &target)?;
                Ok(target.to_string_lossy().to_string())
            }
        } else {
            Err(format!("Unknown dependency source for '{}': {}", name, source))
        }
    }

    /// Clone a git repository.
    fn git_clone(&self, url: &str, target: &Path) -> Result<(), String> {
        let output = std::process::Command::new("git")
            .args(["clone", "--depth", "1", url])
            .arg(target)
            .output()
            .map_err(|e| format!("Failed to run git clone: {}", e))?;

        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("git clone failed: {}", stderr.trim()))
        }
    }

    /// Install a package from a git URL.
    pub fn install_package(&self, url: &str) -> Result<(), String> {
        self.ensure_packages_dir()?;

        // Derive name from URL: https://github.com/user/repo.git -> repo
        let name = url
            .rsplit('/')
            .next()
            .unwrap_or("package")
            .trim_end_matches(".git");

        let target = self.packages_dir.join(name);
        if target.exists() {
            println!("Package '{}' already installed at {}", name, target.display());
            return Ok(());
        }

        println!("Installing '{}' from {}...", name, url);
        self.git_clone(url, &target)?;
        println!("Installed '{}' to {}", name, target.display());
        Ok(())
    }

    /// List all installed packages.
    pub fn list_installed(&self) -> Vec<VortexPackage> {
        let mut packages = Vec::new();
        let dir = &self.packages_dir;
        if !dir.exists() {
            return packages;
        }

        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let manifest_path = path.join("vortex.toml");
                    if manifest_path.exists() {
                        if let Ok(content) = fs::read_to_string(&manifest_path) {
                            if let Ok(pkg) = parse_manifest(&content) {
                                packages.push(pkg);
                                continue;
                            }
                        }
                    }
                    // No manifest -- create a stub entry from directory name
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    packages.push(VortexPackage {
                        name,
                        version: "unknown".to_string(),
                        ..VortexPackage::default()
                    });
                }
            }
        }

        packages
    }
}

/// Generate a default vortex.toml for `vortex init`.
pub fn generate_manifest(name: &str) -> String {
    format!(
        r#"[package]
name = "{}"
version = "0.1.0"
description = ""
entry_point = "main.vx"

[dependencies]
"#,
        name
    )
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_manifest() {
        let content = r#"
[package]
name = "my-project"
version = "0.1.0"
description = "A test project"

[dependencies]
std = { path = "stdlib" }
"#;
        let pkg = parse_manifest(content).unwrap();
        assert_eq!(pkg.name, "my-project");
        assert_eq!(pkg.version, "0.1.0");
        assert_eq!(pkg.description, "A test project");
        assert_eq!(pkg.dependencies.len(), 1);
        assert_eq!(pkg.dependencies[0].0, "std");
        assert_eq!(pkg.dependencies[0].1, "path=stdlib");
    }

    #[test]
    fn test_parse_manifest_git_dependency() {
        let content = r#"
[package]
name = "crypto-app"
version = "0.2.0"

[dependencies]
crypto = { git = "https://github.com/example/crypto.git" }
"#;
        let pkg = parse_manifest(content).unwrap();
        assert_eq!(pkg.name, "crypto-app");
        assert_eq!(pkg.dependencies.len(), 1);
        assert_eq!(pkg.dependencies[0].0, "crypto");
        assert_eq!(
            pkg.dependencies[0].1,
            "git=https://github.com/example/crypto.git"
        );
    }

    #[test]
    fn test_parse_manifest_missing_name() {
        let content = r#"
[package]
version = "0.1.0"
"#;
        assert!(parse_manifest(content).is_err());
    }

    #[test]
    fn test_parse_manifest_with_comments() {
        let content = r#"
# This is a comment
[package]
name = "commented"
version = "1.0.0"

# Dependencies section
[dependencies]
"#;
        let pkg = parse_manifest(content).unwrap();
        assert_eq!(pkg.name, "commented");
        assert_eq!(pkg.version, "1.0.0");
        assert!(pkg.dependencies.is_empty());
    }

    #[test]
    fn test_resolver_path_dependency() {
        let resolver = PackageResolver::new();
        let result = resolver.resolve_dependency("std", "path=stdlib");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "stdlib");
    }

    #[test]
    fn test_list_installed_empty() {
        let resolver = PackageResolver {
            packages_dir: PathBuf::from("/tmp/vortex_test_nonexistent_dir"),
        };
        let packages = resolver.list_installed();
        assert!(packages.is_empty());
    }

    #[test]
    fn test_generate_manifest() {
        let manifest = generate_manifest("my-cool-project");
        assert!(manifest.contains("name = \"my-cool-project\""));
        assert!(manifest.contains("[package]"));
        assert!(manifest.contains("[dependencies]"));
    }

    #[test]
    fn test_parse_multiple_dependencies() {
        let content = r#"
[package]
name = "multi-dep"
version = "0.1.0"

[dependencies]
std = { path = "stdlib" }
crypto = { git = "https://github.com/example/crypto.git" }
utils = { path = "../shared/utils" }
"#;
        let pkg = parse_manifest(content).unwrap();
        assert_eq!(pkg.dependencies.len(), 3);
        assert_eq!(pkg.dependencies[0].0, "std");
        assert_eq!(pkg.dependencies[1].0, "crypto");
        assert_eq!(pkg.dependencies[2].0, "utils");
    }
}

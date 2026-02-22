//! Package registry client for Vortex.
//!
//! Supports a Git-based registry index (like crates.io's index) where each package
//! has a JSON metadata file. Packages are downloaded from git URLs or tarball URLs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

// ── Semantic Versioning ─────────────────────────────────────────────────

/// A parsed semantic version: MAJOR.MINOR.PATCH
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemVer {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub pre: Option<String>,
}

impl SemVer {
    pub fn new(major: u64, minor: u64, patch: u64) -> Self {
        Self { major, minor, patch, pre: None }
    }

    /// Parse "1.2.3" or "1.2.3-beta.1"
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim().trim_start_matches('v');
        let (version_part, pre) = if let Some(idx) = s.find('-') {
            (&s[..idx], Some(s[idx + 1..].to_string()))
        } else {
            (s, None)
        };
        let parts: Vec<&str> = version_part.split('.').collect();
        if parts.len() < 2 || parts.len() > 3 {
            return Err(format!("invalid semver: {}", s));
        }
        let major = parts[0].parse::<u64>().map_err(|_| format!("invalid major: {}", parts[0]))?;
        let minor = parts[1].parse::<u64>().map_err(|_| format!("invalid minor: {}", parts[1]))?;
        let patch = if parts.len() == 3 {
            parts[2].parse::<u64>().map_err(|_| format!("invalid patch: {}", parts[2]))?
        } else {
            0
        };
        Ok(Self { major, minor, patch, pre })
    }

    /// Compare two versions. Returns Ordering.
    pub fn cmp_version(&self, other: &SemVer) -> std::cmp::Ordering {
        self.major.cmp(&other.major)
            .then(self.minor.cmp(&other.minor))
            .then(self.patch.cmp(&other.patch))
    }

    /// Check if this version satisfies the given constraint.
    pub fn satisfies(&self, constraint: &VersionConstraint) -> bool {
        constraint.matches(self)
    }
}

impl fmt::Display for SemVer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(pre) = &self.pre {
            write!(f, "-{}", pre)?;
        }
        Ok(())
    }
}

impl Ord for SemVer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cmp_version(other)
    }
}

impl PartialOrd for SemVer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ── Version Constraints ─────────────────────────────────────────────────

/// A version constraint like "^1.0", "~1.2", ">=1.0 <2.0", or "=1.2.3"
#[derive(Debug, Clone, PartialEq)]
pub enum VersionConstraint {
    /// ^1.2.3 — compatible with 1.x.y where x >= 2
    Caret(SemVer),
    /// ~1.2.3 — compatible with 1.2.x where x >= 3
    Tilde(SemVer),
    /// Exact match
    Exact(SemVer),
    /// >= version
    Gte(SemVer),
    /// > version
    Gt(SemVer),
    /// <= version
    Lte(SemVer),
    /// < version
    Lt(SemVer),
    /// Intersection of two constraints
    And(Box<VersionConstraint>, Box<VersionConstraint>),
    /// Any version
    Any,
}

impl VersionConstraint {
    /// Parse a version constraint string.
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s == "*" || s.is_empty() {
            return Ok(VersionConstraint::Any);
        }

        // Check for compound constraint: ">=1.0 <2.0" or ">=1.0, <2.0"
        // Split on comma or on space-separated operators
        let parts: Vec<&str> = if s.contains(',') {
            s.split(',').map(|p| p.trim()).collect()
        } else {
            // Try splitting on space where next part starts with operator
            let mut result = Vec::new();
            let mut current = String::new();
            for token in s.split_whitespace() {
                if !current.is_empty() && (token.starts_with('>') || token.starts_with('<') || token.starts_with('=') || token.starts_with('^') || token.starts_with('~')) {
                    result.push(current.trim().to_string());
                    current = token.to_string();
                } else {
                    if !current.is_empty() { current.push(' '); }
                    current.push_str(token);
                }
            }
            if !current.is_empty() { result.push(current); }
            if result.len() > 1 {
                // Parse as compound
                let mut constraint = Self::parse_single(&result[0])?;
                for part in &result[1..] {
                    let right = Self::parse_single(part)?;
                    constraint = VersionConstraint::And(Box::new(constraint), Box::new(right));
                }
                return Ok(constraint);
            }
            vec![s]
        };

        if parts.len() == 1 {
            Self::parse_single(parts[0])
        } else {
            let mut constraint = Self::parse_single(parts[0])?;
            for part in &parts[1..] {
                let right = Self::parse_single(part)?;
                constraint = VersionConstraint::And(Box::new(constraint), Box::new(right));
            }
            Ok(constraint)
        }
    }

    fn parse_single(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s.starts_with('^') {
            Ok(VersionConstraint::Caret(SemVer::parse(&s[1..])?))
        } else if s.starts_with('~') {
            Ok(VersionConstraint::Tilde(SemVer::parse(&s[1..])?))
        } else if s.starts_with(">=") {
            Ok(VersionConstraint::Gte(SemVer::parse(&s[2..])?))
        } else if s.starts_with('>') {
            Ok(VersionConstraint::Gt(SemVer::parse(&s[1..])?))
        } else if s.starts_with("<=") {
            Ok(VersionConstraint::Lte(SemVer::parse(&s[2..])?))
        } else if s.starts_with('<') {
            Ok(VersionConstraint::Lt(SemVer::parse(&s[1..])?))
        } else if s.starts_with('=') {
            Ok(VersionConstraint::Exact(SemVer::parse(&s[1..])?))
        } else {
            // Bare version like "1.2.3" treated as caret
            Ok(VersionConstraint::Caret(SemVer::parse(s)?))
        }
    }

    /// Check if a version matches this constraint.
    pub fn matches(&self, v: &SemVer) -> bool {
        match self {
            VersionConstraint::Any => true,
            VersionConstraint::Exact(req) => v.cmp_version(req) == std::cmp::Ordering::Equal,
            VersionConstraint::Gte(req) => v.cmp_version(req) != std::cmp::Ordering::Less,
            VersionConstraint::Gt(req) => v.cmp_version(req) == std::cmp::Ordering::Greater,
            VersionConstraint::Lte(req) => v.cmp_version(req) != std::cmp::Ordering::Greater,
            VersionConstraint::Lt(req) => v.cmp_version(req) == std::cmp::Ordering::Less,
            VersionConstraint::Caret(req) => {
                // ^1.2.3 means >=1.2.3 <2.0.0 (for major > 0)
                // ^0.2.3 means >=0.2.3 <0.3.0
                // ^0.0.3 means >=0.0.3 <0.0.4
                if v.cmp_version(req) == std::cmp::Ordering::Less {
                    return false;
                }
                if req.major > 0 {
                    v.major == req.major
                } else if req.minor > 0 {
                    v.major == 0 && v.minor == req.minor
                } else {
                    v.major == 0 && v.minor == 0 && v.patch == req.patch
                }
            }
            VersionConstraint::Tilde(req) => {
                // ~1.2.3 means >=1.2.3 <1.3.0
                if v.cmp_version(req) == std::cmp::Ordering::Less {
                    return false;
                }
                v.major == req.major && v.minor == req.minor
            }
            VersionConstraint::And(a, b) => a.matches(v) && b.matches(v),
        }
    }
}

impl fmt::Display for VersionConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VersionConstraint::Any => write!(f, "*"),
            VersionConstraint::Exact(v) => write!(f, "={}", v),
            VersionConstraint::Gte(v) => write!(f, ">={}", v),
            VersionConstraint::Gt(v) => write!(f, ">{}", v),
            VersionConstraint::Lte(v) => write!(f, "<={}", v),
            VersionConstraint::Lt(v) => write!(f, "<{}", v),
            VersionConstraint::Caret(v) => write!(f, "^{}", v),
            VersionConstraint::Tilde(v) => write!(f, "~{}", v),
            VersionConstraint::And(a, b) => write!(f, "{}, {}", a, b),
        }
    }
}

// ── Package Metadata ────────────────────────────────────────────────────

/// Metadata for a package in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageMeta {
    pub name: String,
    pub version: String,
    pub description: String,
    #[serde(default)]
    pub authors: Vec<String>,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub homepage: String,
    #[serde(default)]
    pub repository: String,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub dependencies: HashMap<String, String>,
    /// Git URL to clone the package source
    #[serde(default)]
    pub git_url: String,
    /// Tarball URL for download
    #[serde(default)]
    pub tarball_url: String,
}

/// Registry index entry: one file per package, containing all versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageIndex {
    pub name: String,
    pub versions: Vec<PackageMeta>,
}

// ── Registry ────────────────────────────────────────────────────────────

const DEFAULT_REGISTRY: &str = "https://github.com/MangoByteLabs/vortex-registry";

/// A registry client that reads/writes a Git-based package index.
pub struct Registry {
    /// URL of the registry index repo
    pub registry_url: String,
    /// Local cache directory: ~/.vortex/registry/
    pub cache_dir: PathBuf,
    /// Local index directory: ~/.vortex/registry/index/
    pub index_dir: PathBuf,
    /// Downloaded packages: ~/.vortex/packages/
    pub packages_dir: PathBuf,
}

impl Registry {
    pub fn new() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        let vortex_dir = PathBuf::from(&home).join(".vortex");
        Self {
            registry_url: DEFAULT_REGISTRY.to_string(),
            cache_dir: vortex_dir.join("registry"),
            index_dir: vortex_dir.join("registry").join("index"),
            packages_dir: vortex_dir.join("packages"),
        }
    }

    pub fn with_dirs(cache_dir: PathBuf, packages_dir: PathBuf) -> Self {
        Self {
            registry_url: DEFAULT_REGISTRY.to_string(),
            index_dir: cache_dir.join("index"),
            cache_dir,
            packages_dir,
        }
    }

    /// Ensure all necessary directories exist.
    pub fn ensure_dirs(&self) -> Result<(), String> {
        fs::create_dir_all(&self.cache_dir).map_err(|e| format!("create cache dir: {}", e))?;
        fs::create_dir_all(&self.index_dir).map_err(|e| format!("create index dir: {}", e))?;
        fs::create_dir_all(&self.packages_dir).map_err(|e| format!("create packages dir: {}", e))?;
        Ok(())
    }

    /// Sync the registry index by cloning/pulling the index repo.
    pub fn sync_index(&self) -> Result<(), String> {
        self.ensure_dirs()?;
        if self.index_dir.join(".git").exists() {
            // Pull latest
            let output = std::process::Command::new("git")
                .args(["-C", &self.index_dir.to_string_lossy(), "pull", "--ff-only"])
                .output()
                .map_err(|e| format!("git pull: {}", e))?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("git pull failed: {}", stderr.trim()));
            }
        } else {
            // Clone
            let output = std::process::Command::new("git")
                .args(["clone", "--depth", "1", &self.registry_url, &self.index_dir.to_string_lossy()])
                .output()
                .map_err(|e| format!("git clone: {}", e))?;
            if !output.status.success() {
                // Index repo may not exist yet; create empty index
                fs::create_dir_all(&self.index_dir).map_err(|e| format!("create index: {}", e))?;
            }
        }
        Ok(())
    }

    /// Get the index file path for a package name.
    /// Uses a two-level directory structure like crates.io:
    /// - 1-char names: 1/
    /// - 2-char names: 2/
    /// - 3-char names: 3/{first-char}/
    /// - 4+ char names: {first-two}/{next-two}/
    pub fn index_path(&self, name: &str) -> PathBuf {
        let lower = name.to_lowercase();
        match lower.len() {
            0 => self.index_dir.join("_empty"),
            1 => self.index_dir.join("1").join(&lower),
            2 => self.index_dir.join("2").join(&lower),
            3 => self.index_dir.join("3").join(&lower[..1]).join(&lower),
            _ => self.index_dir.join(&lower[..2]).join(&lower[2..4.min(lower.len())]).join(&lower),
        }
    }

    /// Read package index from the local cache.
    pub fn read_package_index(&self, name: &str) -> Result<PackageIndex, String> {
        let path = self.index_path(name);
        if !path.exists() {
            return Err(format!("package '{}' not found in registry", name));
        }
        let content = fs::read_to_string(&path)
            .map_err(|e| format!("read index for '{}': {}", name, e))?;
        // The file contains one JSON object per line (like crates.io)
        let mut versions = Vec::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() { continue; }
            let meta: PackageMeta = serde_json::from_str(line)
                .map_err(|e| format!("parse index entry for '{}': {}", name, e))?;
            versions.push(meta);
        }
        Ok(PackageIndex { name: name.to_string(), versions })
    }

    /// Write a package version to the index.
    pub fn write_package_index(&self, meta: &PackageMeta) -> Result<(), String> {
        self.ensure_dirs()?;
        let path = self.index_path(&meta.name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("create index dirs: {}", e))?;
        }
        let json = serde_json::to_string(meta).map_err(|e| format!("serialize: {}", e))?;
        // Append line to index file
        let mut content = if path.exists() {
            fs::read_to_string(&path).unwrap_or_default()
        } else {
            String::new()
        };
        if !content.is_empty() && !content.ends_with('\n') {
            content.push('\n');
        }
        content.push_str(&json);
        content.push('\n');
        fs::write(&path, &content).map_err(|e| format!("write index: {}", e))?;
        Ok(())
    }

    /// Search for packages matching a query in name or keywords.
    pub fn search(&self, query: &str) -> Result<Vec<PackageMeta>, String> {
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        // Walk the index directory
        self.walk_index(&self.index_dir, &query_lower, &mut results)?;
        Ok(results)
    }

    fn walk_index(&self, dir: &Path, query: &str, results: &mut Vec<PackageMeta>) -> Result<(), String> {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return Ok(()),
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                self.walk_index(&path, query, results)?;
            } else if path.is_file() {
                if let Ok(content) = fs::read_to_string(&path) {
                    for line in content.lines() {
                        let line = line.trim();
                        if line.is_empty() { continue; }
                        if let Ok(meta) = serde_json::from_str::<PackageMeta>(line) {
                            if meta.name.to_lowercase().contains(query)
                                || meta.description.to_lowercase().contains(query)
                                || meta.keywords.iter().any(|k| k.to_lowercase().contains(query))
                            {
                                results.push(meta);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Fetch metadata for a specific package and version.
    pub fn fetch_metadata(&self, name: &str, version_req: &str) -> Result<PackageMeta, String> {
        let index = self.read_package_index(name)?;
        let constraint = VersionConstraint::parse(version_req)?;

        // Find best matching version (latest that satisfies constraint)
        let mut matching: Vec<&PackageMeta> = index.versions.iter()
            .filter(|m| {
                SemVer::parse(&m.version)
                    .map(|v| constraint.matches(&v))
                    .unwrap_or(false)
            })
            .collect();

        matching.sort_by(|a, b| {
            let va = SemVer::parse(&a.version).unwrap_or(SemVer::new(0, 0, 0));
            let vb = SemVer::parse(&b.version).unwrap_or(SemVer::new(0, 0, 0));
            vb.cmp_version(&va)
        });

        matching.first()
            .cloned()
            .cloned()
            .ok_or_else(|| format!("no version of '{}' matches '{}'", name, version_req))
    }

    /// Download a package to the local packages directory.
    pub fn download_package(&self, meta: &PackageMeta) -> Result<PathBuf, String> {
        self.ensure_dirs()?;
        let pkg_dir = self.packages_dir.join(format!("{}-{}", meta.name, meta.version));
        if pkg_dir.exists() {
            return Ok(pkg_dir);
        }

        if !meta.git_url.is_empty() {
            let output = std::process::Command::new("git")
                .args(["clone", "--depth", "1", &meta.git_url, &pkg_dir.to_string_lossy()])
                .output()
                .map_err(|e| format!("git clone: {}", e))?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("download failed: {}", stderr.trim()));
            }
            Ok(pkg_dir)
        } else if !meta.tarball_url.is_empty() {
            // For tarball: use curl + tar
            fs::create_dir_all(&pkg_dir).map_err(|e| format!("create pkg dir: {}", e))?;
            let output = std::process::Command::new("sh")
                .args(["-c", &format!("curl -sL '{}' | tar xz -C '{}'", meta.tarball_url, pkg_dir.display())])
                .output()
                .map_err(|e| format!("download tarball: {}", e))?;
            if !output.status.success() {
                let _ = fs::remove_dir_all(&pkg_dir);
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("tarball download failed: {}", stderr.trim()));
            }
            Ok(pkg_dir)
        } else {
            Err(format!("package '{}' has no download URL", meta.name))
        }
    }

    /// Publish a package to the registry index.
    pub fn publish(&self, meta: &PackageMeta) -> Result<(), String> {
        // Validate
        if meta.name.is_empty() {
            return Err("package name is required".to_string());
        }
        if meta.version.is_empty() {
            return Err("package version is required".to_string());
        }
        SemVer::parse(&meta.version)?;

        // Check for duplicate version
        if let Ok(index) = self.read_package_index(&meta.name) {
            if index.versions.iter().any(|v| v.version == meta.version) {
                return Err(format!("version {} of '{}' already published", meta.version, meta.name));
            }
        }

        // Write to index
        self.write_package_index(meta)?;

        // In a real implementation, we'd commit and push the index repo
        println!("Published {} v{} to registry", meta.name, meta.version);
        Ok(())
    }
}

// ── Dependency Resolution ───────────────────────────────────────────────

/// A resolved dependency with exact version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedDep {
    pub name: String,
    pub version: String,
    pub source: String, // "registry", "git=<url>", "path=<path>"
}

/// A lock file for reproducible builds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockFile {
    pub packages: Vec<ResolvedDep>,
}

impl LockFile {
    pub fn new() -> Self {
        Self { packages: Vec::new() }
    }

    /// Serialize to TOML-like format.
    pub fn to_string(&self) -> String {
        let mut out = String::new();
        out.push_str("# This file is auto-generated by vortex. Do not edit.\n\n");
        for dep in &self.packages {
            out.push_str(&format!("[[package]]\nname = \"{}\"\nversion = \"{}\"\nsource = \"{}\"\n\n",
                dep.name, dep.version, dep.source));
        }
        out
    }

    /// Parse from the TOML-like lock file format.
    pub fn parse(content: &str) -> Result<Self, String> {
        let mut packages = Vec::new();
        let mut current_name = String::new();
        let mut current_version = String::new();
        let mut current_source = String::new();
        let mut in_package = false;

        for line in content.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() {
                if in_package && !current_name.is_empty() {
                    packages.push(ResolvedDep {
                        name: std::mem::take(&mut current_name),
                        version: std::mem::take(&mut current_version),
                        source: std::mem::take(&mut current_source),
                    });
                    in_package = false;
                }
                continue;
            }
            if line == "[[package]]" {
                if in_package && !current_name.is_empty() {
                    packages.push(ResolvedDep {
                        name: std::mem::take(&mut current_name),
                        version: std::mem::take(&mut current_version),
                        source: std::mem::take(&mut current_source),
                    });
                }
                in_package = true;
                continue;
            }
            if in_package {
                if let Some(val) = line.strip_prefix("name = ") {
                    current_name = val.trim_matches('"').to_string();
                } else if let Some(val) = line.strip_prefix("version = ") {
                    current_version = val.trim_matches('"').to_string();
                } else if let Some(val) = line.strip_prefix("source = ") {
                    current_source = val.trim_matches('"').to_string();
                }
            }
        }
        if in_package && !current_name.is_empty() {
            packages.push(ResolvedDep {
                name: current_name,
                version: current_version,
                source: current_source,
            });
        }
        Ok(Self { packages })
    }
}

/// Resolve all dependencies (including transitive) for a manifest.
pub fn resolve_dependencies(
    deps: &[(String, String)],
    registry: &Registry,
) -> Result<LockFile, String> {
    let mut lock = LockFile::new();
    let mut resolved: HashMap<String, SemVer> = HashMap::new();
    let mut queue: Vec<(String, String)> = deps.to_vec();

    while let Some((name, version_req)) = queue.pop() {
        // Skip if already resolved
        if let Some(existing) = resolved.get(&name) {
            // Check compatibility
            let constraint = VersionConstraint::parse(&version_req)?;
            if !constraint.matches(existing) {
                return Err(format!(
                    "version conflict for '{}': resolved {} but '{}' requires {}",
                    name, existing, name, version_req
                ));
            }
            continue;
        }

        // Check if it's a path or git dep (from old format)
        if version_req.starts_with("path=") || version_req.starts_with("git=") {
            let version = SemVer::new(0, 0, 0);
            resolved.insert(name.clone(), version.clone());
            lock.packages.push(ResolvedDep {
                name,
                version: version.to_string(),
                source: version_req,
            });
            continue;
        }

        // Try to resolve from registry
        match registry.fetch_metadata(&name, &version_req) {
            Ok(meta) => {
                let version = SemVer::parse(&meta.version)?;
                resolved.insert(name.clone(), version);
                let source = if !meta.git_url.is_empty() {
                    format!("git={}", meta.git_url)
                } else if !meta.tarball_url.is_empty() {
                    format!("tarball={}", meta.tarball_url)
                } else {
                    "registry".to_string()
                };
                lock.packages.push(ResolvedDep {
                    name: name.clone(),
                    version: meta.version.clone(),
                    source,
                });
                // Enqueue transitive deps
                for (dep_name, dep_ver) in &meta.dependencies {
                    queue.push((dep_name.clone(), dep_ver.clone()));
                }
            }
            Err(_) => {
                // If not in registry, treat version_req as a version and add as-is
                let version = SemVer::parse(&version_req).unwrap_or(SemVer::new(0, 0, 0));
                resolved.insert(name.clone(), version.clone());
                lock.packages.push(ResolvedDep {
                    name,
                    version: version.to_string(),
                    source: "local".to_string(),
                });
            }
        }
    }

    // Sort for determinism
    lock.packages.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(lock)
}

// ── CLI helpers ─────────────────────────────────────────────────────────

/// Add a dependency to the vortex.toml content string.
pub fn add_dependency_to_manifest(content: &str, name: &str, version: &str) -> String {
    let mut lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
    // Find [dependencies] section
    let dep_idx = lines.iter().position(|l| l.trim() == "[dependencies]");
    if let Some(idx) = dep_idx {
        // Insert after the [dependencies] header
        lines.insert(idx + 1, format!("{} = \"{}\"", name, version));
    } else {
        // Add [dependencies] section
        lines.push(String::new());
        lines.push("[dependencies]".to_string());
        lines.push(format!("{} = \"{}\"", name, version));
    }
    lines.join("\n") + "\n"
}

/// Remove a dependency from the vortex.toml content string.
pub fn remove_dependency_from_manifest(content: &str, name: &str) -> Result<String, String> {
    let mut lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
    let mut found = false;
    lines.retain(|l| {
        let trimmed = l.trim();
        if trimmed.starts_with(&format!("{} ", name)) || trimmed.starts_with(&format!("{}=", name)) {
            found = true;
            false
        } else {
            true
        }
    });
    if !found {
        return Err(format!("dependency '{}' not found in manifest", name));
    }
    Ok(lines.join("\n") + "\n")
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semver_parse() {
        let v = SemVer::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
        assert_eq!(v.to_string(), "1.2.3");
    }

    #[test]
    fn test_semver_parse_two_parts() {
        let v = SemVer::parse("1.0").unwrap();
        assert_eq!(v, SemVer::new(1, 0, 0));
    }

    #[test]
    fn test_semver_parse_prerelease() {
        let v = SemVer::parse("1.0.0-beta.1").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.pre, Some("beta.1".to_string()));
    }

    #[test]
    fn test_semver_ordering() {
        let v1 = SemVer::parse("1.0.0").unwrap();
        let v2 = SemVer::parse("1.0.1").unwrap();
        let v3 = SemVer::parse("1.1.0").unwrap();
        let v4 = SemVer::parse("2.0.0").unwrap();
        assert!(v1 < v2);
        assert!(v2 < v3);
        assert!(v3 < v4);
    }

    #[test]
    fn test_caret_constraint() {
        let c = VersionConstraint::parse("^1.2.3").unwrap();
        assert!(c.matches(&SemVer::parse("1.2.3").unwrap()));
        assert!(c.matches(&SemVer::parse("1.9.0").unwrap()));
        assert!(!c.matches(&SemVer::parse("2.0.0").unwrap()));
        assert!(!c.matches(&SemVer::parse("1.2.2").unwrap()));
    }

    #[test]
    fn test_tilde_constraint() {
        let c = VersionConstraint::parse("~1.2.3").unwrap();
        assert!(c.matches(&SemVer::parse("1.2.3").unwrap()));
        assert!(c.matches(&SemVer::parse("1.2.9").unwrap()));
        assert!(!c.matches(&SemVer::parse("1.3.0").unwrap()));
    }

    #[test]
    fn test_range_constraint() {
        let c = VersionConstraint::parse(">=1.0.0 <2.0.0").unwrap();
        assert!(c.matches(&SemVer::parse("1.0.0").unwrap()));
        assert!(c.matches(&SemVer::parse("1.5.0").unwrap()));
        assert!(!c.matches(&SemVer::parse("2.0.0").unwrap()));
        assert!(!c.matches(&SemVer::parse("0.9.0").unwrap()));
    }

    #[test]
    fn test_exact_constraint() {
        let c = VersionConstraint::parse("=1.2.3").unwrap();
        assert!(c.matches(&SemVer::parse("1.2.3").unwrap()));
        assert!(!c.matches(&SemVer::parse("1.2.4").unwrap()));
    }

    #[test]
    fn test_wildcard_constraint() {
        let c = VersionConstraint::parse("*").unwrap();
        assert!(c.matches(&SemVer::parse("0.0.1").unwrap()));
        assert!(c.matches(&SemVer::parse("99.99.99").unwrap()));
    }

    #[test]
    fn test_caret_zero_major() {
        let c = VersionConstraint::parse("^0.2.0").unwrap();
        assert!(c.matches(&SemVer::parse("0.2.0").unwrap()));
        assert!(c.matches(&SemVer::parse("0.2.5").unwrap()));
        assert!(!c.matches(&SemVer::parse("0.3.0").unwrap()));
    }

    #[test]
    fn test_registry_index_path() {
        let reg = Registry::with_dirs(PathBuf::from("/tmp/test_cache"), PathBuf::from("/tmp/test_pkgs"));
        assert!(reg.index_path("a").ends_with("1/a"));
        assert!(reg.index_path("ab").ends_with("2/ab"));
        assert!(reg.index_path("abc").ends_with("3/a/abc"));
        assert!(reg.index_path("abcd").ends_with("ab/cd/abcd"));
        assert!(reg.index_path("serde_json").ends_with("se/rd/serde_json"));
    }

    #[test]
    fn test_write_and_read_index() {
        let dir = std::env::temp_dir().join("vortex_test_registry_rw");
        let _ = fs::remove_dir_all(&dir);
        let reg = Registry::with_dirs(dir.join("cache"), dir.join("pkgs"));
        reg.ensure_dirs().unwrap();

        let meta = PackageMeta {
            name: "testpkg".to_string(),
            version: "1.0.0".to_string(),
            description: "A test package".to_string(),
            authors: vec!["Test".to_string()],
            license: "MIT".to_string(),
            homepage: String::new(),
            repository: String::new(),
            keywords: vec!["test".to_string()],
            dependencies: HashMap::new(),
            git_url: "https://github.com/test/testpkg.git".to_string(),
            tarball_url: String::new(),
        };
        reg.write_package_index(&meta).unwrap();

        let index = reg.read_package_index("testpkg").unwrap();
        assert_eq!(index.versions.len(), 1);
        assert_eq!(index.versions[0].name, "testpkg");
        assert_eq!(index.versions[0].version, "1.0.0");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_search_packages() {
        let dir = std::env::temp_dir().join("vortex_test_registry_search");
        let _ = fs::remove_dir_all(&dir);
        let reg = Registry::with_dirs(dir.join("cache"), dir.join("pkgs"));
        reg.ensure_dirs().unwrap();

        for (name, desc) in [("math-utils", "Math utilities"), ("crypto-lib", "Crypto library"), ("nn-helpers", "Neural net")] {
            reg.write_package_index(&PackageMeta {
                name: name.to_string(),
                version: "1.0.0".to_string(),
                description: desc.to_string(),
                keywords: vec![name.split('-').next().unwrap().to_string()],
                ..Default::default()
            }).unwrap();
        }

        let results = reg.search("math").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "math-utils");

        let results = reg.search("lib").unwrap();
        assert!(results.iter().any(|r| r.name == "crypto-lib"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_fetch_metadata_with_constraint() {
        let dir = std::env::temp_dir().join("vortex_test_registry_fetch");
        let _ = fs::remove_dir_all(&dir);
        let reg = Registry::with_dirs(dir.join("cache"), dir.join("pkgs"));
        reg.ensure_dirs().unwrap();

        for ver in ["1.0.0", "1.1.0", "1.2.0", "2.0.0"] {
            reg.write_package_index(&PackageMeta {
                name: "mypkg".to_string(),
                version: ver.to_string(),
                description: "test".to_string(),
                ..Default::default()
            }).unwrap();
        }

        let meta = reg.fetch_metadata("mypkg", "^1.0").unwrap();
        assert_eq!(meta.version, "1.2.0"); // Latest 1.x

        let meta = reg.fetch_metadata("mypkg", "~1.1").unwrap();
        assert_eq!(meta.version, "1.1.0"); // Only 1.1.x

        let meta = reg.fetch_metadata("mypkg", ">=2.0.0").unwrap();
        assert_eq!(meta.version, "2.0.0");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_publish_duplicate_version() {
        let dir = std::env::temp_dir().join("vortex_test_registry_dup");
        let _ = fs::remove_dir_all(&dir);
        let reg = Registry::with_dirs(dir.join("cache"), dir.join("pkgs"));
        reg.ensure_dirs().unwrap();

        let meta = PackageMeta {
            name: "dup".to_string(),
            version: "1.0.0".to_string(),
            description: "test".to_string(),
            ..Default::default()
        };
        reg.publish(&meta).unwrap();
        let result = reg.publish(&meta);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already published"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_lock_file_roundtrip() {
        let mut lock = LockFile::new();
        lock.packages.push(ResolvedDep {
            name: "math".to_string(),
            version: "1.0.0".to_string(),
            source: "registry".to_string(),
        });
        lock.packages.push(ResolvedDep {
            name: "crypto".to_string(),
            version: "0.2.0".to_string(),
            source: "git=https://example.com/crypto.git".to_string(),
        });

        let serialized = lock.to_string();
        let parsed = LockFile::parse(&serialized).unwrap();
        assert_eq!(parsed.packages.len(), 2);
        assert_eq!(parsed.packages[0].name, "math");
        assert_eq!(parsed.packages[1].name, "crypto");
        assert_eq!(parsed.packages[1].source, "git=https://example.com/crypto.git");
    }

    #[test]
    fn test_resolve_dependencies_with_registry() {
        let dir = std::env::temp_dir().join("vortex_test_resolve");
        let _ = fs::remove_dir_all(&dir);
        let reg = Registry::with_dirs(dir.join("cache"), dir.join("pkgs"));
        reg.ensure_dirs().unwrap();

        // Publish "base" with no deps
        reg.write_package_index(&PackageMeta {
            name: "base".to_string(),
            version: "1.0.0".to_string(),
            description: "base lib".to_string(),
            ..Default::default()
        }).unwrap();

        // Publish "mid" depending on base
        let mut mid_deps = HashMap::new();
        mid_deps.insert("base".to_string(), "^1.0".to_string());
        reg.write_package_index(&PackageMeta {
            name: "mid".to_string(),
            version: "2.0.0".to_string(),
            description: "mid lib".to_string(),
            dependencies: mid_deps,
            ..Default::default()
        }).unwrap();

        // Resolve from top-level deps
        let deps = vec![("mid".to_string(), "^2.0".to_string())];
        let lock = resolve_dependencies(&deps, &reg).unwrap();
        assert_eq!(lock.packages.len(), 2);
        assert!(lock.packages.iter().any(|p| p.name == "mid" && p.version == "2.0.0"));
        assert!(lock.packages.iter().any(|p| p.name == "base" && p.version == "1.0.0"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_add_dependency_to_manifest() {
        let content = "[package]\nname = \"test\"\nversion = \"0.1.0\"\n\n[dependencies]\n";
        let result = add_dependency_to_manifest(content, "math", "^1.0");
        assert!(result.contains("math = \"^1.0\""));
    }

    #[test]
    fn test_remove_dependency_from_manifest() {
        let content = "[package]\nname = \"test\"\n\n[dependencies]\nmath = \"^1.0\"\ncrypto = \"^0.2\"\n";
        let result = remove_dependency_from_manifest(content, "math").unwrap();
        assert!(!result.contains("math"));
        assert!(result.contains("crypto"));
    }

    #[test]
    fn test_remove_nonexistent_dependency() {
        let content = "[dependencies]\nmath = \"^1.0\"\n";
        let result = remove_dependency_from_manifest(content, "nonexistent");
        assert!(result.is_err());
    }
}

impl Default for PackageMeta {
    fn default() -> Self {
        Self {
            name: String::new(),
            version: String::new(),
            description: String::new(),
            authors: Vec::new(),
            license: String::new(),
            homepage: String::new(),
            repository: String::new(),
            keywords: Vec::new(),
            dependencies: HashMap::new(),
            git_url: String::new(),
            tarball_url: String::new(),
        }
    }
}

//! MLIR compilation pipeline for Vortex.
//!
//! Orchestrates the full compilation flow:
//! 1. Vortex source -> Parse -> AST
//! 2. AST -> MLIR IR (via codegen)
//! 3. MLIR IR -> mlir-opt (optimization passes)
//! 4. Optimized MLIR -> mlir-translate (to LLVM IR)
//! 5. LLVM IR -> llc (to PTX or native object)
//!
//! Falls back gracefully when tools are not installed.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::fs;

/// Compilation target
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Target {
    /// NVIDIA GPU via PTX
    NvidiaPTX,
    /// AMD GPU via AMDGCN
    AmdGCN,
    /// LLVM IR (portable)
    LLVMIR,
    /// Native CPU object
    NativeObj,
    /// Textual MLIR (no external tools needed)
    MLIR,
}

impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Target::NvidiaPTX => write!(f, "nvptx64"),
            Target::AmdGCN => write!(f, "amdgcn"),
            Target::LLVMIR => write!(f, "llvm-ir"),
            Target::NativeObj => write!(f, "native"),
            Target::MLIR => write!(f, "mlir"),
        }
    }
}

/// Result of a compilation step
#[derive(Debug)]
pub struct CompileResult {
    pub stage: String,
    pub output_path: PathBuf,
    pub output_content: Option<String>,
}

/// Pipeline configuration
pub struct PipelineConfig {
    pub target: Target,
    pub opt_level: u8,     // 0-3
    pub output_dir: PathBuf,
    pub verbose: bool,
    pub gpu_arch: String,  // e.g., "sm_80" for A100, "gfx90a" for MI250
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            target: Target::MLIR,
            opt_level: 2,
            output_dir: PathBuf::from("."),
            verbose: false,
            gpu_arch: "sm_80".to_string(),
        }
    }
}

/// Check if an external tool is available
fn tool_available(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run an external tool and return its output
fn run_tool(name: &str, args: &[&str], input_file: &Path, verbose: bool) -> Result<String, String> {
    if verbose {
        eprintln!("[pipeline] {} {}", name, args.join(" "));
    }

    let output = Command::new(name)
        .args(args)
        .arg(input_file)
        .output()
        .map_err(|e| format!("failed to run {}: {}", name, e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("{} failed: {}", name, stderr));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// The compilation pipeline
pub struct Pipeline {
    config: PipelineConfig,
}

impl Pipeline {
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// Check which tools are available
    pub fn check_toolchain(&self) -> Vec<(String, bool)> {
        let tools = vec![
            "mlir-opt",
            "mlir-translate",
            "llc",
            "ptxas",
            "clang",
        ];

        tools.iter()
            .map(|t| (t.to_string(), tool_available(t)))
            .collect()
    }

    /// Get the optimization passes for the current target
    pub fn get_passes_for_target(&self) -> Vec<String> {
        self.get_mlir_opt_args()
    }

    /// Run the full compilation pipeline
    pub fn compile(&self, mlir_ir: &str, source_name: &str) -> Result<Vec<CompileResult>, String> {
        let mut results = Vec::new();
        let base_name = Path::new(source_name)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        // Step 1: Write MLIR IR to file
        let mlir_path = self.config.output_dir.join(format!("{}.mlir", base_name));
        fs::write(&mlir_path, mlir_ir)
            .map_err(|e| format!("failed to write MLIR: {}", e))?;

        results.push(CompileResult {
            stage: "codegen".to_string(),
            output_path: mlir_path.clone(),
            output_content: Some(mlir_ir.to_string()),
        });

        if self.config.target == Target::MLIR {
            return Ok(results);
        }

        // Step 2: mlir-opt (optimize MLIR)
        if tool_available("mlir-opt") {
            let opt_path = self.config.output_dir.join(format!("{}.opt.mlir", base_name));
            let opt_args = self.get_mlir_opt_args();
            let opt_args_refs: Vec<&str> = opt_args.iter().map(|s| s.as_str()).collect();

            match run_tool("mlir-opt", &opt_args_refs, &mlir_path, self.config.verbose) {
                Ok(output) => {
                    fs::write(&opt_path, &output)
                        .map_err(|e| format!("failed to write optimized MLIR: {}", e))?;
                    results.push(CompileResult {
                        stage: "mlir-opt".to_string(),
                        output_path: opt_path.clone(),
                        output_content: Some(output),
                    });
                }
                Err(e) => {
                    if self.config.verbose {
                        eprintln!("[pipeline] mlir-opt failed, continuing with unoptimized IR: {}", e);
                    }
                }
            }
        }

        // Step 3: mlir-translate (MLIR -> LLVM IR)
        let llvm_ir_path = self.config.output_dir.join(format!("{}.ll", base_name));
        if tool_available("mlir-translate") {
            let translate_input = if results.len() > 1 {
                &results.last().unwrap().output_path
            } else {
                &mlir_path
            };

            match run_tool(
                "mlir-translate",
                &["--mlir-to-llvmir", "-o", llvm_ir_path.to_str().unwrap()],
                translate_input,
                self.config.verbose,
            ) {
                Ok(_) => {
                    let content = fs::read_to_string(&llvm_ir_path).ok();
                    results.push(CompileResult {
                        stage: "mlir-translate".to_string(),
                        output_path: llvm_ir_path.clone(),
                        output_content: content,
                    });
                }
                Err(e) => {
                    if self.config.verbose {
                        eprintln!("[pipeline] mlir-translate failed: {}", e);
                    }
                    return Ok(results);
                }
            }
        } else {
            if self.config.verbose {
                eprintln!("[pipeline] mlir-translate not found, stopping at MLIR stage");
            }
            return Ok(results);
        }

        // Step 4: Target-specific lowering
        match self.config.target {
            Target::NvidiaPTX => {
                self.compile_to_ptx(&llvm_ir_path, &base_name, &mut results)?;
            }
            Target::AmdGCN => {
                self.compile_to_amdgcn(&llvm_ir_path, &base_name, &mut results)?;
            }
            Target::LLVMIR => {
                // Already have LLVM IR, done
            }
            Target::NativeObj => {
                self.compile_to_native(&llvm_ir_path, &base_name, &mut results)?;
            }
            Target::MLIR => unreachable!(),
        }

        Ok(results)
    }

    fn get_mlir_opt_args(&self) -> Vec<String> {
        let mut args = Vec::new();

        // GPU-specific passes
        if self.config.target == Target::NvidiaPTX || self.config.target == Target::AmdGCN {
            args.push("--gpu-kernel-outlining".to_string());
        }

        // Bufferization and linalg passes
        args.push("--normalize-memrefs".to_string());
        args.push("--one-shot-bufferize".to_string());
        args.push("--convert-linalg-to-loops".to_string());
        args.push("--buffer-deallocation".to_string());

        // Standard optimization pipeline
        args.push("--convert-scf-to-cf".to_string());
        args.push("--convert-arith-to-llvm".to_string());
        args.push("--convert-func-to-llvm".to_string());
        args.push("--convert-cf-to-llvm".to_string());
        args.push("--reconcile-unrealized-casts".to_string());

        if self.config.opt_level >= 1 {
            args.push("--canonicalize".to_string());
            args.push("--cse".to_string());
        }

        // Target-specific GPU lowering
        match self.config.target {
            Target::NvidiaPTX => {
                args.push("--gpu-to-nvvm".to_string());
            }
            Target::AmdGCN => {
                args.push("--gpu-to-rocdl".to_string());
            }
            _ => {}
        }

        args
    }

    fn compile_to_ptx(
        &self,
        llvm_ir: &Path,
        base_name: &str,
        results: &mut Vec<CompileResult>,
    ) -> Result<(), String> {
        let ptx_path = self.config.output_dir.join(format!("{}.ptx", base_name));

        if tool_available("llc") {
            let arch = &self.config.gpu_arch;
            match run_tool(
                "llc",
                &[
                    "-march=nvptx64",
                    &format!("-mcpu={}", arch),
                    "-o", ptx_path.to_str().unwrap(),
                ],
                llvm_ir,
                self.config.verbose,
            ) {
                Ok(_) => {
                    let content = fs::read_to_string(&ptx_path).ok();
                    results.push(CompileResult {
                        stage: "llc (PTX)".to_string(),
                        output_path: ptx_path.clone(),
                        output_content: content,
                    });
                }
                Err(e) => return Err(format!("PTX compilation failed: {}", e)),
            }

            // Optional: assemble PTX to cubin
            if tool_available("ptxas") {
                let cubin_path = self.config.output_dir.join(format!("{}.cubin", base_name));
                match run_tool(
                    "ptxas",
                    &[
                        &format!("-arch={}", arch),
                        "-o", cubin_path.to_str().unwrap(),
                    ],
                    &ptx_path,
                    self.config.verbose,
                ) {
                    Ok(_) => {
                        results.push(CompileResult {
                            stage: "ptxas (cubin)".to_string(),
                            output_path: cubin_path,
                            output_content: None,
                        });
                    }
                    Err(e) => {
                        if self.config.verbose {
                            eprintln!("[pipeline] ptxas failed: {}", e);
                        }
                    }
                }
            }
        } else {
            return Err("llc not found — install LLVM to compile to PTX".to_string());
        }

        Ok(())
    }

    fn compile_to_amdgcn(
        &self,
        llvm_ir: &Path,
        base_name: &str,
        results: &mut Vec<CompileResult>,
    ) -> Result<(), String> {
        let asm_path = self.config.output_dir.join(format!("{}.s", base_name));

        if tool_available("llc") {
            let arch = &self.config.gpu_arch;
            match run_tool(
                "llc",
                &[
                    "-march=amdgcn",
                    &format!("-mcpu={}", arch),
                    "-o", asm_path.to_str().unwrap(),
                ],
                llvm_ir,
                self.config.verbose,
            ) {
                Ok(_) => {
                    let content = fs::read_to_string(&asm_path).ok();
                    results.push(CompileResult {
                        stage: "llc (AMDGCN)".to_string(),
                        output_path: asm_path,
                        output_content: content,
                    });
                }
                Err(e) => return Err(format!("AMDGCN compilation failed: {}", e)),
            }
        } else {
            return Err("llc not found — install LLVM to compile to AMDGCN".to_string());
        }

        Ok(())
    }

    fn compile_to_native(
        &self,
        llvm_ir: &Path,
        base_name: &str,
        results: &mut Vec<CompileResult>,
    ) -> Result<(), String> {
        let obj_path = self.config.output_dir.join(format!("{}.o", base_name));

        if tool_available("llc") {
            match run_tool(
                "llc",
                &[
                    "-filetype=obj",
                    "-o", obj_path.to_str().unwrap(),
                ],
                llvm_ir,
                self.config.verbose,
            ) {
                Ok(_) => {
                    results.push(CompileResult {
                        stage: "llc (native obj)".to_string(),
                        output_path: obj_path,
                        output_content: None,
                    });
                }
                Err(e) => return Err(format!("native compilation failed: {}", e)),
            }
        } else {
            return Err("llc not found — install LLVM to compile to native object".to_string());
        }

        Ok(())
    }
}

/// Print pipeline status and available tools
pub fn print_toolchain_status() {
    let pipeline = Pipeline::new(PipelineConfig::default());
    let tools = pipeline.check_toolchain();

    println!("Vortex Compilation Toolchain Status:");
    println!("====================================");
    for (tool, available) in &tools {
        let status = if *available { "OK" } else { "NOT FOUND" };
        println!("  {:<20} {}", tool, status);
    }
    println!();

    let all_available = tools.iter().all(|(_, a)| *a);
    if all_available {
        println!("Full GPU compilation pipeline available.");
    } else {
        println!("Some tools missing. Install LLVM/MLIR for full GPU compilation.");
        println!("Vortex can still: lex, parse, check, run (interpret), codegen (emit MLIR)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_display() {
        assert_eq!(format!("{}", Target::NvidiaPTX), "nvptx64");
        assert_eq!(format!("{}", Target::AmdGCN), "amdgcn");
        assert_eq!(format!("{}", Target::MLIR), "mlir");
    }

    #[test]
    fn test_pipeline_mlir_target() {
        let config = PipelineConfig {
            target: Target::MLIR,
            output_dir: std::env::temp_dir(),
            ..PipelineConfig::default()
        };
        let pipeline = Pipeline::new(config);
        let results = pipeline.compile("module { }", "test").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].stage, "codegen");
    }

    #[test]
    fn test_check_toolchain() {
        let pipeline = Pipeline::new(PipelineConfig::default());
        let tools = pipeline.check_toolchain();
        assert!(tools.len() >= 3);
    }

    #[test]
    fn test_gpu_optimization_passes_nvidia() {
        let config = PipelineConfig {
            target: Target::NvidiaPTX,
            ..PipelineConfig::default()
        };
        let pipeline = Pipeline::new(config);
        let passes = pipeline.get_passes_for_target();
        assert!(passes.contains(&"--gpu-kernel-outlining".to_string()));
        assert!(passes.contains(&"--gpu-to-nvvm".to_string()));
        assert!(passes.contains(&"--normalize-memrefs".to_string()));
    }

    #[test]
    fn test_gpu_optimization_passes_amd() {
        let config = PipelineConfig {
            target: Target::AmdGCN,
            ..PipelineConfig::default()
        };
        let pipeline = Pipeline::new(config);
        let passes = pipeline.get_passes_for_target();
        assert!(passes.contains(&"--gpu-kernel-outlining".to_string()));
        assert!(passes.contains(&"--gpu-to-rocdl".to_string()));
    }

    #[test]
    fn test_gpu_optimization_passes_mlir_no_outlining() {
        let config = PipelineConfig {
            target: Target::MLIR,
            ..PipelineConfig::default()
        };
        let pipeline = Pipeline::new(config);
        let passes = pipeline.get_passes_for_target();
        assert!(!passes.contains(&"--gpu-kernel-outlining".to_string()));
    }
}

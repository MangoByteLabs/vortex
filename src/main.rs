mod architectures;
mod ast;
mod bytecarry;
mod autodiff;
mod tensor_autodiff;
mod codegen;
mod distributed;
mod ctcheck;
mod fusion;
pub mod crypto;
mod dyntensor;
mod fields;
mod interpreter;
mod lexer;
mod local_learn;
mod memory;
pub mod modmath;
mod module;
mod msm;
mod ntt;
mod ode;
mod pairing;
mod poly;
mod parser;
mod pipeline;
mod quantize;
mod sparse;
mod spiking;
mod ssm;
mod runtime;
mod gpu_runtime;
mod typeck;

use codespan_reporting::files::SimpleFiles;
use codespan_reporting::term;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: vortex <command> [file.vx] [options]");
        eprintln!("Commands: run, check, parse, lex, codegen, compile, toolchain");
        std::process::exit(1);
    }

    let command = &args[1];

    // Commands that don't need a file argument
    if command == "toolchain" {
        pipeline::print_toolchain_status();
        return;
    }

    if args.len() < 3 {
        eprintln!("Usage: vortex {} <file.vx>", command);
        std::process::exit(1);
    }

    let filename = &args[2];

    let source = match fs::read_to_string(filename) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading {}: {}", filename, e);
            std::process::exit(1);
        }
    };

    let mut files = SimpleFiles::new();
    let file_id = files.add(filename.clone(), source.clone());

    let writer = StandardStream::stderr(ColorChoice::Auto);
    let config = term::Config::default();

    match command.as_str() {
        "lex" => {
            let tokens = lexer::lex(&source);
            for tok in &tokens {
                println!("{:?}", tok);
            }
            println!("\n{} tokens", tokens.len());
        }
        "parse" => {
            let tokens = lexer::lex(&source);
            match parser::parse(tokens, file_id) {
                Ok(program) => {
                    println!("{}", program);
                }
                Err(diagnostics) => {
                    for diag in &diagnostics {
                        term::emit(&mut writer.lock(), &config, &files, diag).unwrap();
                    }
                    std::process::exit(1);
                }
            }
        }
        "check" => {
            let tokens = lexer::lex(&source);
            match parser::parse(tokens, file_id) {
                Ok(program) => {
                    let fd = PathBuf::from(filename).parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));
                    let mut resolver = module::ModuleResolver::new(fd);
                    let program = match resolver.resolve_imports(&program) { Ok(p)=>p, Err(e)=>{eprintln!("Import error: {}",e);std::process::exit(1);} };
                    match typeck::check(&program, file_id) {
                    Ok(()) => {
                        println!("Type check passed.");
                    }
                    Err(diagnostics) => {
                        for diag in &diagnostics {
                            term::emit(&mut writer.lock(), &config, &files, diag).unwrap();
                        }
                        std::process::exit(1);
                    }
                    }
                }
                Err(diagnostics) => {
                    for diag in &diagnostics {
                        term::emit(&mut writer.lock(), &config, &files, diag).unwrap();
                    }
                    std::process::exit(1);
                }
            }
        }
        "codegen" | "emit" => {
            let tokens = lexer::lex(&source);
            match parser::parse(tokens, file_id) {
                Ok(program) => {
                    let mlir = codegen::generate_mlir(&program);
                    print!("{}", mlir);
                }
                Err(diagnostics) => {
                    for diag in &diagnostics {
                        term::emit(&mut writer.lock(), &config, &files, diag).unwrap();
                    }
                    std::process::exit(1);
                }
            }
        }
        "compile" => {
            let tokens = lexer::lex(&source);
            match parser::parse(tokens, file_id) {
                Ok(program) => {
                    // Resolve imports
                    let file_dir = PathBuf::from(filename)
                        .parent()
                        .map(|p| p.to_path_buf())
                        .unwrap_or_else(|| PathBuf::from("."));
                    let mut resolver = module::ModuleResolver::new(file_dir);
                    let program = match resolver.resolve_imports(&program) {
                        Ok(p) => p,
                        Err(e) => {
                            eprintln!("Import error: {}", e);
                            std::process::exit(1);
                        }
                    };

                    // Generate MLIR
                    let mlir = codegen::generate_mlir(&program);

                    // Determine target from args
                    let target = if args.len() > 3 {
                        match args[3].as_str() {
                            "ptx" | "nvptx" | "cuda" => pipeline::Target::NvidiaPTX,
                            "amdgcn" | "amd" | "hip" => pipeline::Target::AmdGCN,
                            "llvm" | "llvm-ir" => pipeline::Target::LLVMIR,
                            "native" | "obj" => pipeline::Target::NativeObj,
                            "mlir" => pipeline::Target::MLIR,
                            _ => {
                                eprintln!("Unknown target: {}. Options: ptx, amdgcn, llvm, native, mlir", args[3]);
                                std::process::exit(1);
                            }
                        }
                    } else {
                        pipeline::Target::MLIR
                    };

                    let gpu_arch = if args.len() > 4 {
                        args[4].clone()
                    } else {
                        "sm_80".to_string()
                    };

                    let config = pipeline::PipelineConfig {
                        target,
                        opt_level: 2,
                        output_dir: PathBuf::from("."),
                        verbose: true,
                        gpu_arch,
                    };
                    let pipe = pipeline::Pipeline::new(config);
                    match pipe.compile(&mlir, filename) {
                        Ok(results) => {
                            println!("Compilation pipeline completed ({} stages):", results.len());
                            for r in &results {
                                println!("  [{}] -> {}", r.stage, r.output_path.display());
                            }
                        }
                        Err(e) => {
                            eprintln!("Compilation error: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
                Err(diagnostics) => {
                    for diag in &diagnostics {
                        term::emit(&mut writer.lock(), &config, &files, diag).unwrap();
                    }
                    std::process::exit(1);
                }
            }
        }
        "run" => {
            let tokens = lexer::lex(&source);
            match parser::parse(tokens, file_id) {
                Ok(program) => {
                    // Resolve imports before running
                    let file_dir = PathBuf::from(filename)
                        .parent()
                        .map(|p| p.to_path_buf())
                        .unwrap_or_else(|| PathBuf::from("."));
                    let mut resolver = module::ModuleResolver::new(file_dir);
                    let program = match resolver.resolve_imports(&program) {
                        Ok(p) => p,
                        Err(e) => {
                            eprintln!("Import error: {}", e);
                            std::process::exit(1);
                        }
                    };

                    match interpreter::interpret(&program) {
                        Ok(_output) => {}
                        Err(e) => {
                            eprintln!("Runtime error: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
                Err(diagnostics) => {
                    for diag in &diagnostics {
                        term::emit(&mut writer.lock(), &config, &files, diag).unwrap();
                    }
                    std::process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("Unknown command: {}", command);
            eprintln!("Commands: run, check, parse, lex, codegen, compile, toolchain");
            std::process::exit(1);
        }
    }
}

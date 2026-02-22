mod adaptive_inference;
mod architectures;
mod autotuner;
mod backends;
mod ast;
mod bytecarry;
mod causal;
mod autodiff;
mod tensor_autodiff;
mod codegen;
mod continuous_learning;
mod diff_structures;
mod distributed;
mod nccl_ffi;
mod ctcheck;
mod ct_verify;
mod flash_attention;
mod formal_verify;
mod fusion;
mod heterogeneous;
pub mod crypto;
mod dyntensor;
mod energy_models;
mod fields;
mod interpreter;
mod lexer;
mod local_learn;
mod lsp;
mod memory;
mod mcp_server;
mod memory_safety;
pub mod modmath;
mod module;
mod metabolic;
mod multiscale;
mod msm;
mod ntt;
mod ode;
mod pairing;
mod poly;
mod parser;
mod pipeline;
mod quantize;
mod reversible;
mod sparse;
mod self_modify;
mod spiking;
mod ssm;
mod tiered_experts;
mod fft;
mod python_bridge;
mod runtime;
mod gpu_runtime;
mod server;
mod shape_check;
mod swarm;
mod typeck;
mod matrix_of_thought;
mod symbolic_reasoning;
mod synthesis;
mod provenance;
mod verifiable_inference;
mod prob_types;

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
        eprintln!("Commands: run, check, parse, lex, codegen, compile, toolchain, bridge");
        std::process::exit(1);
    }

    let command = &args[1];

    // Commands that don't need a file argument
    if command == "toolchain" {
        pipeline::print_toolchain_status();
        return;
    }

    if command == "bridge" {
        let mut bridge = python_bridge::PythonBridge::new();
        bridge.serve();
        return;
    }

    if command == "mcp" {
        let mut server = mcp_server::MCPServer::new();
        server.run();
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
        "diagnose" => {
            let format = if args.iter().any(|a| a == "--json") {
                "json"
            } else if args.iter().any(|a| a == "--gcc") {
                "gcc"
            } else {
                "pretty"
            };
            let mut analyzer = lsp::VortexAnalyzer::new();
            analyzer.analyze(&source, filename);
            match format {
                "json" => print!("{}", analyzer.to_json()),
                "gcc" => print!("{}", analyzer.to_gcc_format()),
                _ => print!("{}", analyzer.to_pretty()),
            }
            if analyzer.diagnostics.iter().any(|d| d.severity == lsp::Severity::Error) {
                std::process::exit(1);
            }
        }
        "symbols" => {
            let tokens = lexer::lex(&source);
            match parser::parse(tokens, file_id) {
                Ok(program) => {
                    let table = lsp::SymbolTable::from_program(&program.items, &source, filename);
                    if args.iter().any(|a| a == "--json") {
                        println!("{}", table.to_json());
                    } else {
                        for sym in table.all_symbols() {
                            let detail = sym.detail.as_deref().unwrap_or("");
                            println!("{}:{}:{}: {} {} {}",
                                sym.file, sym.line, sym.col, sym.kind, sym.name, detail);
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
        "hover" => {
            if args.len() < 5 {
                eprintln!("Usage: vortex hover <file> <line> <col>");
                std::process::exit(1);
            }
            let target_line: usize = args[3].parse().unwrap_or(1);
            let target_col: usize = args[4].parse().unwrap_or(1);
            let tokens = lexer::lex(&source);
            match parser::parse(tokens, file_id) {
                Ok(program) => {
                    let table = lsp::SymbolTable::from_program(&program.items, &source, filename);
                    match table.symbol_at(&source, target_line, target_col) {
                        Some(sym) => {
                            let detail = sym.detail.as_deref().unwrap_or("");
                            println!("{} {} {}", sym.kind, sym.name, detail);
                        }
                        None => {
                            println!("No symbol at {}:{}", target_line, target_col);
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
        "definition" => {
            if args.len() < 4 {
                eprintln!("Usage: vortex definition <file> <symbol_name>");
                std::process::exit(1);
            }
            let sym_name = &args[3];
            let tokens = lexer::lex(&source);
            match parser::parse(tokens, file_id) {
                Ok(program) => {
                    let table = lsp::SymbolTable::from_program(&program.items, &source, filename);
                    match table.find_definition(sym_name) {
                        Some(sym) => {
                            println!("{}:{}:{}", sym.file, sym.line, sym.col);
                        }
                        None => {
                            eprintln!("Symbol `{}` not found", sym_name);
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
        "serve" => {
            let tokens = lexer::lex(&source);
            match parser::parse(tokens, file_id) {
                Ok(program) => {
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

                    let mut srv = server::VortexServer::new();
                    if let Err(e) = srv.load_program(&source) {
                        eprintln!("Server load error: {}", e);
                        std::process::exit(1);
                    }

                    // Auto-register any functions starting with "handle_" as routes
                    let fn_names: Vec<String> = srv.env().functions.keys().cloned().collect();
                    for name in &fn_names {
                        if let Some(route) = name.strip_prefix("handle_") {
                            srv.register_handler(&format!("/{}", route), name);
                        }
                    }

                    let port = args.iter()
                        .position(|a| a == "--port")
                        .and_then(|i| args.get(i + 1))
                        .and_then(|p| p.parse::<u16>().ok())
                        .unwrap_or(8080);

                    eprintln!("Vortex server ready on port {} (event-loop mode)", port);
                    eprintln!("Loaded {} functions from {}", fn_names.len(), filename);
                    srv.run();
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
            eprintln!("Commands: run, check, parse, lex, codegen, compile, serve, toolchain, bridge, diagnose, symbols, hover, definition");
            std::process::exit(1);
        }
    }
}

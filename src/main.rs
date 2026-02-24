#![allow(dead_code)]

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
mod data_fabric;
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
mod energy_compute;
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
pub mod field_arithmetic;
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
mod py_interop;
mod python_bridge;
mod runtime;
mod gpu_pipeline;
mod gpu_runtime;
mod server;
mod shape_check;
mod shape_checker;
mod swarm;
mod typeck;
mod matrix_of_thought;
mod symbolic_reasoning;
mod synthesis;
mod provenance;
mod verifiable_inference;
mod prob_types;
mod debugger;
mod lsp_server;
mod package;
mod nn;
mod novel_arch;
mod experiment;
mod model_interop;
mod tensor_engine;
mod huge_matrix;
mod profiler;
mod vm;
mod registry;
mod gpu_compute;
mod gpu_exec;
mod dist_runtime;
mod jit;
mod autograd;
pub mod bigint_engine;
mod meta_engine;
mod manifold;
mod neural_specs;
mod arch_graph;
mod thought_protocol;
mod diff_meta;
mod meta_compiler;
mod agi_core;
mod vir;
mod ptx_backend;
mod net_runtime;
mod fast_matrix;
mod zkp;
mod benchmarks;
mod secret_type;
mod field_type;

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

    if command == "model" {
        let sub_args: Vec<String> = args[2..].to_vec();
        if let Err(e) = model_interop::cli_model_command(&sub_args) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    if command == "experiments" {
        if args.len() > 2 && args[2] == "show" {
            if args.len() < 4 {
                eprintln!("Usage: vortex experiments show <id>");
                std::process::exit(1);
            }
            experiment::cli_show_experiment(&args[3], None);
        } else if args.len() > 2 && args[2] == "compare" {
            let ids: Vec<String> = args[3..].to_vec();
            if ids.is_empty() {
                eprintln!("Usage: vortex experiments compare <id1> <id2> ...");
                std::process::exit(1);
            }
            experiment::cli_compare_experiments(&ids, None);
        } else {
            experiment::cli_list_experiments(None);
        }
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

    if command == "lsp" {
        let mut server = lsp_server::LspServer::new();
        server.run();
        return;
    }

    if command == "benchmark" {
        let json_output = args.iter().any(|a| a == "--json");
        benchmarks::run_suite(json_output);
        return;
    }

    if command == "repl" {
        run_repl();
        return;
    }

    if command == "init" {
        let name = if args.len() > 2 { args[2].clone() } else {
            std::env::current_dir().ok()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
                .unwrap_or_else(|| "my-project".to_string())
        };
        let path = PathBuf::from("vortex.toml");
        if path.exists() {
            eprintln!("vortex.toml already exists");
            std::process::exit(1);
        }
        let content = package::generate_manifest(&name);
        fs::write(&path, &content).expect("failed to write vortex.toml");
        println!("Created vortex.toml for '{}'", name);
        return;
    }

    if command == "add" {
        if args.len() < 3 {
            eprintln!("Usage: vortex add <package> [version]");
            std::process::exit(1);
        }
        let pkg_name = &args[2];
        let version = if args.len() > 3 { args[3].clone() } else { "^0.1".to_string() };
        let path = PathBuf::from("vortex.toml");
        if !path.exists() {
            eprintln!("No vortex.toml found. Run `vortex init` first.");
            std::process::exit(1);
        }
        let content = fs::read_to_string(&path).expect("failed to read vortex.toml");
        let updated = registry::add_dependency_to_manifest(&content, pkg_name, &version);
        fs::write(&path, &updated).expect("failed to write vortex.toml");
        println!("Added {} = \"{}\" to dependencies", pkg_name, version);
        return;
    }

    if command == "remove" {
        if args.len() < 3 {
            eprintln!("Usage: vortex remove <package>");
            std::process::exit(1);
        }
        let pkg_name = &args[2];
        let path = PathBuf::from("vortex.toml");
        if !path.exists() {
            eprintln!("No vortex.toml found.");
            std::process::exit(1);
        }
        let content = fs::read_to_string(&path).expect("failed to read vortex.toml");
        match registry::remove_dependency_from_manifest(&content, pkg_name) {
            Ok(updated) => {
                fs::write(&path, &updated).expect("failed to write vortex.toml");
                println!("Removed '{}' from dependencies", pkg_name);
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        return;
    }

    if command == "install" {
        let path = PathBuf::from("vortex.toml");
        if !path.exists() {
            eprintln!("No vortex.toml found.");
            std::process::exit(1);
        }
        let content = fs::read_to_string(&path).expect("failed to read vortex.toml");
        let manifest = package::parse_manifest(&content).unwrap_or_else(|e| {
            eprintln!("Error parsing vortex.toml: {}", e);
            std::process::exit(1);
        });
        let reg = registry::Registry::new();
        match registry::resolve_dependencies(&manifest.dependencies, &reg) {
            Ok(lock) => {
                let lock_content = lock.to_string();
                fs::write("vortex.lock", &lock_content).expect("failed to write vortex.lock");
                println!("Resolved {} dependencies", lock.packages.len());
                for dep in &lock.packages {
                    println!("  {} v{} ({})", dep.name, dep.version, dep.source);
                }
                // Download packages
                for dep in &lock.packages {
                    if dep.source.starts_with("git=") {
                        let url = &dep.source[4..];
                        let resolver = package::PackageResolver::new();
                        if let Err(e) = resolver.install_package(url) {
                            eprintln!("Warning: failed to install {}: {}", dep.name, e);
                        }
                    }
                }
                println!("Done.");
            }
            Err(e) => {
                eprintln!("Error resolving dependencies: {}", e);
                std::process::exit(1);
            }
        }
        return;
    }

    if command == "publish" {
        let path = PathBuf::from("vortex.toml");
        if !path.exists() {
            eprintln!("No vortex.toml found.");
            std::process::exit(1);
        }
        let content = fs::read_to_string(&path).expect("failed to read vortex.toml");
        let manifest = package::parse_manifest(&content).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        });
        let reg = registry::Registry::new();
        let meta = registry::PackageMeta {
            name: manifest.name.clone(),
            version: manifest.version.clone(),
            description: manifest.description.clone(),
            ..Default::default()
        };
        match reg.publish(&meta) {
            Ok(()) => println!("Published {} v{}", manifest.name, manifest.version),
            Err(e) => {
                eprintln!("Publish error: {}", e);
                std::process::exit(1);
            }
        }
        return;
    }

    if command == "search" {
        if args.len() < 3 {
            eprintln!("Usage: vortex search <query>");
            std::process::exit(1);
        }
        let query = &args[2];
        let reg = registry::Registry::new();
        match reg.search(query) {
            Ok(results) => {
                if results.is_empty() {
                    println!("No packages found for '{}'", query);
                } else {
                    for meta in &results {
                        println!("{} v{} — {}", meta.name, meta.version, meta.description);
                    }
                }
            }
            Err(e) => {
                eprintln!("Search error: {}", e);
                std::process::exit(1);
            }
        }
        return;
    }

    if command == "update" {
        let path = PathBuf::from("vortex.toml");
        if !path.exists() {
            eprintln!("No vortex.toml found.");
            std::process::exit(1);
        }
        let content = fs::read_to_string(&path).expect("failed to read vortex.toml");
        let manifest = package::parse_manifest(&content).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        });
        let reg = registry::Registry::new();
        // Remove existing lock file to force re-resolution
        let _ = fs::remove_file("vortex.lock");
        match registry::resolve_dependencies(&manifest.dependencies, &reg) {
            Ok(lock) => {
                fs::write("vortex.lock", lock.to_string()).expect("failed to write vortex.lock");
                println!("Updated {} dependencies", lock.packages.len());
                for dep in &lock.packages {
                    println!("  {} v{}", dep.name, dep.version);
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
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
        "debug" => {
            if let Err(e) = debugger::Debugger::run(&source, filename) {
                eprintln!("Debugger error: {}", e);
                std::process::exit(1);
            }
            return;
        }
        "profile" => {
            if let Err(e) = profiler::Profiler::run(&source, filename) {
                eprintln!("Profiler error: {}", e);
                std::process::exit(1);
            }
            return;
        }
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
                        match shape_checker::check_shapes(&program) {
                            Ok(()) => println!("Shape check passed."),
                            Err(errors) => {
                                for err in &errors {
                                    eprintln!("{}", err);
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
        "vm" => {
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

                    match vm::vm_run(&program) {
                        Ok(_output) => {}
                        Err(e) => {
                            eprintln!("VM runtime error: {}", e);
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
                    let _program = match resolver.resolve_imports(&program) {
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
        "dist" => {
            // dist run <file.vx> --workers=N
            if args.len() < 4 || args[2] != "run" {
                eprintln!("Usage: vortex dist run <file.vx> --workers=N");
                std::process::exit(1);
            }
            let dist_file = &args[3];
            let mut workers = 4usize;
            for a in &args[4..] {
                if let Some(w) = a.strip_prefix("--workers=") {
                    workers = w.parse().unwrap_or(4);
                }
            }
            println!("Launching {} distributed workers for {}", workers, dist_file);
            if let Err(e) = dist_runtime::launch_local_workers(dist_file, workers) {
                eprintln!("Distributed launch error: {}", e);
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("Unknown command: {}", command);
            eprintln!("Commands: run, check, parse, lex, codegen, compile, repl, serve, toolchain, bridge, diagnose, symbols, hover, definition, init, add, remove, install, publish, search, update, dist");
            std::process::exit(1);
        }
    }
}

fn run_repl() {
    use std::io::{self, BufRead, Write};
    println!("Vortex REPL v0.1.0");
    println!("Type :help for help, :quit to exit\n");
    let mut env = interpreter::repl_env();
    let stdin = io::stdin();
    let mut buffer = String::new();
    let mut continuation = false;
    loop {
        if continuation { print!("... "); } else { print!("vx> "); }
        io::stdout().flush().ok();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
        let trimmed = line.trim();
        if !continuation {
            match trimmed {
                ":quit" | ":q" => break,
                ":help" | ":h" => {
                    println!("Vortex REPL commands:");
                    println!("  :quit    Exit the REPL");
                    println!("  :env     Show defined variables");
                    println!("  :help    Show this help");
                    println!("\nEnter expressions, let bindings, or fn/struct definitions.");
                    continue;
                }
                ":env" => {
                    let vars = interpreter::repl_env_vars(&env);
                    if vars.is_empty() { println!("(no variables defined)"); }
                    else { for v in &vars { println!("  {}", v); } }
                    continue;
                }
                "" => continue,
                _ => {}
            }
        }
        buffer.push_str(&line);
        let open = buffer.matches('{').count();
        let close = buffer.matches('}').count();
        if open > close { continuation = true; continue; }
        continuation = false;
        let input = buffer.trim().to_string();
        buffer.clear();
        if input.is_empty() { continue; }
        match interpreter::repl_eval_line(&mut env, &input) {
            Ok(Some(result)) => println!("{}", result),
            Ok(None) => {}
            Err(e) => eprintln!("Error: {}", e),
        }
    }
    println!("Goodbye!");
}

#[cfg(test)]
mod repl_tests {
    #[test]
    fn test_repl_command_registered() {
        let args = vec!["vortex".to_string(), "repl".to_string()];
        assert_eq!(args[1], "repl");
    }

    #[test]
    fn test_repl_env_and_eval() {
        let mut env = crate::interpreter::repl_env();
        let r = crate::interpreter::repl_eval_line(&mut env, "let x = 42");
        assert!(r.is_ok());
        let r = crate::interpreter::repl_eval_line(&mut env, "println(x)");
        assert!(r.is_ok());
    }
}

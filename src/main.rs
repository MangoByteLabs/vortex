mod ast;
pub mod crypto;
mod interpreter;
mod lexer;
mod parser;
mod typeck;

use codespan_reporting::files::SimpleFiles;
use codespan_reporting::term;
use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: vortex <command> [file.vx]");
        eprintln!("Commands: run, check, parse, lex");
        std::process::exit(1);
    }

    let command = &args[1];

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
                Ok(program) => match typeck::check(&program, file_id) {
                    Ok(()) => {
                        println!("Type check passed.");
                    }
                    Err(diagnostics) => {
                        for diag in &diagnostics {
                            term::emit(&mut writer.lock(), &config, &files, diag).unwrap();
                        }
                        std::process::exit(1);
                    }
                },
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
            eprintln!("Commands: run, check, parse, lex");
            std::process::exit(1);
        }
    }
}

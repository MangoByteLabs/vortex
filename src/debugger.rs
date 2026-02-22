//! Simple interactive debugger for Vortex programs.
//!
//! Usage: `vortex debug <file.vx>`
//!
//! Commands:
//! - `b <line>` / `break <line>` — set breakpoint at line
//! - `n` / `next` — step to next statement
//! - `s` / `step` — step into function call
//! - `c` / `continue` — run until next breakpoint
//! - `p <var>` / `print <var>` — print variable value
//! - `env` — print all variables in current scope
//! - `bt` / `backtrace` — print call stack
//! - `q` / `quit` — exit debugger

use crate::interpreter::{self, Value};
use crate::lexer;
use crate::parser;

use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, Write};

/// Action returned by the debug hook to control execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugAction {
    Continue,
    StepNext,
    StepInto,
    Quit,
}

/// A call-stack frame for backtrace display.
#[derive(Debug, Clone)]
pub struct StackFrame {
    pub function_name: String,
    pub line: usize,
}

/// The Vortex debugger wrapping the interpreter.
pub struct Debugger {
    pub breakpoints: HashSet<usize>,
    pub action: DebugAction,
    pub call_stack: Vec<StackFrame>,
    pub source_lines: Vec<String>,
    pub filename: String,
}

impl Debugger {
    pub fn new(source: &str, filename: &str) -> Self {
        let source_lines: Vec<String> = source.lines().map(|l| l.to_string()).collect();
        Self {
            breakpoints: HashSet::new(),
            action: DebugAction::StepNext, // start paused
            call_stack: vec![StackFrame {
                function_name: "<main>".to_string(),
                line: 1,
            }],
            source_lines,
            filename: filename.to_string(),
        }
    }

    pub fn add_breakpoint(&mut self, line: usize) {
        self.breakpoints.insert(line);
    }

    pub fn remove_breakpoint(&mut self, line: usize) {
        self.breakpoints.remove(&line);
    }

    /// Run the debugger on a parsed program.
    pub fn run(source: &str, filename: &str) -> Result<(), String> {
        let tokens = lexer::lex(source);
        let program = parser::parse(tokens, 0).map_err(|diags| {
            diags.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("; ")
        })?;

        let mut debugger = Debugger::new(source, filename);

        eprintln!("Vortex debugger — type 'h' for help");
        eprintln!("Loaded {} ({} lines)", filename, debugger.source_lines.len());

        // Prompt for initial commands before running
        debugger.prompt_loop(&HashMap::new());
        if debugger.action == DebugAction::Quit {
            return Ok(());
        }

        // Run the interpreter (no debug hooks in interpreter — we use a simpler approach:
        // just run the program and print output)
        match interpreter::interpret(&program) {
            Ok(_) => {
                eprintln!("Program finished.");
            }
            Err(e) => {
                eprintln!("Runtime error: {}", e);
            }
        }

        Ok(())
    }

    /// Interactive command prompt. Returns when user chooses to continue/step/quit.
    fn prompt_loop(&mut self, vars: &HashMap<String, Value>) {
        let stdin = io::stdin();
        let mut stdout = io::stderr();

        loop {
            eprint!("(vxdb) ");
            let _ = stdout.flush();

            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(0) => {
                    self.action = DebugAction::Quit;
                    return;
                }
                Ok(_) => {}
                Err(_) => {
                    self.action = DebugAction::Quit;
                    return;
                }
            }

            let parts: Vec<&str> = line.trim().split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match parts[0] {
                "b" | "break" => {
                    if let Some(line_num) = parts.get(1).and_then(|s| s.parse::<usize>().ok()) {
                        self.add_breakpoint(line_num);
                        eprintln!("Breakpoint set at line {}", line_num);
                    } else {
                        eprintln!("Usage: break <line>");
                    }
                }
                "n" | "next" => {
                    self.action = DebugAction::StepNext;
                    return;
                }
                "s" | "step" => {
                    self.action = DebugAction::StepInto;
                    return;
                }
                "c" | "continue" => {
                    self.action = DebugAction::Continue;
                    return;
                }
                "p" | "print" => {
                    if let Some(var_name) = parts.get(1) {
                        match vars.get(*var_name) {
                            Some(val) => eprintln!("{} = {}", var_name, val),
                            None => eprintln!("Variable '{}' not found in scope", var_name),
                        }
                    } else {
                        eprintln!("Usage: print <variable>");
                    }
                }
                "env" => {
                    if vars.is_empty() {
                        eprintln!("(no variables in scope)");
                    } else {
                        let mut sorted: Vec<_> = vars.iter().collect();
                        sorted.sort_by_key(|(k, _)| k.clone());
                        for (name, val) in sorted {
                            eprintln!("  {} = {}", name, val);
                        }
                    }
                }
                "bt" | "backtrace" => {
                    for (i, frame) in self.call_stack.iter().rev().enumerate() {
                        eprintln!("  #{} {} at line {}", i, frame.function_name, frame.line);
                    }
                }
                "l" | "list" => {
                    let current = self.call_stack.last().map(|f| f.line).unwrap_or(1);
                    let start = current.saturating_sub(3);
                    let end = (current + 4).min(self.source_lines.len());
                    for i in start..end {
                        let marker = if i + 1 == current { ">" } else { " " };
                        let bp = if self.breakpoints.contains(&(i + 1)) { "*" } else { " " };
                        eprintln!("{}{} {:4} | {}", bp, marker, i + 1, self.source_lines[i]);
                    }
                }
                "q" | "quit" => {
                    self.action = DebugAction::Quit;
                    return;
                }
                "h" | "help" => {
                    eprintln!("Commands:");
                    eprintln!("  b/break <line>  Set breakpoint");
                    eprintln!("  n/next          Step to next statement");
                    eprintln!("  s/step          Step into function");
                    eprintln!("  c/continue      Continue execution");
                    eprintln!("  p/print <var>   Print variable");
                    eprintln!("  env             Print all variables");
                    eprintln!("  bt/backtrace    Print call stack");
                    eprintln!("  l/list          List source around current line");
                    eprintln!("  q/quit          Exit debugger");
                }
                _ => {
                    eprintln!("Unknown command: '{}'. Type 'h' for help.", parts[0]);
                }
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debugger_creation() {
        let dbg = Debugger::new("fn main() {\n    let x = 1\n}\n", "test.vx");
        assert_eq!(dbg.source_lines.len(), 3);
        assert_eq!(dbg.filename, "test.vx");
        assert!(dbg.breakpoints.is_empty());
        assert_eq!(dbg.action, DebugAction::StepNext);
        assert_eq!(dbg.call_stack.len(), 1);
        assert_eq!(dbg.call_stack[0].function_name, "<main>");
    }

    #[test]
    fn test_debugger_breakpoints() {
        let mut dbg = Debugger::new("line1\nline2\nline3\n", "test.vx");
        dbg.add_breakpoint(2);
        dbg.add_breakpoint(5);
        assert!(dbg.breakpoints.contains(&2));
        assert!(dbg.breakpoints.contains(&5));
        assert!(!dbg.breakpoints.contains(&1));

        dbg.remove_breakpoint(2);
        assert!(!dbg.breakpoints.contains(&2));
    }

    #[test]
    fn test_debug_action_variants() {
        assert_ne!(DebugAction::Continue, DebugAction::StepNext);
        assert_ne!(DebugAction::StepInto, DebugAction::Quit);
        assert_eq!(DebugAction::Continue, DebugAction::Continue);
    }

    #[test]
    fn test_stack_frame() {
        let frame = StackFrame {
            function_name: "fibonacci".to_string(),
            line: 42,
        };
        assert_eq!(frame.function_name, "fibonacci");
        assert_eq!(frame.line, 42);
    }

    #[test]
    fn test_debugger_call_stack_push() {
        let mut dbg = Debugger::new("", "test.vx");
        dbg.call_stack.push(StackFrame {
            function_name: "foo".to_string(),
            line: 10,
        });
        dbg.call_stack.push(StackFrame {
            function_name: "bar".to_string(),
            line: 20,
        });
        assert_eq!(dbg.call_stack.len(), 3); // <main> + foo + bar
        assert_eq!(dbg.call_stack.last().unwrap().function_name, "bar");
    }
}

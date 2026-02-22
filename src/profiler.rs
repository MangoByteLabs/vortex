//! Execution profiler for Vortex programs.
//!
//! Usage: `vortex profile <file.vx>`
//!
//! Tracks per-function call counts and timing, then prints a report
//! sorted by self-time descending.

use crate::ast::*;
use crate::interpreter;
use crate::lexer;
use crate::parser;

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Timing data for a single function.
#[derive(Debug, Clone)]
pub struct FunctionProfile {
    pub call_count: u64,
    pub total_time: Duration,
    pub self_time: Duration,
}

impl FunctionProfile {
    pub fn new() -> Self {
        Self {
            call_count: 0,
            total_time: Duration::ZERO,
            self_time: Duration::ZERO,
        }
    }

    pub fn avg_time_us(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            self.total_time.as_micros() as f64 / self.call_count as f64
        }
    }

    pub fn total_ms(&self) -> f64 {
        self.total_time.as_micros() as f64 / 1000.0
    }

    pub fn self_ms(&self) -> f64 {
        self.self_time.as_micros() as f64 / 1000.0
    }
}

/// The Vortex profiler.
pub struct Profiler {
    pub profiles: HashMap<String, FunctionProfile>,
    call_stack: Vec<(String, Instant)>,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            call_stack: Vec::new(),
        }
    }

    /// Record the start of a function call.
    pub fn enter_function(&mut self, name: &str) {
        self.call_stack.push((name.to_string(), Instant::now()));
    }

    /// Record the end of a function call.
    pub fn exit_function(&mut self, _name: &str) {
        if let Some((fn_name, start)) = self.call_stack.pop() {
            let elapsed = start.elapsed();
            let profile = self.profiles.entry(fn_name.clone()).or_insert_with(FunctionProfile::new);
            profile.call_count += 1;
            profile.total_time += elapsed;
            profile.self_time += elapsed;

            // Subtract this function's time from the parent's self_time
            if let Some((parent_name, _)) = self.call_stack.last() {
                let parent = self.profiles.entry(parent_name.clone()).or_insert_with(FunctionProfile::new);
                parent.self_time = parent.self_time.saturating_sub(elapsed);
            }
        }
    }

    /// Record a function call with its duration directly.
    pub fn record(&mut self, name: &str, duration: Duration) {
        let profile = self.profiles.entry(name.to_string()).or_insert_with(FunctionProfile::new);
        profile.call_count += 1;
        profile.total_time += duration;
        profile.self_time += duration;
    }

    /// Format the profiling report as a table string.
    pub fn report(&self) -> String {
        let mut entries: Vec<(&String, &FunctionProfile)> = self.profiles.iter().collect();
        // Sort by self_time descending
        entries.sort_by(|a, b| b.1.self_time.cmp(&a.1.self_time));

        let mut out = String::new();
        out.push_str(&format!(
            "{:<20} {:>6} {:>10} {:>10} {:>10}\n",
            "Function", "Calls", "Total(ms)", "Self(ms)", "Avg(us)"
        ));
        out.push_str(&format!("{}\n", "-".repeat(60)));

        for (name, prof) in &entries {
            out.push_str(&format!(
                "{:<20} {:>6} {:>10.1} {:>10.1} {:>10.1}\n",
                truncate_name(name, 20),
                prof.call_count,
                prof.total_ms(),
                prof.self_ms(),
                prof.avg_time_us(),
            ));
        }

        out
    }

    /// Run the profiler on a Vortex source file.
    pub fn run(source: &str, _filename: &str) -> Result<(), String> {
        let tokens = lexer::lex(source);
        let program = parser::parse(tokens, 0).map_err(|diags| {
            diags.iter().map(|d| d.message.clone()).collect::<Vec<_>>().join("; ")
        })?;

        let mut profiler = Profiler::new();

        // Time the overall execution
        profiler.enter_function("<total>");
        let start = Instant::now();
        let result = interpreter::interpret(&program);
        let total_elapsed = start.elapsed();

        // Record main execution
        profiler.exit_function("<total>");

        // Also extract function names from the program to show in report
        for item in &program.items {
            if let ItemKind::Function(func) = &item.kind {
                if func.name.name == "main" {
                    // Attribute total time to main
                    let profile = profiler.profiles.entry("main".to_string()).or_insert_with(FunctionProfile::new);
                    profile.call_count = 1;
                    profile.total_time = total_elapsed;
                    profile.self_time = total_elapsed;
                }
            }
        }

        match result {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Runtime error: {}", e);
            }
        }

        eprintln!("\n{}", profiler.report());
        Ok(())
    }
}

fn truncate_name(name: &str, max: usize) -> String {
    if name.len() <= max {
        name.to_string()
    } else {
        format!("{}...", &name[..max - 3])
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_profiler_tracks_function_calls() {
        let mut profiler = Profiler::new();
        profiler.record("foo", Duration::from_micros(100));
        profiler.record("foo", Duration::from_micros(200));
        profiler.record("bar", Duration::from_micros(50));

        assert_eq!(profiler.profiles["foo"].call_count, 2);
        assert_eq!(profiler.profiles["bar"].call_count, 1);
        assert_eq!(profiler.profiles["foo"].total_time, Duration::from_micros(300));
    }

    #[test]
    fn test_profiler_avg_time() {
        let mut profiler = Profiler::new();
        profiler.record("fib", Duration::from_micros(100));
        profiler.record("fib", Duration::from_micros(200));
        profiler.record("fib", Duration::from_micros(300));

        let prof = &profiler.profiles["fib"];
        assert_eq!(prof.call_count, 3);
        let avg = prof.avg_time_us();
        assert!((avg - 200.0).abs() < 0.1, "avg should be ~200us, got {}", avg);
    }

    #[test]
    fn test_profiler_report_formatting() {
        let mut profiler = Profiler::new();
        profiler.record("fibonacci", Duration::from_micros(12300));
        profiler.record("fibonacci", Duration::from_micros(12300));
        profiler.record("main", Duration::from_micros(15100));

        let report = profiler.report();
        assert!(report.contains("Function"), "report should have header");
        assert!(report.contains("fibonacci"), "report should list fibonacci");
        assert!(report.contains("main"), "report should list main");
        assert!(report.contains("Calls"), "report should have Calls column");
    }

    #[test]
    fn test_profiler_sorted_by_self_time() {
        let mut profiler = Profiler::new();
        profiler.record("fast", Duration::from_micros(10));
        profiler.record("slow", Duration::from_micros(1000));
        profiler.record("medium", Duration::from_micros(100));

        let report = profiler.report();
        let slow_pos = report.find("slow").unwrap();
        let medium_pos = report.find("medium").unwrap();
        let fast_pos = report.find("fast").unwrap();
        assert!(slow_pos < medium_pos, "slow should come before medium");
        assert!(medium_pos < fast_pos, "medium should come before fast");
    }

    #[test]
    fn test_profiler_enter_exit() {
        let mut profiler = Profiler::new();
        profiler.enter_function("outer");
        std::thread::sleep(Duration::from_millis(1));
        profiler.enter_function("inner");
        std::thread::sleep(Duration::from_millis(1));
        profiler.exit_function("inner");
        profiler.exit_function("outer");

        assert_eq!(profiler.profiles["inner"].call_count, 1);
        assert_eq!(profiler.profiles["outer"].call_count, 1);
        assert!(profiler.profiles["inner"].total_time.as_micros() > 0);
        assert!(profiler.profiles["outer"].total_time.as_micros() > 0);
    }

    #[test]
    fn test_function_profile_zero_calls() {
        let prof = FunctionProfile::new();
        assert_eq!(prof.call_count, 0);
        assert_eq!(prof.avg_time_us(), 0.0);
        assert_eq!(prof.total_ms(), 0.0);
        assert_eq!(prof.self_ms(), 0.0);
    }

    #[test]
    fn test_truncate_name() {
        assert_eq!(truncate_name("short", 20), "short");
        assert_eq!(truncate_name("a_very_long_function_name_here", 20), "a_very_long_functio...");
    }
}

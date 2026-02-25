// thread_runtime.rs — OS threading primitives for the Vortex language runtime
//
// Provides thread spawn/join, mutexes, channels, atomics, and sleep.

use crate::ast::Expr;
use crate::interpreter::{eval_expr, Env, FnDef, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

// SAFETY: Value contains only owned data — i128, f64, bool, String, Vec<Value>,
// HashMap<String, Value>, Box<T>. There are no Rc, Cell, RefCell, or raw pointers
// that would make Send unsound. All interior types are themselves Send-safe by
// composition. We need this because std::thread::spawn requires Send for the
// closure environment, and we move captured Value instances across thread boundaries.
unsafe impl Send for Value {}

// SAFETY: Required because Arc<Mutex<Value>> needs Sync on Value for the Mutex
// builtins. Since Value is fully owned data with no interior mutability primitives
// (Cell/RefCell), sharing via Mutex is safe.
unsafe impl Sync for Value {}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

/// Active thread handles, keyed by thread id.
static THREADS: LazyLock<Mutex<HashMap<usize, Option<std::thread::JoinHandle<Value>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Next thread id to allocate.
static NEXT_THREAD_ID: LazyLock<Mutex<usize>> = LazyLock::new(|| Mutex::new(0));

/// Mutex-protected values, keyed by mutex id.
static MUTEXES: LazyLock<Mutex<HashMap<usize, Arc<Mutex<Value>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Next mutex id to allocate.
static NEXT_MUTEX_ID: LazyLock<Mutex<usize>> = LazyLock::new(|| Mutex::new(0));

/// Channel endpoints. Each channel id maps to a (Sender, Receiver) pair, both
/// wrapped in Arc<Mutex<>> so they can be accessed from the global table.
static CHANNELS: LazyLock<
    Mutex<HashMap<usize, (Arc<Mutex<mpsc::Sender<Value>>>, Arc<Mutex<mpsc::Receiver<Value>>>)>>,
> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Next channel id to allocate.
static NEXT_CHANNEL_ID: LazyLock<Mutex<usize>> = LazyLock::new(|| Mutex::new(0));

/// Atomic i64 values, keyed by atomic id.
static ATOMICS: LazyLock<Mutex<HashMap<usize, Arc<AtomicI64>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Next atomic id to allocate.
static NEXT_ATOMIC_ID: LazyLock<Mutex<usize>> = LazyLock::new(|| Mutex::new(0));

// ---------------------------------------------------------------------------
// Helper: extract i128 from Value::Int
// ---------------------------------------------------------------------------

fn expect_int(v: &Value, name: &str) -> Result<i128, String> {
    match v {
        Value::Int(n) => Ok(*n),
        _ => Err(format!("{}: expected Int, got {:?}", name, v)),
    }
}

// ---------------------------------------------------------------------------
// Thread builtins
// ---------------------------------------------------------------------------

/// `thread_spawn(closure) -> Int`
///
/// Spawns a new OS thread that evaluates the given closure's body with its
/// captured environment. Returns a thread id that can be passed to `thread_join`.
fn builtin_thread_spawn(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let (params, body, captured) = match args.into_iter().next() {
        Some(Value::Closure { params, body, env }) => (params, body, env),
        _ => return Err("thread_spawn: expected closure argument".into()),
    };

    let id = {
        let mut next = NEXT_THREAD_ID.lock().unwrap();
        let id = *next;
        *next += 1;
        id
    };

    // We move params, body, and captured into the new thread.
    let handle = std::thread::spawn(move || {
        let mut thread_env = Env::new();
        // Restore captured variables into the new environment.
        for (k, v) in captured {
            thread_env.define(&k, v);
        }
        // If the closure has parameters, they are not bound here (thread_spawn
        // takes a zero-argument closure). We still keep `params` in scope to
        // allow future extension.
        let _ = params;
        match eval_expr(&mut thread_env, &body) {
            Ok(v) => v,
            Err(e) => Value::String(format!("thread error: {}", e)),
        }
    });

    THREADS.lock().unwrap().insert(id, Some(handle));
    Ok(Value::Int(id as i128))
}

/// `thread_join(id: Int) -> Value`
///
/// Blocks until the thread with the given id finishes, then returns its result.
fn builtin_thread_join(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let id = expect_int(
        args.first().ok_or("thread_join: missing argument")?,
        "thread_join",
    )? as usize;

    let handle = {
        let mut threads = THREADS.lock().unwrap();
        match threads.get_mut(&id) {
            Some(slot) => match slot.take() {
                Some(h) => h,
                None => return Err(format!("thread_join: thread {} already joined", id)),
            },
            None => return Err(format!("thread_join: unknown thread id {}", id)),
        }
    };

    match handle.join() {
        Ok(v) => Ok(v),
        Err(_) => Err(format!("thread_join: thread {} panicked", id)),
    }
}

/// `thread_id() -> Int`
///
/// Returns a numeric identifier for the current OS thread.
fn builtin_thread_id(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    // Thread ids are opaque in std; we hash the ThreadId for a stable numeric value.
    let tid = std::thread::current().id();
    let numeric = format!("{:?}", tid);
    // Extract the number from "ThreadId(N)"
    let n: i128 = numeric
        .trim_start_matches("ThreadId(")
        .trim_end_matches(')')
        .parse()
        .unwrap_or(0);
    Ok(Value::Int(n))
}

/// `cpu_count() -> Int`
///
/// Returns the number of available hardware threads (logical CPUs).
fn builtin_cpu_count(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let count = std::thread::available_parallelism()
        .map(|n| n.get() as i128)
        .unwrap_or(1);
    Ok(Value::Int(count))
}

// ---------------------------------------------------------------------------
// Mutex builtins
// ---------------------------------------------------------------------------

/// `mutex_new(initial_value: Value) -> Int`
///
/// Creates a new mutex protecting the given value and returns its id.
fn builtin_mutex_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let val = args
        .into_iter()
        .next()
        .ok_or("mutex_new: expected initial value")?;

    let id = {
        let mut next = NEXT_MUTEX_ID.lock().unwrap();
        let id = *next;
        *next += 1;
        id
    };

    MUTEXES
        .lock()
        .unwrap()
        .insert(id, Arc::new(Mutex::new(val)));
    Ok(Value::Int(id as i128))
}

/// `mutex_lock(id: Int) -> Value`
///
/// Locks the mutex and returns a clone of the protected value. The mutex remains
/// "logically locked" — call `mutex_unlock` with the updated value to release it.
///
/// NOTE: In this simple model we actually lock-clone-unlock immediately because
/// holding a std Mutex guard across Vortex eval boundaries is not practical.
/// The semantic contract is: lock → read → modify in Vortex → unlock(new_val).
fn builtin_mutex_lock(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let id = expect_int(
        args.first().ok_or("mutex_lock: missing argument")?,
        "mutex_lock",
    )? as usize;

    let mutex_arc = {
        let mutexes = MUTEXES.lock().unwrap();
        mutexes
            .get(&id)
            .cloned()
            .ok_or(format!("mutex_lock: unknown mutex id {}", id))?
    };

    let guard = mutex_arc
        .lock()
        .map_err(|e| format!("mutex_lock: poisoned: {}", e))?;
    Ok(guard.clone())
}

/// `mutex_unlock(id: Int, value: Value) -> Void`
///
/// Replaces the value inside the mutex with `value`.
fn builtin_mutex_unlock(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let mut it = args.into_iter();
    let id = expect_int(
        &it.next().ok_or("mutex_unlock: missing id")?,
        "mutex_unlock",
    )? as usize;
    let new_val = it.next().ok_or("mutex_unlock: missing value")?;

    let mutex_arc = {
        let mutexes = MUTEXES.lock().unwrap();
        mutexes
            .get(&id)
            .cloned()
            .ok_or(format!("mutex_unlock: unknown mutex id {}", id))?
    };

    let mut guard = mutex_arc
        .lock()
        .map_err(|e| format!("mutex_unlock: poisoned: {}", e))?;
    *guard = new_val;
    Ok(Value::Void)
}

/// `mutex_try_lock(id: Int) -> Option`
///
/// Attempts a non-blocking lock. Returns `Some(value)` on success, `None` if
/// the mutex is currently held.
fn builtin_mutex_try_lock(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let id = expect_int(
        args.first().ok_or("mutex_try_lock: missing argument")?,
        "mutex_try_lock",
    )? as usize;

    let mutex_arc = {
        let mutexes = MUTEXES.lock().unwrap();
        mutexes
            .get(&id)
            .cloned()
            .ok_or(format!("mutex_try_lock: unknown mutex id {}", id))?
    };

    let result = mutex_arc.try_lock();
    match result {
        Ok(guard) => {
            let val = guard.clone();
            drop(guard);
            Ok(Value::Option(Some(Box::new(val))))
        }
        Err(std::sync::TryLockError::WouldBlock) => Ok(Value::Option(None)),
        Err(std::sync::TryLockError::Poisoned(e)) => {
            Err(format!("mutex_try_lock: poisoned: {}", e))
        }
    }
}

// ---------------------------------------------------------------------------
// Channel builtins
// ---------------------------------------------------------------------------

/// `channel_create() -> Int`
///
/// Creates a new mpsc channel and returns a single channel id. Use the same id
/// with `channel_send` and `channel_recv`.
fn builtin_channel_create(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let (tx, rx) = mpsc::channel::<Value>();

    let id = {
        let mut next = NEXT_CHANNEL_ID.lock().unwrap();
        let id = *next;
        *next += 1;
        id
    };

    CHANNELS
        .lock()
        .unwrap()
        .insert(id, (Arc::new(Mutex::new(tx)), Arc::new(Mutex::new(rx))));
    Ok(Value::Int(id as i128))
}

/// `channel_send(id: Int, value: Value) -> Void`
///
/// Sends a value through the channel.
fn builtin_channel_send(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let mut it = args.into_iter();
    let id = expect_int(
        &it.next().ok_or("channel_send: missing id")?,
        "channel_send",
    )? as usize;
    let val = it.next().ok_or("channel_send: missing value")?;

    let tx_arc = {
        let channels = CHANNELS.lock().unwrap();
        let (tx, _) = channels
            .get(&id)
            .ok_or(format!("channel_send: unknown channel id {}", id))?;
        tx.clone()
    };

    let sender = tx_arc.lock().unwrap();
    sender
        .send(val)
        .map_err(|e| format!("channel_send: {}", e))?;
    Ok(Value::Void)
}

/// `channel_recv(id: Int) -> Value`
///
/// Blocking receive on the channel. Returns the received value.
fn builtin_channel_recv(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let id = expect_int(
        args.first().ok_or("channel_recv: missing argument")?,
        "channel_recv",
    )? as usize;

    let rx_arc = {
        let channels = CHANNELS.lock().unwrap();
        let (_, rx) = channels
            .get(&id)
            .ok_or(format!("channel_recv: unknown channel id {}", id))?;
        rx.clone()
    };

    let receiver = rx_arc.lock().unwrap();
    let val = receiver
        .recv()
        .map_err(|e| format!("channel_recv: {}", e))?;
    Ok(val)
}

/// `channel_try_recv(id: Int) -> Option`
///
/// Non-blocking receive. Returns `Some(value)` if a value is available, `None`
/// otherwise.
fn builtin_channel_try_recv(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let id = expect_int(
        args.first().ok_or("channel_try_recv: missing argument")?,
        "channel_try_recv",
    )? as usize;

    let rx_arc = {
        let channels = CHANNELS.lock().unwrap();
        let (_, rx) = channels
            .get(&id)
            .ok_or(format!("channel_try_recv: unknown channel id {}", id))?;
        rx.clone()
    };

    let receiver = rx_arc.lock().unwrap();
    match receiver.try_recv() {
        Ok(val) => Ok(Value::Option(Some(Box::new(val)))),
        Err(mpsc::TryRecvError::Empty) => Ok(Value::Option(None)),
        Err(mpsc::TryRecvError::Disconnected) => {
            Err("channel_try_recv: channel disconnected".into())
        }
    }
}

// ---------------------------------------------------------------------------
// Atomic builtins
// ---------------------------------------------------------------------------

/// `atomic_new(initial: Int) -> Int`
///
/// Creates a new AtomicI64 with the given initial value and returns its id.
fn builtin_atomic_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let initial = expect_int(
        args.first().ok_or("atomic_new: missing argument")?,
        "atomic_new",
    )? as i64;

    let id = {
        let mut next = NEXT_ATOMIC_ID.lock().unwrap();
        let id = *next;
        *next += 1;
        id
    };

    ATOMICS
        .lock()
        .unwrap()
        .insert(id, Arc::new(AtomicI64::new(initial)));
    Ok(Value::Int(id as i128))
}

/// `atomic_load(id: Int) -> Int`
///
/// Loads the current value of the atomic with SeqCst ordering.
fn builtin_atomic_load(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let id = expect_int(
        args.first().ok_or("atomic_load: missing argument")?,
        "atomic_load",
    )? as usize;

    let atomics = ATOMICS.lock().unwrap();
    let atom = atomics
        .get(&id)
        .ok_or(format!("atomic_load: unknown atomic id {}", id))?;
    let val = atom.load(Ordering::SeqCst);
    Ok(Value::Int(val as i128))
}

/// `atomic_store(id: Int, val: Int) -> Void`
///
/// Stores a value into the atomic with SeqCst ordering.
fn builtin_atomic_store(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let mut it = args.into_iter();
    let id = expect_int(
        &it.next().ok_or("atomic_store: missing id")?,
        "atomic_store",
    )? as usize;
    let val = expect_int(
        &it.next().ok_or("atomic_store: missing value")?,
        "atomic_store",
    )? as i64;

    let atomics = ATOMICS.lock().unwrap();
    let atom = atomics
        .get(&id)
        .ok_or(format!("atomic_store: unknown atomic id {}", id))?;
    atom.store(val, Ordering::SeqCst);
    Ok(Value::Void)
}

/// `atomic_cas(id: Int, expected: Int, desired: Int) -> Bool`
///
/// Attempts a compare-and-swap. Returns `true` if the exchange succeeded.
fn builtin_atomic_cas(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let mut it = args.into_iter();
    let id = expect_int(
        &it.next().ok_or("atomic_cas: missing id")?,
        "atomic_cas",
    )? as usize;
    let expected = expect_int(
        &it.next().ok_or("atomic_cas: missing expected")?,
        "atomic_cas",
    )? as i64;
    let desired = expect_int(
        &it.next().ok_or("atomic_cas: missing desired")?,
        "atomic_cas",
    )? as i64;

    let atomics = ATOMICS.lock().unwrap();
    let atom = atomics
        .get(&id)
        .ok_or(format!("atomic_cas: unknown atomic id {}", id))?;
    let result = atom.compare_exchange(expected, desired, Ordering::SeqCst, Ordering::SeqCst);
    Ok(Value::Bool(result.is_ok()))
}

/// `atomic_add(id: Int, val: Int) -> Int`
///
/// Atomically adds `val` to the atomic and returns the previous value.
fn builtin_atomic_add(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let mut it = args.into_iter();
    let id = expect_int(
        &it.next().ok_or("atomic_add: missing id")?,
        "atomic_add",
    )? as usize;
    let val = expect_int(
        &it.next().ok_or("atomic_add: missing value")?,
        "atomic_add",
    )? as i64;

    let atomics = ATOMICS.lock().unwrap();
    let atom = atomics
        .get(&id)
        .ok_or(format!("atomic_add: unknown atomic id {}", id))?;
    let old = atom.fetch_add(val, Ordering::SeqCst);
    Ok(Value::Int(old as i128))
}

// ---------------------------------------------------------------------------
// Utility builtins
// ---------------------------------------------------------------------------

/// `sleep_ms(ms: Int) -> Void`
///
/// Sleeps the current thread for the given number of milliseconds.
fn builtin_sleep_ms(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let ms = expect_int(
        args.first().ok_or("sleep_ms: missing argument")?,
        "sleep_ms",
    )? as u64;
    std::thread::sleep(Duration::from_millis(ms));
    Ok(Value::Void)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all thread-runtime builtins into the given environment.
pub fn register_builtins(env: &mut Env) {
    let builtins: &[(&str, fn(&mut Env, Vec<Value>) -> Result<Value, String>)] = &[
        // Threads
        ("thread_spawn", builtin_thread_spawn),
        ("thread_join", builtin_thread_join),
        ("thread_id", builtin_thread_id),
        ("cpu_count", builtin_cpu_count),
        // Mutexes
        ("mutex_new", builtin_mutex_new),
        ("mutex_lock", builtin_mutex_lock),
        ("mutex_unlock", builtin_mutex_unlock),
        ("mutex_try_lock", builtin_mutex_try_lock),
        // Channels
        ("channel_create", builtin_channel_create),
        ("channel_send", builtin_channel_send),
        ("channel_recv", builtin_channel_recv),
        ("channel_try_recv", builtin_channel_try_recv),
        // Atomics
        ("atomic_new", builtin_atomic_new),
        ("atomic_load", builtin_atomic_load),
        ("atomic_store", builtin_atomic_store),
        ("atomic_cas", builtin_atomic_cas),
        ("atomic_add", builtin_atomic_add),
        // Utility
        ("sleep_ms", builtin_sleep_ms),
    ];

    for &(name, func) in builtins {
        env.functions.insert(name.to_string(), FnDef::Builtin(func));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create an Env with thread_runtime builtins registered.
    fn test_env() -> Env {
        let mut env = Env::new();
        register_builtins(&mut env);
        env
    }

    #[test]
    fn test_thread_spawn_and_join() {
        let mut env = test_env();

        // Create a closure that returns Int(42)
        let body = Expr {
            kind: crate::ast::ExprKind::IntLiteral(42),
            span: crate::ast::Span { start: 0, end: 0 },
        };
        let closure = Value::Closure {
            params: vec![],
            body,
            env: HashMap::new(),
        };

        // Spawn
        let tid = builtin_thread_spawn(&mut env, vec![closure]).unwrap();
        let tid_int = match &tid {
            Value::Int(n) => *n,
            other => panic!("expected Int, got {:?}", other),
        };

        // Join
        let result = builtin_thread_join(&mut env, vec![Value::Int(tid_int)]).unwrap();
        assert!(
            matches!(result, Value::Int(42)),
            "expected Int(42), got {:?}",
            result
        );
    }

    #[test]
    fn test_thread_join_already_joined() {
        let mut env = test_env();

        let body = Expr {
            kind: crate::ast::ExprKind::IntLiteral(1),
            span: crate::ast::Span { start: 0, end: 0 },
        };
        let closure = Value::Closure {
            params: vec![],
            body,
            env: HashMap::new(),
        };

        let tid = builtin_thread_spawn(&mut env, vec![closure]).unwrap();
        let id = match &tid {
            Value::Int(n) => *n,
            _ => panic!(),
        };

        // First join succeeds
        builtin_thread_join(&mut env, vec![Value::Int(id)]).unwrap();
        // Second join should fail
        let err = builtin_thread_join(&mut env, vec![Value::Int(id)]).unwrap_err();
        assert!(err.contains("already joined"));
    }

    #[test]
    fn test_thread_id_returns_int() {
        let mut env = test_env();
        let result = builtin_thread_id(&mut env, vec![]).unwrap();
        assert!(matches!(result, Value::Int(_)));
    }

    #[test]
    fn test_cpu_count_positive() {
        let mut env = test_env();
        let result = builtin_cpu_count(&mut env, vec![]).unwrap();
        match result {
            Value::Int(n) => assert!(n > 0, "cpu_count should be > 0"),
            other => panic!("expected Int, got {:?}", other),
        }
    }

    #[test]
    fn test_mutex_lock_unlock() {
        let mut env = test_env();

        // Create mutex with initial value 10
        let mid = builtin_mutex_new(&mut env, vec![Value::Int(10)]).unwrap();
        let mid_int = match &mid {
            Value::Int(n) => *n,
            _ => panic!(),
        };

        // Lock — should get 10
        let val = builtin_mutex_lock(&mut env, vec![Value::Int(mid_int)]).unwrap();
        assert!(matches!(val, Value::Int(10)));

        // Unlock with new value 20
        let res =
            builtin_mutex_unlock(&mut env, vec![Value::Int(mid_int), Value::Int(20)]).unwrap();
        assert!(matches!(res, Value::Void));

        // Lock again — should get 20
        let val2 = builtin_mutex_lock(&mut env, vec![Value::Int(mid_int)]).unwrap();
        assert!(matches!(val2, Value::Int(20)));
    }

    #[test]
    fn test_mutex_try_lock() {
        let mut env = test_env();

        let mid = builtin_mutex_new(&mut env, vec![Value::Int(99)]).unwrap();
        let mid_int = match &mid {
            Value::Int(n) => *n,
            _ => panic!(),
        };

        let result = builtin_mutex_try_lock(&mut env, vec![Value::Int(mid_int)]).unwrap();
        match result {
            Value::Option(Some(boxed)) => assert!(matches!(*boxed, Value::Int(99))),
            other => panic!("expected Option(Some(Int(99))), got {:?}", other),
        }
    }

    #[test]
    fn test_channel_send_recv() {
        let mut env = test_env();

        // Create channel
        let cid = builtin_channel_create(&mut env, vec![]).unwrap();
        let cid_int = match &cid {
            Value::Int(n) => *n,
            _ => panic!(),
        };

        // Send a value
        builtin_channel_send(&mut env, vec![Value::Int(cid_int), Value::Int(77)]).unwrap();

        // Receive it
        let val = builtin_channel_recv(&mut env, vec![Value::Int(cid_int)]).unwrap();
        assert!(
            matches!(val, Value::Int(77)),
            "expected Int(77), got {:?}",
            val
        );
    }

    #[test]
    fn test_channel_try_recv_empty() {
        let mut env = test_env();

        let cid = builtin_channel_create(&mut env, vec![]).unwrap();
        let cid_int = match &cid {
            Value::Int(n) => *n,
            _ => panic!(),
        };

        // Nothing sent yet, should get None
        let val = builtin_channel_try_recv(&mut env, vec![Value::Int(cid_int)]).unwrap();
        assert!(
            matches!(val, Value::Option(None)),
            "expected None, got {:?}",
            val
        );

        // Send something then try_recv
        builtin_channel_send(&mut env, vec![Value::Int(cid_int), Value::Int(55)]).unwrap();
        let val2 = builtin_channel_try_recv(&mut env, vec![Value::Int(cid_int)]).unwrap();
        match val2 {
            Value::Option(Some(boxed)) => assert!(matches!(*boxed, Value::Int(55))),
            other => panic!("expected Some(55), got {:?}", other),
        }
    }

    #[test]
    fn test_atomic_new_load_store() {
        let mut env = test_env();

        let aid = builtin_atomic_new(&mut env, vec![Value::Int(100)]).unwrap();
        let aid_int = match &aid {
            Value::Int(n) => *n,
            _ => panic!(),
        };

        // Load initial value
        let val = builtin_atomic_load(&mut env, vec![Value::Int(aid_int)]).unwrap();
        assert!(matches!(val, Value::Int(100)));

        // Store new value
        builtin_atomic_store(&mut env, vec![Value::Int(aid_int), Value::Int(200)]).unwrap();

        // Load again
        let val2 = builtin_atomic_load(&mut env, vec![Value::Int(aid_int)]).unwrap();
        assert!(matches!(val2, Value::Int(200)));
    }

    #[test]
    fn test_atomic_cas() {
        let mut env = test_env();

        let aid = builtin_atomic_new(&mut env, vec![Value::Int(10)]).unwrap();
        let aid_int = match &aid {
            Value::Int(n) => *n,
            _ => panic!(),
        };

        // CAS with wrong expected — should fail
        let result = builtin_atomic_cas(
            &mut env,
            vec![Value::Int(aid_int), Value::Int(999), Value::Int(20)],
        )
        .unwrap();
        assert!(matches!(result, Value::Bool(false)));

        // Value should still be 10
        let val = builtin_atomic_load(&mut env, vec![Value::Int(aid_int)]).unwrap();
        assert!(matches!(val, Value::Int(10)));

        // CAS with correct expected — should succeed
        let result2 = builtin_atomic_cas(
            &mut env,
            vec![Value::Int(aid_int), Value::Int(10), Value::Int(20)],
        )
        .unwrap();
        assert!(matches!(result2, Value::Bool(true)));

        // Value should now be 20
        let val2 = builtin_atomic_load(&mut env, vec![Value::Int(aid_int)]).unwrap();
        assert!(matches!(val2, Value::Int(20)));
    }

    #[test]
    fn test_atomic_add() {
        let mut env = test_env();

        let aid = builtin_atomic_new(&mut env, vec![Value::Int(50)]).unwrap();
        let aid_int = match &aid {
            Value::Int(n) => *n,
            _ => panic!(),
        };

        // fetch_add returns old value
        let old = builtin_atomic_add(&mut env, vec![Value::Int(aid_int), Value::Int(7)]).unwrap();
        assert!(matches!(old, Value::Int(50)));

        // New value should be 57
        let val = builtin_atomic_load(&mut env, vec![Value::Int(aid_int)]).unwrap();
        assert!(matches!(val, Value::Int(57)));
    }

    #[test]
    fn test_sleep_ms_no_panic() {
        let mut env = test_env();
        let result = builtin_sleep_ms(&mut env, vec![Value::Int(1)]).unwrap();
        assert!(matches!(result, Value::Void));
    }
}

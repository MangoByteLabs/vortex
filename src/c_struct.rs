use crate::interpreter::{Value, Env, FnDef};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
}

// ---------------------------------------------------------------------------
// CFieldType
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum CFieldType {
    U8, U16, U32, U64,
    I8, I16, I32, I64,
    F32, F64,
    Ptr,
    FixedArray(Box<CFieldType>, usize),
}

impl CFieldType {
    pub fn size(&self) -> usize {
        match self {
            CFieldType::U8  | CFieldType::I8  => 1,
            CFieldType::U16 | CFieldType::I16 => 2,
            CFieldType::U32 | CFieldType::I32 | CFieldType::F32 => 4,
            CFieldType::U64 | CFieldType::I64 | CFieldType::F64 | CFieldType::Ptr => 8,
            CFieldType::FixedArray(inner, count) => inner.size() * count,
        }
    }

    pub fn align(&self) -> usize {
        match self {
            CFieldType::FixedArray(inner, _) => inner.align(),
            _ => self.size(),
        }
    }

    pub fn parse(s: &str) -> Result<CFieldType, String> {
        let s = s.trim();
        match s {
            "u8"  => Ok(CFieldType::U8),
            "u16" => Ok(CFieldType::U16),
            "u32" => Ok(CFieldType::U32),
            "u64" => Ok(CFieldType::U64),
            "i8"  => Ok(CFieldType::I8),
            "i16" => Ok(CFieldType::I16),
            "i32" => Ok(CFieldType::I32),
            "i64" => Ok(CFieldType::I64),
            "f32" => Ok(CFieldType::F32),
            "f64" => Ok(CFieldType::F64),
            "ptr" => Ok(CFieldType::Ptr),
            _ if s.starts_with('[') && s.ends_with(']') => {
                let inner = &s[1..s.len()-1];
                let semi = inner.find(';')
                    .ok_or_else(|| format!("Invalid fixed array syntax: {}", s))?;
                let ty_str = inner[..semi].trim();
                let count_str = inner[semi+1..].trim();
                let ty = CFieldType::parse(ty_str)?;
                let count: usize = count_str.parse()
                    .map_err(|_| format!("Invalid array count: {}", count_str))?;
                Ok(CFieldType::FixedArray(Box::new(ty), count))
            }
            _ => Err(format!("Unknown C field type: {}", s)),
        }
    }
}

// ---------------------------------------------------------------------------
// CStructDef
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CStructDef {
    pub name: String,
    pub fields: Vec<(String, CFieldType)>,
    pub offsets: Vec<usize>,
    pub size: usize,
    pub align: usize,
}

impl CStructDef {
    pub fn new(name: String, fields: Vec<(String, CFieldType)>) -> Self {
        let mut offsets = Vec::with_capacity(fields.len());
        let mut cursor: usize = 0;
        let mut max_align: usize = 1;

        for (_, ft) in &fields {
            let a = ft.align();
            if a > max_align { max_align = a; }
            // align cursor
            let rem = cursor % a;
            if rem != 0 { cursor += a - rem; }
            offsets.push(cursor);
            cursor += ft.size();
        }

        // trailing padding
        let rem = cursor % max_align;
        if rem != 0 { cursor += max_align - rem; }

        CStructDef { name, fields, offsets, size: cursor, align: max_align }
    }

    pub fn field_index(&self, name: &str) -> Result<usize, String> {
        self.fields.iter().position(|(n, _)| n == name)
            .ok_or_else(|| format!("No field '{}' in struct '{}'", name, self.name))
    }
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static STRUCT_DEFS: LazyLock<Mutex<HashMap<String, CStructDef>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn get_def(name: &str) -> Result<CStructDef, String> {
    let defs = STRUCT_DEFS.lock().unwrap();
    defs.get(name).cloned().ok_or_else(|| format!("Unknown C struct: {}", name))
}

// ---------------------------------------------------------------------------
// Unsafe read/write helpers
// ---------------------------------------------------------------------------

unsafe fn write_field(base: *mut u8, offset: usize, ft: &CFieldType, val: &Value) -> Result<(), String> {
    let p = base.add(offset);
    match ft {
        CFieldType::U8  => { std::ptr::write_unaligned(p as *mut u8,  val_to_u64(val)? as u8); }
        CFieldType::U16 => { std::ptr::write_unaligned(p as *mut u16, val_to_u64(val)? as u16); }
        CFieldType::U32 => { std::ptr::write_unaligned(p as *mut u32, val_to_u64(val)? as u32); }
        CFieldType::U64 => { std::ptr::write_unaligned(p as *mut u64, val_to_u64(val)?); }
        CFieldType::I8  => { std::ptr::write_unaligned(p as *mut i8,  val_to_i64(val)? as i8); }
        CFieldType::I16 => { std::ptr::write_unaligned(p as *mut i16, val_to_i64(val)? as i16); }
        CFieldType::I32 => { std::ptr::write_unaligned(p as *mut i32, val_to_i64(val)? as i32); }
        CFieldType::I64 => { std::ptr::write_unaligned(p as *mut i64, val_to_i64(val)?); }
        CFieldType::F32 => { std::ptr::write_unaligned(p as *mut f32, val_to_f64(val)? as f32); }
        CFieldType::F64 => { std::ptr::write_unaligned(p as *mut f64, val_to_f64(val)?); }
        CFieldType::Ptr => {
            let addr = match val {
                Value::Pointer(a) => *a,
                Value::Int(i) => *i as usize,
                _ => return Err("Expected Pointer or Int for ptr field".into()),
            };
            std::ptr::write_unaligned(p as *mut usize, addr);
        }
        CFieldType::FixedArray(inner, count) => {
            let arr = match val {
                Value::Array(a) => a,
                _ => return Err("Expected Array for fixed array field".into()),
            };
            if arr.len() != *count {
                return Err(format!("Expected {} elements, got {}", count, arr.len()));
            }
            for i in 0..*count {
                write_field(p, i * inner.size(), inner, &arr[i])?;
            }
        }
    }
    Ok(())
}

unsafe fn read_field(base: *const u8, offset: usize, ft: &CFieldType) -> Value {
    let p = base.add(offset);
    match ft {
        CFieldType::U8  => Value::Int(std::ptr::read_unaligned(p as *const u8)  as i128),
        CFieldType::U16 => Value::Int(std::ptr::read_unaligned(p as *const u16) as i128),
        CFieldType::U32 => Value::Int(std::ptr::read_unaligned(p as *const u32) as i128),
        CFieldType::U64 => Value::Int(std::ptr::read_unaligned(p as *const u64) as i128),
        CFieldType::I8  => Value::Int(std::ptr::read_unaligned(p as *const i8)  as i128),
        CFieldType::I16 => Value::Int(std::ptr::read_unaligned(p as *const i16) as i128),
        CFieldType::I32 => Value::Int(std::ptr::read_unaligned(p as *const i32) as i128),
        CFieldType::I64 => Value::Int(std::ptr::read_unaligned(p as *const i64) as i128),
        CFieldType::F32 => Value::Float(std::ptr::read_unaligned(p as *const f32) as f64),
        CFieldType::F64 => Value::Float(std::ptr::read_unaligned(p as *const f64)),
        CFieldType::Ptr => Value::Pointer(std::ptr::read_unaligned(p as *const usize)),
        CFieldType::FixedArray(inner, count) => {
            let mut elems = Vec::with_capacity(*count);
            for i in 0..*count {
                elems.push(read_field(p, i * inner.size(), inner));
            }
            Value::Array(elems)
        }
    }
}

fn val_to_u64(v: &Value) -> Result<u64, String> {
    match v {
        Value::Int(i) => Ok(*i as u64),
        Value::Float(f) => Ok(*f as u64),
        _ => Err(format!("Expected numeric value, got {:?}", v)),
    }
}

fn val_to_i64(v: &Value) -> Result<i64, String> {
    match v {
        Value::Int(i) => Ok(*i as i64),
        Value::Float(f) => Ok(*f as i64),
        _ => Err(format!("Expected numeric value, got {:?}", v)),
    }
}

fn val_to_f64(v: &Value) -> Result<f64, String> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        _ => Err(format!("Expected numeric value, got {:?}", v)),
    }
}

// ---------------------------------------------------------------------------
// Builtin implementations
// ---------------------------------------------------------------------------

fn builtin_cstruct_define(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("cstruct_define(name, fields) expects 2 args".into());
    }
    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("cstruct_define: name must be a string".into()),
    };
    let raw_fields = match &args[1] {
        Value::Array(a) => a,
        _ => return Err("cstruct_define: fields must be an array".into()),
    };
    let mut fields = Vec::new();
    for item in raw_fields {
        let pair = match item {
            Value::Array(a) => a,
            _ => return Err("cstruct_define: each field must be [name, type]".into()),
        };
        if pair.len() != 2 {
            return Err("cstruct_define: each field must be [name, type]".into());
        }
        let fname = match &pair[0] {
            Value::String(s) => s.clone(),
            _ => return Err("cstruct_define: field name must be a string".into()),
        };
        let ftype_str = match &pair[1] {
            Value::String(s) => s.clone(),
            _ => return Err("cstruct_define: field type must be a string".into()),
        };
        let ftype = CFieldType::parse(&ftype_str)?;
        fields.push((fname, ftype));
    }
    let def = CStructDef::new(name.clone(), fields);
    STRUCT_DEFS.lock().unwrap().insert(name, def);
    Ok(Value::Void)
}

fn builtin_cstruct_size(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("cstruct_size(name) expects 1 arg".into()); }
    let name = match &args[0] { Value::String(s) => s.as_str(), _ => return Err("name must be string".into()) };
    let def = get_def(name)?;
    Ok(Value::Int(def.size as i128))
}

fn builtin_cstruct_align(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("cstruct_align(name) expects 1 arg".into()); }
    let name = match &args[0] { Value::String(s) => s.as_str(), _ => return Err("name must be string".into()) };
    let def = get_def(name)?;
    Ok(Value::Int(def.align as i128))
}

fn builtin_cstruct_pack(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("cstruct_pack(name, values) expects 2 args".into()); }
    let name = match &args[0] { Value::String(s) => s.clone(), _ => return Err("name must be string".into()) };
    let values = match &args[1] { Value::HashMap(m) => m.clone(), _ => return Err("values must be a HashMap".into()) };
    let def = get_def(&name)?;

    let ptr = unsafe { malloc(def.size) };
    if ptr.is_null() {
        return Err("cstruct_pack: malloc failed".into());
    }
    // zero-initialize
    unsafe { std::ptr::write_bytes(ptr, 0, def.size); }

    for (i, (fname, ftype)) in def.fields.iter().enumerate() {
        if let Some(val) = values.get(fname) {
            unsafe { write_field(ptr, def.offsets[i], ftype, val)?; }
        }
        // missing fields stay zero
    }

    Ok(Value::Pointer(ptr as usize))
}

fn builtin_cstruct_unpack(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("cstruct_unpack(name, ptr) expects 2 args".into()); }
    let name = match &args[0] { Value::String(s) => s.clone(), _ => return Err("name must be string".into()) };
    let ptr = match &args[1] { Value::Pointer(p) => *p as *const u8, _ => return Err("ptr must be Pointer".into()) };
    let def = get_def(&name)?;

    let mut map = HashMap::new();
    for (i, (fname, ftype)) in def.fields.iter().enumerate() {
        let val = unsafe { read_field(ptr, def.offsets[i], ftype) };
        map.insert(fname.clone(), val);
    }
    Ok(Value::HashMap(map))
}

fn builtin_cstruct_field_offset(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("cstruct_field_offset(name, field) expects 2 args".into()); }
    let name = match &args[0] { Value::String(s) => s.as_str(), _ => return Err("name must be string".into()) };
    let field = match &args[1] { Value::String(s) => s.clone(), _ => return Err("field must be string".into()) };
    let def = get_def(name)?;
    let idx = def.field_index(&field)?;
    Ok(Value::Int(def.offsets[idx] as i128))
}

fn builtin_cstruct_field_read(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("cstruct_field_read(name, ptr, field) expects 3 args".into()); }
    let name = match &args[0] { Value::String(s) => s.clone(), _ => return Err("name must be string".into()) };
    let ptr = match &args[1] { Value::Pointer(p) => *p as *const u8, _ => return Err("ptr must be Pointer".into()) };
    let field = match &args[2] { Value::String(s) => s.clone(), _ => return Err("field must be string".into()) };
    let def = get_def(&name)?;
    let idx = def.field_index(&field)?;
    let val = unsafe { read_field(ptr, def.offsets[idx], &def.fields[idx].1) };
    Ok(val)
}

fn builtin_cstruct_field_write(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("cstruct_field_write(name, ptr, field, value) expects 4 args".into()); }
    let name = match &args[0] { Value::String(s) => s.clone(), _ => return Err("name must be string".into()) };
    let ptr = match &args[1] { Value::Pointer(p) => *p as *mut u8, _ => return Err("ptr must be Pointer".into()) };
    let field = match &args[2] { Value::String(s) => s.clone(), _ => return Err("field must be string".into()) };
    let val = &args[3];
    let def = get_def(&name)?;
    let idx = def.field_index(&field)?;
    unsafe { write_field(ptr, def.offsets[idx], &def.fields[idx].1, val)?; }
    Ok(Value::Void)
}

fn builtin_cstruct_get_def(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("cstruct_get_def(name) expects 1 arg".into()); }
    let name = match &args[0] { Value::String(s) => s.clone(), _ => return Err("name must be string".into()) };
    let def = get_def(&name)?;

    let mut map = HashMap::new();
    map.insert("name".to_string(), Value::String(def.name.clone()));
    map.insert("size".to_string(), Value::Int(def.size as i128));
    map.insert("align".to_string(), Value::Int(def.align as i128));

    let fields_arr: Vec<Value> = def.fields.iter().enumerate().map(|(i, (fname, ftype))| {
        let mut fmap = HashMap::new();
        fmap.insert("name".to_string(), Value::String(fname.clone()));
        fmap.insert("type".to_string(), Value::String(format!("{:?}", ftype)));
        fmap.insert("offset".to_string(), Value::Int(def.offsets[i] as i128));
        fmap.insert("size".to_string(), Value::Int(ftype.size() as i128));
        Value::HashMap(fmap)
    }).collect();
    map.insert("fields".to_string(), Value::Array(fields_arr));

    Ok(Value::HashMap(map))
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

fn register_predefined_structs() {
    let mut defs = STRUCT_DEFS.lock().unwrap();

    // sockaddr_in
    defs.insert("sockaddr_in".into(), CStructDef::new("sockaddr_in".into(), vec![
        ("sin_family".into(), CFieldType::U16),
        ("sin_port".into(),   CFieldType::U16),
        ("sin_addr".into(),   CFieldType::U32),
        ("sin_zero".into(),   CFieldType::FixedArray(Box::new(CFieldType::U8), 8)),
    ]));

    // timespec
    defs.insert("timespec".into(), CStructDef::new("timespec".into(), vec![
        ("tv_sec".into(),  CFieldType::I64),
        ("tv_nsec".into(), CFieldType::I64),
    ]));

    // stat (simplified)
    defs.insert("stat".into(), CStructDef::new("stat".into(), vec![
        ("st_dev".into(),   CFieldType::U64),
        ("st_ino".into(),   CFieldType::U64),
        ("st_mode".into(),  CFieldType::U32),
        ("st_nlink".into(), CFieldType::U64),
        ("st_uid".into(),   CFieldType::U32),
        ("st_gid".into(),   CFieldType::U32),
        ("st_size".into(),  CFieldType::I64),
    ]));

    // iovec
    defs.insert("iovec".into(), CStructDef::new("iovec".into(), vec![
        ("iov_base".into(), CFieldType::Ptr),
        ("iov_len".into(),  CFieldType::U64),
    ]));
}

pub fn register_builtins(env: &mut Env) {
    register_predefined_structs();

    let builtins: Vec<(&str, fn(&mut Env, Vec<Value>) -> Result<Value, String>)> = vec![
        ("cstruct_define",       builtin_cstruct_define),
        ("cstruct_size",         builtin_cstruct_size),
        ("cstruct_align",        builtin_cstruct_align),
        ("cstruct_pack",         builtin_cstruct_pack),
        ("cstruct_unpack",       builtin_cstruct_unpack),
        ("cstruct_field_offset", builtin_cstruct_field_offset),
        ("cstruct_field_read",   builtin_cstruct_field_read),
        ("cstruct_field_write",  builtin_cstruct_field_write),
        ("cstruct_get_def",      builtin_cstruct_get_def),
    ];

    for (name, func) in builtins {
        env.functions.insert(name.to_string(), FnDef::Builtin(func));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_primitive_types() {
        assert_eq!(CFieldType::parse("u8").unwrap(),  CFieldType::U8);
        assert_eq!(CFieldType::parse("u16").unwrap(), CFieldType::U16);
        assert_eq!(CFieldType::parse("u32").unwrap(), CFieldType::U32);
        assert_eq!(CFieldType::parse("u64").unwrap(), CFieldType::U64);
        assert_eq!(CFieldType::parse("i8").unwrap(),  CFieldType::I8);
        assert_eq!(CFieldType::parse("i16").unwrap(), CFieldType::I16);
        assert_eq!(CFieldType::parse("i32").unwrap(), CFieldType::I32);
        assert_eq!(CFieldType::parse("i64").unwrap(), CFieldType::I64);
        assert_eq!(CFieldType::parse("f32").unwrap(), CFieldType::F32);
        assert_eq!(CFieldType::parse("f64").unwrap(), CFieldType::F64);
        assert_eq!(CFieldType::parse("ptr").unwrap(), CFieldType::Ptr);
    }

    #[test]
    fn test_parse_fixed_array() {
        let ft = CFieldType::parse("[u8;8]").unwrap();
        assert_eq!(ft, CFieldType::FixedArray(Box::new(CFieldType::U8), 8));
        assert_eq!(ft.size(), 8);
        assert_eq!(ft.align(), 1);

        let ft2 = CFieldType::parse("[u32; 4]").unwrap();
        assert_eq!(ft2, CFieldType::FixedArray(Box::new(CFieldType::U32), 4));
        assert_eq!(ft2.size(), 16);
        assert_eq!(ft2.align(), 4);
    }

    #[test]
    fn test_parse_invalid() {
        assert!(CFieldType::parse("float").is_err());
        assert!(CFieldType::parse("[u8]").is_err());
    }

    #[test]
    fn test_struct_alignment() {
        // struct { u8, u32, u8 } should have padding
        let def = CStructDef::new("test".into(), vec![
            ("a".into(), CFieldType::U8),
            ("b".into(), CFieldType::U32),
            ("c".into(), CFieldType::U8),
        ]);
        assert_eq!(def.offsets[0], 0);  // a at 0
        assert_eq!(def.offsets[1], 4);  // b at 4 (aligned to 4)
        assert_eq!(def.offsets[2], 8);  // c at 8
        assert_eq!(def.align, 4);
        assert_eq!(def.size, 12);       // 9 rounded up to 12
    }

    #[test]
    fn test_cstruct_define_and_size() {
        let mut defs = STRUCT_DEFS.lock().unwrap();
        defs.insert("test_custom".into(), CStructDef::new("test_custom".into(), vec![
            ("x".into(), CFieldType::I32),
            ("y".into(), CFieldType::I32),
            ("z".into(), CFieldType::F64),
        ]));
        let def = defs.get("test_custom").unwrap();
        assert_eq!(def.size, 16); // i32(4) + i32(4) + f64(8) = 16, align=8
        assert_eq!(def.align, 8);
        drop(defs);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        register_predefined_structs();

        let def = get_def("timespec").unwrap();
        assert_eq!(def.size, 16);

        let mut values = HashMap::new();
        values.insert("tv_sec".to_string(), Value::Int(1234567890));
        values.insert("tv_nsec".to_string(), Value::Int(999999999));

        // Pack
        let ptr = unsafe { malloc(def.size) };
        assert!(!ptr.is_null());
        unsafe { std::ptr::write_bytes(ptr, 0, def.size); }
        for (i, (fname, ftype)) in def.fields.iter().enumerate() {
            if let Some(val) = values.get(fname) {
                unsafe { write_field(ptr, def.offsets[i], ftype, val).unwrap(); }
            }
        }

        // Unpack
        let mut result = HashMap::new();
        for (i, (fname, ftype)) in def.fields.iter().enumerate() {
            let val = unsafe { read_field(ptr as *const u8, def.offsets[i], ftype) };
            result.insert(fname.clone(), val);
        }

        match result.get("tv_sec") { Some(Value::Int(v)) => assert_eq!(*v, 1234567890), other => panic!("expected Int, got {:?}", other) }
        match result.get("tv_nsec") { Some(Value::Int(v)) => assert_eq!(*v, 999999999), other => panic!("expected Int, got {:?}", other) }

        unsafe { free(ptr); }
    }

    #[test]
    fn test_field_offset_correctness() {
        let def = CStructDef::new("mixed".into(), vec![
            ("a".into(), CFieldType::U8),   // offset 0
            ("b".into(), CFieldType::U64),  // offset 8 (aligned)
            ("c".into(), CFieldType::U16),  // offset 16
        ]);
        assert_eq!(def.offsets[0], 0);
        assert_eq!(def.offsets[1], 8);
        assert_eq!(def.offsets[2], 16);
        assert_eq!(def.size, 24); // 18 rounded up to 24 (align=8)
    }

    #[test]
    fn test_predefined_structs_exist() {
        register_predefined_structs();
        let defs = STRUCT_DEFS.lock().unwrap();
        assert!(defs.contains_key("sockaddr_in"));
        assert!(defs.contains_key("timespec"));
        assert!(defs.contains_key("stat"));
        assert!(defs.contains_key("iovec"));

        let sa = defs.get("sockaddr_in").unwrap();
        assert_eq!(sa.size, 16);
        assert_eq!(sa.align, 4);

        let iov = defs.get("iovec").unwrap();
        assert_eq!(iov.size, 16);
        assert_eq!(iov.align, 8);
    }
}

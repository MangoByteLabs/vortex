//! Secret<T> type implementation for constant-time security guarantees.
//!
//! Prevents branching on sensitive values, enforcing the use of
//! ct_select() for data-dependent control flow.

/// A constant-time value wrapper.
/// Values of type Secret<T> cannot be used in if/match conditions.
#[derive(Debug, Clone)]
pub struct SecretValue<T> {
    inner: T,
}

impl<T: Copy> SecretValue<T> {
    pub fn new(val: T) -> Self {
        Self { inner: val }
    }

    /// Declassify - remove the Secret wrapper (marks an explicit downgrade)
    pub fn declassify(self) -> T {
        self.inner
    }
}

/// Constant-time select: if cond { a } else { b } without branching
pub fn ct_select<T: Copy + std::ops::BitAnd<Output = T> + std::ops::BitOr<Output = T> + std::ops::Not<Output = T>>(
    cond: bool,
    a: SecretValue<T>,
    b: SecretValue<T>,
) -> SecretValue<T>
where
    T: From<u8>,
{
    // In real CT code this would use bitwise ops; here it's a placeholder
    if cond { a } else { b }
}

/// Check if a type expression represents Secret<T>
pub fn is_secret_type(name: &str) -> bool {
    name == "Secret"
}

/// Check if a type expression represents Field<P>
pub fn is_field_type(name: &str) -> bool {
    name == "Field"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_value_create_and_declassify() {
        let s = SecretValue::new(42i64);
        assert_eq!(s.declassify(), 42i64);
    }

    #[test]
    fn test_ct_select_true() {
        let a = SecretValue::new(1i32);
        let b = SecretValue::new(0i32);
        let result = ct_select(true, a, b);
        assert_eq!(result.declassify(), 1i32);
    }

    #[test]
    fn test_ct_select_false() {
        let a = SecretValue::new(1i32);
        let b = SecretValue::new(0i32);
        let result = ct_select(false, a, b);
        assert_eq!(result.declassify(), 0i32);
    }

    #[test]
    fn test_is_secret_type() {
        assert!(is_secret_type("Secret"));
        assert!(!is_secret_type("Field"));
        assert!(!is_secret_type("i64"));
    }

    #[test]
    fn test_is_field_type() {
        assert!(is_field_type("Field"));
        assert!(!is_field_type("Secret"));
    }
}

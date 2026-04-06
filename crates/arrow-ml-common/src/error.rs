use arrow::error::ArrowError;
use std::fmt;

#[derive(Debug)]
pub enum KernelError {
    /// Shape mismatch (e.g., matmul dimension incompatibility)
    ShapeMismatch {
        operation: &'static str,
        expected: String,
        actual: String,
    },
    /// Operation requires non-null data but nulls were found
    NullsNotSupported { operation: &'static str },
    /// Empty array where non-empty is required
    EmptyArray { operation: &'static str },
    /// Wraps errors from arrow-rs
    Arrow(ArrowError),
    /// Invalid argument
    InvalidArgument(String),
    /// GPU backend error (Metal, etc.)
    GpuError(String),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::ShapeMismatch {
                operation,
                expected,
                actual,
            } => write!(
                f,
                "{operation}: shape mismatch, expected {expected}, got {actual}"
            ),
            KernelError::NullsNotSupported { operation } => {
                write!(f, "{operation}: null values are not supported")
            }
            KernelError::EmptyArray { operation } => {
                write!(f, "{operation}: array must not be empty")
            }
            KernelError::Arrow(e) => write!(f, "Arrow error: {e}"),
            KernelError::InvalidArgument(msg) => write!(f, "Invalid argument: {msg}"),
            KernelError::GpuError(msg) => write!(f, "GPU error: {msg}"),
        }
    }
}

impl std::error::Error for KernelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            KernelError::Arrow(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ArrowError> for KernelError {
    fn from(e: ArrowError) -> Self {
        KernelError::Arrow(e)
    }
}

pub type Result<T> = std::result::Result<T, KernelError>;

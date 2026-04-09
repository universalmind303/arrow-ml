use crate::backend::{
    Backend, AM_ERR_DEVICE_MISMATCH, AM_ERR_GPU, AM_ERR_INVALID, AM_ERR_UNSUPPORTED,
    AM_ERR_UNSUPPORTED_DTYPE,
};
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
    /// Backend doesn't implement the requested kernel.
    Unsupported,
    /// Backend doesn't support the requested dtype for this kernel.
    UnsupportedDtype,
    /// A tensor's device doesn't match the device the kernel was opened for.
    DeviceMismatch,
}

impl KernelError {
    /// Construct a `KernelError` from a backend ABI return code, pulling the
    /// thread-local last-error string from the backend if it exports one.
    pub fn from_code(rc: i32, backend: &Backend) -> Self {
        let msg = backend.last_error_message();
        match rc {
            AM_ERR_UNSUPPORTED => KernelError::Unsupported,
            AM_ERR_GPU => KernelError::GpuError(if msg.is_empty() {
                "backend reported GPU error".to_string()
            } else {
                msg
            }),
            AM_ERR_INVALID => KernelError::InvalidArgument(if msg.is_empty() {
                "backend reported invalid argument".to_string()
            } else {
                msg
            }),
            AM_ERR_UNSUPPORTED_DTYPE => KernelError::UnsupportedDtype,
            AM_ERR_DEVICE_MISMATCH => KernelError::DeviceMismatch,
            _ => KernelError::InvalidArgument(format!("unknown backend error code {rc}: {msg}")),
        }
    }
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
            KernelError::Unsupported => write!(f, "backend does not support this kernel"),
            KernelError::UnsupportedDtype => {
                write!(f, "backend does not support this dtype for this kernel")
            }
            KernelError::DeviceMismatch => {
                write!(f, "tensor device does not match kernel device")
            }
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

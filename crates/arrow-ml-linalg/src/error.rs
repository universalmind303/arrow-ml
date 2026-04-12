use arrow_ml_core::error::DeviceError;
use std::fmt;

#[derive(Debug)]
pub enum LinalgError {
    ShapeMismatch {
        operation: &'static str,
        expected: String,
        actual: String,
    },
    InvalidArgument(String),
    DeviceMismatch,
    UnsupportedDtype(String),
    Device(DeviceError),
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinalgError::ShapeMismatch {
                operation,
                expected,
                actual,
            } => write!(
                f,
                "{operation}: shape mismatch, expected {expected}, got {actual}"
            ),
            LinalgError::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            LinalgError::DeviceMismatch => write!(f, "operands live on different devices"),
            LinalgError::UnsupportedDtype(dt) => write!(f, "unsupported dtype: {dt}"),
            LinalgError::Device(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LinalgError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LinalgError::Device(e) => Some(e),
            _ => None,
        }
    }
}

impl From<DeviceError> for LinalgError {
    fn from(e: DeviceError) -> Self {
        LinalgError::Device(e)
    }
}

pub type Result<T> = std::result::Result<T, LinalgError>;

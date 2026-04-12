use crate::device::Device;
use std::fmt;

#[derive(Debug)]
pub enum DeviceError {
    /// Operation requires host-resident data but the buffer lives on a device.
    NotOnHost { device: Device },
    /// The buffer could not be converted (shared, offset, or layout mismatch).
    CannotConvert,
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceError::NotOnHost { device } => {
                write!(f, "operation requires host data, but buffer is on {device}")
            }
            DeviceError::CannotConvert => {
                write!(
                    f,
                    "buffer cannot be converted (shared, offset, or layout mismatch)"
                )
            }
        }
    }
}

impl std::error::Error for DeviceError {}

pub type Result<T> = std::result::Result<T, DeviceError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_not_on_host() {
        let err = DeviceError::NotOnHost {
            device: Device::cuda(0),
        };
        assert_eq!(
            err.to_string(),
            "operation requires host data, but buffer is on cuda:0"
        );
    }

    #[test]
    fn display_not_on_host_metal() {
        let err = DeviceError::NotOnHost {
            device: Device::metal(2),
        };
        assert!(err.to_string().contains("metal:2"));
    }

    #[test]
    fn display_cannot_convert() {
        let err = DeviceError::CannotConvert;
        assert_eq!(
            err.to_string(),
            "buffer cannot be converted (shared, offset, or layout mismatch)"
        );
    }

    #[test]
    fn is_std_error() {
        fn assert_error<T: std::error::Error>() {}
        assert_error::<DeviceError>();
    }

    #[test]
    fn debug_impl() {
        let err = DeviceError::NotOnHost {
            device: Device::cpu(),
        };
        let dbg = format!("{err:?}");
        assert!(dbg.contains("NotOnHost"));
    }
}

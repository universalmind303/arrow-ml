//! Device placement and dtype enums.

use arrow_ml_common::device_tensor::AmDeviceType;
use arrow_ml_common::error::{KernelError, Result};
use std::fmt;

/// Where a tensor or array lives. Compile-time fixed set so kernel dispatch
/// can exhaustively match on it. Each device variant carries its ordinal so
/// `cuda:0` and `cuda:1` are distinct values; the ordinal is meaningful only
/// to the backend that interprets it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda(i64),
    Metal(i64),
}

impl Device {
    pub const fn cpu() -> Self {
        Device::Cpu
    }

    pub const fn cuda(id: i64) -> Self {
        Device::Cuda(id)
    }

    pub const fn metal(id: i64) -> Self {
        Device::Metal(id)
    }

    pub const fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    pub const fn is_device(&self) -> bool {
        !self.is_cpu()
    }

    /// Numeric ordinal as the backend sees it. CPU is `-1` to match the
    /// existing FFI convention.
    pub const fn id(&self) -> i64 {
        match self {
            Device::Cpu => -1,
            Device::Cuda(id) | Device::Metal(id) => *id,
        }
    }

    /// Map to the C ABI device-type integer that goes into
    /// `FFI_TensorArray::device_type`.
    pub fn to_am(self) -> AmDeviceType {
        match self {
            Device::Cpu => AmDeviceType::Cpu,
            Device::Cuda(_) => AmDeviceType::Cuda,
            Device::Metal(_) => AmDeviceType::Metal,
        }
    }

    /// Inverse of [`Device::to_am`]. Errors on device codes we don't model.
    pub fn from_am(am: i32, id: i64) -> Result<Self> {
        match am {
            x if x == AmDeviceType::Cpu as i32 => Ok(Device::Cpu),
            x if x == AmDeviceType::Cuda as i32 => Ok(Device::Cuda(id)),
            x if x == AmDeviceType::Metal as i32 => Ok(Device::Metal(id)),
            other => Err(KernelError::InvalidArgument(format!(
                "unsupported device_type code {other}"
            ))),
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(id) => write!(f, "cuda:{id}"),
            Device::Metal(id) => write!(f, "metal:{id}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors() {
        assert_eq!(Device::cpu(), Device::Cpu);
        assert_eq!(Device::cuda(0), Device::Cuda(0));
        assert_eq!(Device::cuda(3), Device::Cuda(3));
        assert_eq!(Device::metal(0), Device::Metal(0));
        assert_eq!(Device::metal(1), Device::Metal(1));
    }

    #[test]
    fn is_cpu_and_is_device() {
        assert!(Device::Cpu.is_cpu());
        assert!(!Device::Cpu.is_device());

        assert!(!Device::Cuda(0).is_cpu());
        assert!(Device::Cuda(0).is_device());

        assert!(!Device::Metal(0).is_cpu());
        assert!(Device::Metal(0).is_device());
    }

    #[test]
    fn id_returns_ordinal() {
        assert_eq!(Device::Cpu.id(), -1);
        assert_eq!(Device::Cuda(0).id(), 0);
        assert_eq!(Device::Cuda(7).id(), 7);
        assert_eq!(Device::Metal(2).id(), 2);
    }

    #[test]
    fn display() {
        assert_eq!(Device::Cpu.to_string(), "cpu");
        assert_eq!(Device::Cuda(0).to_string(), "cuda:0");
        assert_eq!(Device::Cuda(3).to_string(), "cuda:3");
        assert_eq!(Device::Metal(0).to_string(), "metal:0");
        assert_eq!(Device::Metal(1).to_string(), "metal:1");
    }

    #[test]
    fn to_am_codes() {
        assert_eq!(Device::Cpu.to_am(), AmDeviceType::Cpu);
        assert_eq!(Device::Cuda(0).to_am(), AmDeviceType::Cuda);
        assert_eq!(Device::Cuda(5).to_am(), AmDeviceType::Cuda);
        assert_eq!(Device::Metal(0).to_am(), AmDeviceType::Metal);
    }

    #[test]
    fn from_am_roundtrip() {
        let cases = [
            (Device::Cpu, AmDeviceType::Cpu as i32, 0),
            (Device::Cuda(0), AmDeviceType::Cuda as i32, 0),
            (Device::Cuda(3), AmDeviceType::Cuda as i32, 3),
            (Device::Metal(0), AmDeviceType::Metal as i32, 0),
            (Device::Metal(1), AmDeviceType::Metal as i32, 1),
        ];
        for (expected, code, id) in cases {
            assert_eq!(Device::from_am(code, id).unwrap(), expected);
        }
    }

    #[test]
    fn from_am_unsupported_code() {
        let result = Device::from_am(999, 0);
        assert!(result.is_err());
    }

    #[test]
    fn eq_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Device::Cpu);
        set.insert(Device::Cuda(0));
        set.insert(Device::Cuda(1));
        set.insert(Device::Metal(0));
        assert_eq!(set.len(), 4);

        // Same device deduplicates
        set.insert(Device::Cuda(0));
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn copy_semantics() {
        let d = Device::cuda(0);
        let d2 = d;
        assert_eq!(d, d2);
    }
}

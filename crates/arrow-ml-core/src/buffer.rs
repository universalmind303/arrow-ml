//! Device-aware buffer.
//!
//! [`DeviceBuffer`] is the device-aware analog of [`arrow::buffer::Buffer`]:
//! a chunk of bytes living on some [`Device`]. Internally an enum tagged by
//! device, exposing a uniform API regardless of where the bytes physically
//! live.

use crate::device::Device;
use crate::error::{DeviceError, Result};
use arrow::buffer::MutableBuffer;
use arrow::datatypes::ArrowNativeType;
use arrow_ml_common::backend::{AmStatus, Backend};
use arrow_ml_common::device_tensor::AmDeviceType;
use arrow_ml_common::BackendRegistry;
use std::ffi::c_void;
use std::sync::Arc;

/// A buffer of bytes on some device. The device-aware analog of
/// [`arrow::buffer::Buffer`].
#[derive(Debug, Clone)]
pub struct DeviceBuffer {
    inner: DeviceBufferInner,
}

#[derive(Debug, Clone)]
enum DeviceBufferInner {
    Host(arrow::buffer::Buffer),
    Device(DeviceView),
}

/// A view into a device-resident allocation. Mirrors arrow's `Buffer`
/// layout: the `Arc<DeviceAlloc>` owns the allocation (freed on last
/// drop), while `ptr` and `length` describe the current window into it.
/// Cloning bumps the `Arc` refcount and copies the pointer + length.
#[derive(Clone)]
struct DeviceView {
    alloc: Arc<DeviceAlloc>,
    ptr: *mut c_void,
    length: usize,
}

// SAFETY: the raw pointer is managed by the backend dylib kept alive
// via Arc<DeviceAlloc> → Arc<Backend>. It is never dereferenced from
// Rust — only passed back through the backend's C ABI.
unsafe impl Send for DeviceView {}
unsafe impl Sync for DeviceView {}

impl DeviceView {
    fn new(alloc: DeviceAlloc) -> Self {
        let ptr = alloc.device_ptr;
        let length = alloc.len_bytes;
        Self {
            alloc: Arc::new(alloc),
            ptr,
            length,
        }
    }
}

impl std::fmt::Debug for DeviceView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceView")
            .field("device", &self.alloc.device)
            .field("ptr", &self.ptr)
            .field("length", &self.length)
            .field("device_id", &self.alloc.device_id)
            .finish_non_exhaustive()
    }
}

/// The underlying device allocation. Freed via the backend's
/// `device_free` when the last `Arc<DeviceAlloc>` is dropped.
pub(crate) struct DeviceAlloc {
    pub device: Device,
    pub backend: Arc<Backend>,
    pub device_ptr: *mut c_void,
    pub len_bytes: usize,
    pub device_id: i64,
}

unsafe impl Send for DeviceAlloc {}
unsafe impl Sync for DeviceAlloc {}

impl Drop for DeviceAlloc {
    fn drop(&mut self) {
        if !self.device_ptr.is_null() {
            unsafe {
                (self.backend.device_free)(self.device_ptr, self.len_bytes as u64);
            }
        }
    }
}

impl DeviceBuffer {
    /// Allocate a buffer with `capacity` bytes on `device`. The contents
    /// are uninitialized for device buffers; host buffers are zeroed.
    pub fn new(capacity: usize, device: Device) -> Self {
        match device {
            Device::Cpu => Self::from(arrow::buffer::Buffer::from_vec(vec![0u8; capacity])),
            Device::Metal(id) | Device::Cuda(id) => {
                let am_dev = device.to_am();
                let backend = BackendRegistry::global()
                    .best_for_device(am_dev, id)
                    .expect("no backend loaded for target device");

                let mut device_ptr: *mut c_void = std::ptr::null_mut();
                let rc = unsafe {
                    (backend.device_alloc)(am_dev as i32, id, capacity as u64, &mut device_ptr)
                };
                assert_eq!(rc, AmStatus::Ok as i32, "device_alloc failed (rc={rc})");

                Self {
                    inner: DeviceBufferInner::Device(DeviceView::new(DeviceAlloc {
                        device,
                        backend,
                        device_ptr,
                        len_bytes: capacity,
                        device_id: id,
                    })),
                }
            }
        }
    }

    /// Where this buffer's bytes live.
    pub fn device(&self) -> Device {
        match &self.inner {
            DeviceBufferInner::Host(_) => Device::Cpu,
            DeviceBufferInner::Device(v) => v.alloc.device,
        }
    }

    /// Returns the offset, in bytes, of the buffer's data pointer relative
    /// to the start of its underlying allocation.
    pub fn ptr_offset(&self) -> usize {
        match &self.inner {
            DeviceBufferInner::Host(b) => b.ptr_offset(),
            DeviceBufferInner::Device(v) => unsafe {
                (v.ptr as *const u8).offset_from(v.alloc.device_ptr as *const u8) as usize
            },
        }
    }

    /// Returns the number of strong references to the underlying
    /// allocation.
    pub fn strong_count(&self) -> usize {
        match &self.inner {
            DeviceBufferInner::Host(b) => b.strong_count(),
            DeviceBufferInner::Device(v) => Arc::strong_count(&v.alloc),
        }
    }

    /// Returns the number of bytes in this buffer.
    pub fn len(&self) -> usize {
        match &self.inner {
            DeviceBufferInner::Host(b) => b.len(),
            DeviceBufferInner::Device(v) => v.length,
        }
    }

    /// Returns the capacity of this buffer.
    pub fn capacity(&self) -> usize {
        match &self.inner {
            DeviceBufferInner::Host(b) => b.capacity(),
            DeviceBufferInner::Device(v) => v.alloc.len_bytes,
        }
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the byte slice stored in this buffer.
    pub fn as_slice(&self) -> Result<&[u8]> {
        match &self.inner {
            DeviceBufferInner::Host(b) => Ok(b.as_slice()),
            DeviceBufferInner::Device(v) => Err(DeviceError::NotOnHost {
                device: v.alloc.device,
            }),
        }
    }

    /// Returns a new [`DeviceBuffer`] that is a slice of this buffer
    /// starting at `offset`. `O(1)`; does not copy data.
    ///
    /// # Panics
    ///
    /// Panics iff `offset` is larger than `len`.
    pub fn slice(&self, offset: usize) -> Self {
        let mut s = self.clone();
        s.advance(offset);
        s
    }

    /// Increases the offset of this buffer by `offset`.
    ///
    /// # Panics
    ///
    /// Panics iff `offset` is larger than `len`.
    pub fn advance(&mut self, offset: usize) {
        match &mut self.inner {
            DeviceBufferInner::Host(b) => {
                b.advance(offset);
            }
            DeviceBufferInner::Device(v) => {
                assert!(
                    offset <= v.length,
                    "the offset of the new Buffer cannot exceed the existing length: \
                     offset={offset} length={}",
                    v.length
                );
                v.length -= offset;
                v.ptr = unsafe { (v.ptr as *mut u8).add(offset) as *mut c_void };
            }
        }
    }

    /// Returns a new [`DeviceBuffer`] that is a slice of this buffer
    /// starting at `offset` and is `length` bytes long. `O(1)`; does not
    /// copy data.
    ///
    /// # Panics
    ///
    /// Panics iff `(offset + length)` is larger than the existing length.
    pub fn slice_with_length(&self, offset: usize, length: usize) -> Self {
        match &self.inner {
            DeviceBufferInner::Host(b) => Self {
                inner: DeviceBufferInner::Host(b.slice_with_length(offset, length)),
            },
            DeviceBufferInner::Device(v) => {
                assert!(
                    offset.saturating_add(length) <= v.length,
                    "the offset of the new Buffer cannot exceed the existing length: \
                     slice offset={offset} length={length} selflen={}",
                    v.length
                );
                let ptr = unsafe { (v.ptr as *mut u8).add(offset) as *mut c_void };
                Self {
                    inner: DeviceBufferInner::Device(DeviceView {
                        alloc: v.alloc.clone(),
                        ptr,
                        length,
                    }),
                }
            }
        }
    }

    /// Returns a pointer to the start of this buffer.
    ///
    /// Note that this should be used cautiously, and the returned pointer
    /// should not be stored anywhere, to avoid dangling pointers.
    pub fn as_ptr(&self) -> *const u8 {
        match &self.inner {
            DeviceBufferInner::Host(b) => b.as_ptr(),
            DeviceBufferInner::Device(v) => v.ptr as *const u8,
        }
    }

    /// View buffer as a slice of a specific type.
    ///
    /// # Panics
    ///
    /// Panics if the underlying buffer is not aligned correctly for type
    /// `T`.
    pub fn typed_data<T: ArrowNativeType>(&self) -> Result<&[T]> {
        match &self.inner {
            DeviceBufferInner::Host(b) => Ok(b.typed_data::<T>()),
            DeviceBufferInner::Device(v) => Err(DeviceError::NotOnHost {
                device: v.alloc.device,
            }),
        }
    }

    /// Returns a [`MutableBuffer`] for mutating the buffer if this buffer
    /// is not shared, host-resident, and has a compatible allocation.
    pub fn into_mutable(self) -> Result<MutableBuffer> {
        let Self { inner } = self;
        match inner {
            DeviceBufferInner::Host(b) => b.into_mutable().map_err(|_| DeviceError::CannotConvert),
            DeviceBufferInner::Device(v) => Err(DeviceError::NotOnHost {
                device: v.alloc.device,
            }),
        }
    }

    /// Converts self into a `Vec`, if possible.
    pub fn into_vec<T: ArrowNativeType>(self) -> Result<Vec<T>> {
        let Self { inner } = self;
        match inner {
            DeviceBufferInner::Host(b) => b.into_vec::<T>().map_err(|_| DeviceError::CannotConvert),
            DeviceBufferInner::Device(v) => Err(DeviceError::NotOnHost {
                device: v.alloc.device,
            }),
        }
    }

    /// Returns true if this buffer is equal to `other`, using pointer
    /// comparisons. Cheaper than `PartialEq::eq` but may return false when
    /// the buffers are logically equal.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        match (&self.inner, &other.inner) {
            (DeviceBufferInner::Host(a), DeviceBufferInner::Host(b)) => a.ptr_eq(b),
            (DeviceBufferInner::Device(a), DeviceBufferInner::Device(b)) => {
                a.ptr == b.ptr && a.length == b.length
            }
            _ => false,
        }
    }

    /// Move this buffer to `device`. If the buffer already lives on the
    /// requested device, this is a cheap clone (refcount bump). Otherwise
    /// the bytes are copied across the host/device boundary via the
    /// backend's transfer ops.
    pub fn to(&self, device: Device) -> Self {
        if self.device() == device {
            return self.clone();
        }

        let nbytes = self.len();

        match (self.device(), device) {
            (Device::Cpu, Device::Metal(id)) | (Device::Cpu, Device::Cuda(id)) => {
                let am_dev = device.to_am();
                let backend = BackendRegistry::global()
                    .best_for_device(am_dev, id)
                    .expect("no backend loaded for target device");

                let mut device_ptr: *mut c_void = std::ptr::null_mut();
                let rc = unsafe {
                    (backend.device_alloc)(am_dev as i32, id, nbytes as u64, &mut device_ptr)
                };
                assert_eq!(rc, AmStatus::Ok as i32, "device_alloc failed (rc={rc})");

                let rc = unsafe {
                    (backend.device_copy)(
                        self.as_ptr() as *const c_void,
                        AmDeviceType::Cpu as i32,
                        device_ptr,
                        am_dev as i32,
                        nbytes as u64,
                    )
                };
                assert_eq!(
                    rc,
                    AmStatus::Ok as i32,
                    "host->device copy failed (rc={rc})"
                );

                Self {
                    inner: DeviceBufferInner::Device(DeviceView::new(DeviceAlloc {
                        device,
                        backend,
                        device_ptr,
                        len_bytes: nbytes,
                        device_id: id,
                    })),
                }
            }
            (Device::Metal(_) | Device::Cuda(_), Device::Cpu) => {
                let v = match &self.inner {
                    DeviceBufferInner::Device(v) => v,
                    _ => unreachable!(),
                };
                let src_am = v.alloc.device.to_am();
                let mut host_vec: Vec<u8> = vec![0u8; nbytes];
                let rc = unsafe {
                    (v.alloc.backend.device_copy)(
                        v.ptr as *const c_void,
                        src_am as i32,
                        host_vec.as_mut_ptr() as *mut c_void,
                        AmDeviceType::Cpu as i32,
                        nbytes as u64,
                    )
                };
                assert_eq!(
                    rc,
                    AmStatus::Ok as i32,
                    "device->host copy failed (rc={rc})"
                );
                arrow::buffer::Buffer::from_vec(host_vec).into()
            }
            _ => todo!(),
        }
    }
}

impl TryFrom<DeviceBuffer> for arrow::buffer::Buffer {
    type Error = DeviceError;

    fn try_from(buf: DeviceBuffer) -> Result<Self> {
        match buf.inner {
            DeviceBufferInner::Host(b) => Ok(b),
            DeviceBufferInner::Device(v) => Err(DeviceError::NotOnHost {
                device: v.alloc.device,
            }),
        }
    }
}

impl From<arrow::buffer::Buffer> for DeviceBuffer {
    fn from(buffer: arrow::buffer::Buffer) -> Self {
        Self {
            inner: DeviceBufferInner::Host(buffer),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{RefUnwindSafe, UnwindSafe};
    use std::thread;

    fn host(bytes: &[u8]) -> DeviceBuffer {
        arrow::buffer::Buffer::from(bytes).into()
    }

    #[test]
    fn test_from_raw_parts() {
        let buf = host(&[0, 1, 2, 3, 4]);
        assert_eq!(5, buf.len());
        assert!(!buf.as_ptr().is_null());
        assert_eq!([0, 1, 2, 3, 4], buf.as_slice().unwrap());
    }

    #[test]
    fn test_from_vec() {
        let buf: DeviceBuffer = arrow::buffer::Buffer::from_vec(vec![0u8, 1, 2, 3, 4]).into();
        assert_eq!(5, buf.len());
        assert!(!buf.as_ptr().is_null());
        assert_eq!([0, 1, 2, 3, 4], buf.as_slice().unwrap());
    }

    #[test]
    fn test_copy() {
        let buf = host(&[0, 1, 2, 3, 4]);
        let buf2 = buf;
        assert_eq!(5, buf2.len());
        assert_eq!(64, buf2.capacity());
        assert!(!buf2.as_ptr().is_null());
        assert_eq!([0, 1, 2, 3, 4], buf2.as_slice().unwrap());
    }

    #[test]
    fn test_slice() {
        let buf = host(&[2, 4, 6, 8, 10]);
        let buf2 = buf.slice(2);

        assert_eq!([6, 8, 10], buf2.as_slice().unwrap());
        assert_eq!(3, buf2.len());
        assert_eq!(unsafe { buf.as_ptr().offset(2) }, buf2.as_ptr());

        let buf3 = buf2.slice_with_length(1, 2);
        assert_eq!([8, 10], buf3.as_slice().unwrap());
        assert_eq!(2, buf3.len());
        assert_eq!(unsafe { buf.as_ptr().offset(3) }, buf3.as_ptr());

        let buf4 = buf.slice(5);
        let empty_slice: [u8; 0] = [];
        assert_eq!(empty_slice, buf4.as_slice().unwrap());
        assert_eq!(0, buf4.len());
        assert!(buf4.is_empty());
        assert_eq!(buf2.slice_with_length(2, 1).as_slice().unwrap(), &[10]);
    }

    #[test]
    #[should_panic(expected = "the offset of the new Buffer cannot exceed the existing length")]
    fn test_slice_offset_out_of_bound() {
        let buf = host(&[2, 4, 6, 8, 10]);
        buf.slice(6);
    }

    #[test]
    fn test_access_concurrently() {
        let buffer = host(&[1, 2, 3, 4, 5]);
        let buffer2 = buffer.clone();
        assert_eq!([1, 2, 3, 4, 5], buffer.as_slice().unwrap());

        let buffer_copy = thread::spawn(move || buffer).join();

        assert!(buffer_copy.is_ok());
        assert_eq!(
            buffer2.as_slice().unwrap(),
            buffer_copy.ok().unwrap().as_slice().unwrap()
        );
    }

    macro_rules! check_as_typed_data {
        ($input: expr, $native_t: ty) => {{
            let buffer: DeviceBuffer = arrow::buffer::Buffer::from_slice_ref($input).into();
            let slice: &[$native_t] = buffer.typed_data::<$native_t>().unwrap();
            assert_eq!($input, slice);
        }};
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_as_typed_data() {
        check_as_typed_data!(&[1i8, 3i8, 6i8], i8);
        check_as_typed_data!(&[1u8, 3u8, 6u8], u8);
        check_as_typed_data!(&[1i16, 3i16, 6i16], i16);
        check_as_typed_data!(&[1i32, 3i32, 6i32], i32);
        check_as_typed_data!(&[1i64, 3i64, 6i64], i64);
        check_as_typed_data!(&[1u16, 3u16, 6u16], u16);
        check_as_typed_data!(&[1u32, 3u32, 6u32], u32);
        check_as_typed_data!(&[1u64, 3u64, 6u64], u64);
        check_as_typed_data!(&[1f32, 3f32, 6f32], f32);
        check_as_typed_data!(&[1f64, 3f64, 6f64], f64);
    }

    #[test]
    fn test_unwind_safe() {
        fn assert_unwind_safe<T: RefUnwindSafe + UnwindSafe>() {}
        assert_unwind_safe::<DeviceBuffer>()
    }

    #[test]
    #[should_panic(expected = "the offset of the new Buffer cannot exceed the existing length")]
    fn slice_overflow() {
        let buffer: DeviceBuffer =
            arrow::buffer::Buffer::from(MutableBuffer::from_len_zeroed(12)).into();
        buffer.slice_with_length(2, usize::MAX);
    }

    #[test]
    fn test_vec_interop() {
        // Empty vec
        let a: Vec<i128> = Vec::new();
        let b: DeviceBuffer = arrow::buffer::Buffer::from_vec(a).into();
        b.into_vec::<i128>().unwrap();

        // Vec with values
        let mut a: Vec<i128> = Vec::with_capacity(3);
        a.extend_from_slice(&[1, 2, 3]);
        let b: DeviceBuffer = arrow::buffer::Buffer::from_vec(a).into();
        let back = b.into_vec::<i128>().unwrap();
        assert_eq!(back.len(), 3);
        assert_eq!(back.capacity(), 3);

        // Sharing prevents conversion
        let b: DeviceBuffer = arrow::buffer::Buffer::from_vec(vec![1u32, 2, 3]).into();
        let _shared = b.clone();
        b.into_vec::<u32>().unwrap_err();

        // Offset prevents conversion
        let a: Vec<u32> = vec![1, 3, 4, 6];
        let b: DeviceBuffer = arrow::buffer::Buffer::from_vec(a).into();
        b.slice(8).into_vec::<u32>().unwrap_err();
    }

    #[test]
    fn test_strong_count() {
        let buffer: DeviceBuffer =
            arrow::buffer::Buffer::from_iter(std::iter::repeat_n(0_u8, 100)).into();
        assert_eq!(buffer.strong_count(), 1);

        let buffer2 = buffer.clone();
        assert_eq!(buffer.strong_count(), 2);

        let buffer3 = buffer2.clone();
        assert_eq!(buffer.strong_count(), 3);

        drop(buffer);
        assert_eq!(buffer2.strong_count(), 2);
        assert_eq!(buffer3.strong_count(), 2);
    }

    #[test]
    fn test_ptr_eq() {
        let buffer = host(&[1, 2, 3, 4, 5]);
        let same = buffer.clone();
        assert!(buffer.ptr_eq(&same));

        let other = host(&[1, 2, 3, 4, 5]);
        assert!(!buffer.ptr_eq(&other));

        let sliced = buffer.slice(1);
        assert!(!buffer.ptr_eq(&sliced));
    }

    #[test]
    fn test_into_mutable() {
        let buffer: DeviceBuffer = arrow::buffer::Buffer::from_vec(vec![1u32, 2, 3]).into();
        let mutable = buffer.into_mutable().unwrap();
        let buffer: DeviceBuffer = arrow::buffer::Buffer::from(mutable).into();
        assert_eq!(buffer.typed_data::<u32>().unwrap(), &[1, 2, 3]);

        // Sharing prevents conversion
        let buffer: DeviceBuffer = arrow::buffer::Buffer::from_vec(vec![1u32, 2, 3]).into();
        let _shared = buffer.clone();
        buffer.into_mutable().unwrap_err();
    }

    #[test]
    fn test_to_same_device_clones() {
        let buffer = host(&[1, 2, 3, 4, 5]);
        let moved = buffer.to(Device::Cpu);
        assert!(buffer.ptr_eq(&moved));
        assert_eq!(buffer.strong_count(), 2);
    }

    // ---------------------------------------------------------------
    // Device-path tests (require a loaded GPU backend)
    // ---------------------------------------------------------------

    fn has_metal() -> bool {
        dbg!(BackendRegistry::global()
            .loaded_backends())
            .contains(&"metal")
    }

    fn host_f32(data: &[f32]) -> DeviceBuffer {
        arrow::buffer::Buffer::from_slice_ref(data).into()
    }

    #[test]
    fn device_to_returns_different_pointer() {
        if !has_metal() {
            return;
        }
        let host = host_f32(&[1.0, 2.0, 3.0]);
        let device = host.to(Device::metal(0));

        assert_ne!(
            host.as_ptr(),
            device.as_ptr(),
            "device buffer must be a separate allocation"
        );
    }

    #[test]
    fn device_reports_correct_device() {
        if !has_metal() {
            return;
        }
        let host = host_f32(&[1.0]);
        let device = host.to(Device::metal(0));
        assert_eq!(device.device(), Device::Metal(0));
    }

    #[test]
    fn device_preserves_length() {
        if !has_metal() {
            return;
        }
        let host = host_f32(&[1.0, 2.0, 3.0, 4.0]);
        let device = host.to(Device::metal(0));
        assert_eq!(device.len(), host.len());
    }

    #[test]
    fn device_as_slice_errors() {
        if !has_metal() {
            return;
        }
        let device = host_f32(&[1.0]).to(Device::metal(0));
        assert!(device.as_slice().is_err());
    }

    #[test]
    fn device_typed_data_errors() {
        if !has_metal() {
            return;
        }
        let device = host_f32(&[1.0]).to(Device::metal(0));
        assert!(device.typed_data::<f32>().is_err());
    }

    #[test]
    fn device_into_mutable_errors() {
        if !has_metal() {
            return;
        }
        let device = host_f32(&[1.0]).to(Device::metal(0));
        assert!(device.into_mutable().is_err());
    }

    #[test]
    fn device_into_vec_errors() {
        if !has_metal() {
            return;
        }
        let device = host_f32(&[1.0]).to(Device::metal(0));
        assert!(device.into_vec::<f32>().is_err());
    }

    #[test]
    fn device_try_into_arrow_buffer_errors() {
        if !has_metal() {
            return;
        }
        let device = host_f32(&[1.0]).to(Device::metal(0));
        let result: std::result::Result<arrow::buffer::Buffer, _> = device.try_into();
        assert!(result.is_err());
    }

    #[test]
    fn device_round_trip_preserves_data() {
        if !has_metal() {
            return;
        }
        let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        let host = host_f32(&data);
        let device = host.to(Device::metal(0));
        let back = device.to(Device::cpu());

        assert_eq!(back.device(), Device::Cpu);
        assert_eq!(back.typed_data::<f32>().unwrap(), &data[..]);
    }

    #[test]
    fn device_round_trip_returns_new_allocation() {
        if !has_metal() {
            return;
        }
        let host = host_f32(&[1.0, 2.0, 3.0]);
        let back = host.to(Device::metal(0)).to(Device::cpu());

        assert_ne!(
            host.as_ptr(),
            back.as_ptr(),
            "round-tripped buffer must be a fresh host allocation"
        );
        assert!(!host.ptr_eq(&back));
    }

    #[test]
    fn device_clone_shares_allocation() {
        if !has_metal() {
            return;
        }
        let device = host_f32(&[1.0, 2.0]).to(Device::metal(0));
        let cloned = device.clone();

        assert!(device.ptr_eq(&cloned));
        assert_eq!(device.strong_count(), 2);
    }

    #[test]
    fn device_clone_refcount_lifecycle() {
        if !has_metal() {
            return;
        }
        let device = host_f32(&[1.0]).to(Device::metal(0));
        assert_eq!(device.strong_count(), 1);

        let c2 = device.clone();
        assert_eq!(device.strong_count(), 2);

        let c3 = c2.clone();
        assert_eq!(device.strong_count(), 3);

        drop(c2);
        assert_eq!(device.strong_count(), 2);

        drop(c3);
        assert_eq!(device.strong_count(), 1);
    }

    #[test]
    fn device_to_same_device_is_cheap() {
        if !has_metal() {
            return;
        }
        let device = host_f32(&[1.0, 2.0]).to(Device::metal(0));
        let same = device.to(Device::metal(0));

        assert!(device.ptr_eq(&same));
        assert_eq!(device.strong_count(), 2);
    }

    #[test]
    fn device_slice_preserves_device() {
        if !has_metal() {
            return;
        }
        let device = host_f32(&[1.0, 2.0, 3.0, 4.0]).to(Device::metal(0));
        let sliced = device.slice(8); // skip 2 floats
        assert_eq!(sliced.device(), Device::Metal(0));
        assert_eq!(sliced.len(), 8); // 2 floats remaining
    }

    #[test]
    fn device_slice_with_length_round_trip() {
        if !has_metal() {
            return;
        }
        let data = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let device = host_f32(&data).to(Device::metal(0));

        let middle = device.slice_with_length(4, 12); // floats [1..4]
        assert_eq!(middle.len(), 12);

        let back = middle.to(Device::cpu());
        assert_eq!(back.typed_data::<f32>().unwrap(), &[20.0, 30.0, 40.0]);
    }

    #[test]
    fn device_slice_shares_underlying_alloc() {
        if !has_metal() {
            return;
        }
        let device = host_f32(&[1.0, 2.0, 3.0]).to(Device::metal(0));
        let sliced = device.slice(4);

        // Slice bumps refcount on the same underlying allocation
        assert_eq!(device.strong_count(), 2);

        drop(sliced);
        assert_eq!(device.strong_count(), 1);
    }

    #[test]
    fn device_ptr_offset() {
        if !has_metal() {
            return;
        }
        let device = host_f32(&[1.0, 2.0, 3.0, 4.0]).to(Device::metal(0));
        assert_eq!(device.ptr_offset(), 0);

        let sliced = device.slice(8);
        assert_eq!(sliced.ptr_offset(), 8);
    }

    #[test]
    fn device_empty_buffer() {
        if !has_metal() {
            return;
        }
        let empty = host_f32(&[]);
        dbg!(&empty);
        let empty = empty.to(Device::metal(0));
        dbg!(&empty);

        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
        assert_eq!(empty.device(), Device::Metal(0));

        let back = empty.to(Device::cpu());
        assert!(back.is_empty());
    }
}

//! Immutable native storage crosses the media/inference threads without copying.
//! A CPU mutation takes an owned copy, so retained decoder/GPU buffers never alias
//! mutable Rust data. The native release callback returns buffers to their pool.
use std::{
    ffi::c_void,
    ops::{Deref, DerefMut},
    sync::Arc,
};

#[derive(Debug)]
pub struct NativeStorage {
    owner: *mut c_void,
    pixels: *const u8,
    len: usize,
    pixel_buffer: *mut c_void,
    release: unsafe extern "C" fn(*mut c_void),
}

// Construction requires immutable CPU-readable storage until the final release.
// Both CoreVideo retain/release and the Metal buffer pool are thread safe.
unsafe impl Send for NativeStorage {}
unsafe impl Sync for NativeStorage {}

impl Drop for NativeStorage {
    fn drop(&mut self) {
        unsafe { (self.release)(self.owner) }
    }
}

#[derive(Debug, Clone)]
pub enum FrameData {
    Owned(Vec<u8>),
    Native(Arc<NativeStorage>),
}
impl Default for FrameData {
    fn default() -> Self {
        Self::Owned(Vec::new())
    }
}
impl From<Vec<u8>> for FrameData {
    fn from(v: Vec<u8>) -> Self {
        Self::Owned(v)
    }
}
impl FrameData {
    /// # Safety
    /// `pixels` must be valid, immutable, CPU-readable storage for `len` bytes.
    /// `owner` owns that storage until `release(owner)` runs exactly once, on any
    /// thread. A non-null `pixel_buffer` must be a retained BGRA CVPixelBuffer
    /// owned by the same lifetime. GPU producers must finish before construction.
    pub(crate) unsafe fn from_native(
        owner: *mut c_void,
        pixels: *const u8,
        len: usize,
        pixel_buffer: *mut c_void,
        release: unsafe extern "C" fn(*mut c_void),
    ) -> Self {
        Self::Native(Arc::new(NativeStorage {
            owner,
            pixels,
            len,
            pixel_buffer,
            release,
        }))
    }
    pub(crate) fn pixel_buffer(&self) -> *mut c_void {
        match self {
            Self::Native(v) => v.pixel_buffer,
            Self::Owned(_) => std::ptr::null_mut(),
        }
    }
    pub(crate) fn take_owned(&mut self) -> Option<Vec<u8>> {
        match self {
            Self::Owned(v) => Some(std::mem::take(v)),
            Self::Native(_) => None,
        }
    }
    fn owned_mut(&mut self) -> &mut Vec<u8> {
        if matches!(self, Self::Native(_)) {
            *self = Self::Owned(self.to_vec());
        }
        match self {
            Self::Owned(v) => v,
            Self::Native(_) => unreachable!(),
        }
    }
    #[cfg(test)]
    pub(crate) fn truncate(&mut self, len: usize) {
        self.owned_mut().truncate(len);
    }
}
impl Deref for FrameData {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v,
            Self::Native(v) => unsafe { std::slice::from_raw_parts(v.pixels, v.len) },
        }
    }
}
impl DerefMut for FrameData {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.owned_mut()
    }
}
impl PartialEq for FrameData {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Owner {
        pixels: [u8; 4],
        released: Arc<AtomicUsize>,
    }
    unsafe extern "C" fn release(owner: *mut c_void) {
        let owner = unsafe { Box::from_raw(owner.cast::<Owner>()) };
        owner.released.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn native_storage_is_immutable_and_released_once_across_threads() {
        let released = Arc::new(AtomicUsize::new(0));
        let owner = Box::into_raw(Box::new(Owner {
            pixels: [10, 20, 30, 255],
            released: released.clone(),
        }));
        let pixels = unsafe { (*owner).pixels.as_ptr() };
        let data = unsafe {
            FrameData::from_native(owner.cast(), pixels, 4, std::ptr::null_mut(), release)
        };
        let retained = data.clone();
        let mut edited = data.clone();
        edited[0] = 99;
        assert_eq!(data[0], 10);
        assert_eq!(edited[0], 99);
        edited.truncate(3);
        assert_eq!(edited.len(), 3);
        assert_eq!(data.len(), 4);
        drop(data);
        assert_eq!(released.load(Ordering::SeqCst), 0);
        std::thread::spawn(move || {
            assert_eq!(&*retained, &[10, 20, 30, 255]);
            drop(retained);
        })
        .join()
        .unwrap();
        assert_eq!(released.load(Ordering::SeqCst), 1);
        drop(edited);
        assert_eq!(released.load(Ordering::SeqCst), 1);
    }
}

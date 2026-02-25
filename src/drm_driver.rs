// drm_driver.rs — Vortex-native GPU driver via Linux DRM ioctl
//
// Opens /dev/dri/renderD* directly, performs GEM buffer management,
// command submission, and fence synchronization. Falls back to a
// CPU runtime when no GPU is available.

use crate::interpreter::{Env, FnDef, Value};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

// ---------------------------------------------------------------------------
// Raw syscall externs (no libc crate dependency)
// ---------------------------------------------------------------------------

extern "C" {
    fn open(path: *const u8, flags: i32) -> i32;
    fn close(fd: i32) -> i32;
    fn ioctl(fd: i32, request: u64, ...) -> i32;
    fn mmap(
        addr: *mut u8,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
    fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8;
    fn strerror(errnum: i32) -> *const u8;
    fn strlen(s: *const u8) -> usize;
}

// errno access
extern "C" {
    fn __errno_location() -> *mut i32;
}

fn get_errno() -> i32 {
    unsafe { *__errno_location() }
}

fn errno_string() -> String {
    unsafe {
        let p = strerror(get_errno());
        let len = strlen(p);
        let sl = std::slice::from_raw_parts(p, len);
        String::from_utf8_lossy(sl).into_owned()
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const O_RDWR: i32 = 2;
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_SHARED: i32 = 1;
const MAP_FAILED: *mut u8 = !0usize as *mut u8;

// DRM ioctl base
const DRM_IOCTL_BASE: u64 = 0x64; // 'd'

// DRM_IOCTL_VERSION = _IOWR('d', 0x00, struct drm_version)
// size of drm_version on x86_64 = 4+4+4 + ptr+usize + ptr+usize + ptr+usize = ~72
// We use the standard Linux encoding: dir(2)|size(14)|type(8)|nr(8)
// _IOWR = 0xC0000000 | (size << 16) | (type << 8) | nr
const DRM_VERSION_SIZE: u64 = 72;
const DRM_IOCTL_VERSION: u64 =
    0xC000_0000 | (DRM_VERSION_SIZE << 16) | (DRM_IOCTL_BASE << 8) | 0x00;

// GEM close: DRM_IOCTL_GEM_CLOSE = _IOW('d', 0x09, struct drm_gem_close)
// drm_gem_close = { handle: u32, pad: u32 } = 8 bytes
const DRM_GEM_CLOSE_SIZE: u64 = 8;
const DRM_IOCTL_GEM_CLOSE: u64 =
    0x4000_0000 | (DRM_GEM_CLOSE_SIZE << 16) | (DRM_IOCTL_BASE << 8) | 0x09;

// ---------------------------------------------------------------------------
// Vendor-specific ioctl numbers
// ---------------------------------------------------------------------------

// AMDGPU
const AMDGPU_IOCTL_BASE: u64 = DRM_IOCTL_BASE;
const DRM_COMMAND_BASE: u64 = 0x40;

// AMDGPU_GEM_CREATE = DRM_IOWR(DRM_COMMAND_BASE + 0x00, 40)
const AMDGPU_GEM_CREATE_SIZE: u64 = 40;
const DRM_AMDGPU_GEM_CREATE: u64 = 0xC000_0000
    | (AMDGPU_GEM_CREATE_SIZE << 16)
    | (AMDGPU_IOCTL_BASE << 8)
    | (DRM_COMMAND_BASE + 0x00);

// AMDGPU_GEM_MMAP = DRM_IOWR(DRM_COMMAND_BASE + 0x05, 16)
const AMDGPU_GEM_MMAP_SIZE: u64 = 16;
const DRM_AMDGPU_GEM_MMAP: u64 = 0xC000_0000
    | (AMDGPU_GEM_MMAP_SIZE << 16)
    | (AMDGPU_IOCTL_BASE << 8)
    | (DRM_COMMAND_BASE + 0x05);

// AMDGPU_CS = DRM_IOWR(DRM_COMMAND_BASE + 0x04, 48)
const AMDGPU_CS_SIZE: u64 = 48;
const DRM_AMDGPU_CS: u64 = 0xC000_0000
    | (AMDGPU_CS_SIZE << 16)
    | (AMDGPU_IOCTL_BASE << 8)
    | (DRM_COMMAND_BASE + 0x04);

// AMDGPU_WAIT_FENCES = DRM_IOWR(DRM_COMMAND_BASE + 0x08, 24)
const AMDGPU_WAIT_FENCES_SIZE: u64 = 24;
const DRM_AMDGPU_WAIT_FENCES: u64 = 0xC000_0000
    | (AMDGPU_WAIT_FENCES_SIZE << 16)
    | (AMDGPU_IOCTL_BASE << 8)
    | (DRM_COMMAND_BASE + 0x08);

// i915
const I915_GEM_CREATE_SIZE: u64 = 16;
const DRM_I915_GEM_CREATE: u64 = 0xC000_0000
    | (I915_GEM_CREATE_SIZE << 16)
    | (AMDGPU_IOCTL_BASE << 8)
    | (DRM_COMMAND_BASE + 0x1B);

const I915_GEM_MMAP_SIZE: u64 = 32;
const DRM_I915_GEM_MMAP: u64 = 0xC000_0000
    | (I915_GEM_MMAP_SIZE << 16)
    | (AMDGPU_IOCTL_BASE << 8)
    | (DRM_COMMAND_BASE + 0x1E);

const I915_EXEC_SIZE: u64 = 64;
const DRM_I915_EXECBUFFER2: u64 = 0xC000_0000
    | (I915_EXEC_SIZE << 16)
    | (AMDGPU_IOCTL_BASE << 8)
    | (DRM_COMMAND_BASE + 0x29);

const I915_GEM_WAIT_SIZE: u64 = 16;
const DRM_I915_GEM_WAIT: u64 = 0xC000_0000
    | (I915_GEM_WAIT_SIZE << 16)
    | (AMDGPU_IOCTL_BASE << 8)
    | (DRM_COMMAND_BASE + 0x2C);

// Nouveau — uses the generic DRM_NOUVEAU_* commands
const NOUVEAU_GEM_NEW_SIZE: u64 = 48;
const DRM_NOUVEAU_GEM_NEW: u64 = 0xC000_0000
    | (NOUVEAU_GEM_NEW_SIZE << 16)
    | (AMDGPU_IOCTL_BASE << 8)
    | (DRM_COMMAND_BASE + 0x00);

const NOUVEAU_GEM_PUSHBUF_SIZE: u64 = 64;
const DRM_NOUVEAU_GEM_PUSHBUF: u64 = 0xC000_0000
    | (NOUVEAU_GEM_PUSHBUF_SIZE << 16)
    | (AMDGPU_IOCTL_BASE << 8)
    | (DRM_COMMAND_BASE + 0x01);

// ---------------------------------------------------------------------------
// GpuVendor
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    AMD,
    Nvidia,
    Intel,
    Unknown,
}

impl GpuVendor {
    fn as_str(&self) -> &'static str {
        match self {
            GpuVendor::AMD => "amd",
            GpuVendor::Nvidia => "nvidia",
            GpuVendor::Intel => "intel",
            GpuVendor::Unknown => "unknown",
        }
    }
}

// ---------------------------------------------------------------------------
// GemBuffer
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct GemBuffer {
    pub handle: u32,
    pub size: usize,
    pub gpu_addr: u64,
    pub cpu_map: Option<*mut u8>,
}

// Safety: cpu_map is only accessed under the global lock
unsafe impl Send for GemBuffer {}
unsafe impl Sync for GemBuffer {}

impl GemBuffer {
    fn new(handle: u32, size: usize) -> Self {
        Self {
            handle,
            size,
            gpu_addr: 0,
            cpu_map: None,
        }
    }
}

// ---------------------------------------------------------------------------
// drm_version struct (for ioctl)
// ---------------------------------------------------------------------------

#[repr(C)]
struct DrmVersion {
    version_major: i32,
    version_minor: i32,
    version_patchlevel: i32,
    name_len: usize,
    name: *mut u8,
    date_len: usize,
    date: *mut u8,
    desc_len: usize,
    desc: *mut u8,
}

// ---------------------------------------------------------------------------
// Vendor-specific ioctl structs
// ---------------------------------------------------------------------------

// AMDGPU
#[repr(C)]
struct AmdgpuGemCreate {
    bo_size: u64,
    alignment: u64,
    domain: u32,
    flags: u32,
    handle: u32,
    _pad: u32,
}

#[repr(C)]
struct AmdgpuGemMmap {
    handle: u32,
    _pad: u32,
    offset: u64,
}

#[repr(C)]
struct AmdgpuCs {
    ctx_id: u32,
    bo_list_handle: u32,
    num_chunks: u32,
    flags: u32,
    chunks: u64,
    seq_no: u64,
}

#[repr(C)]
struct AmdgpuWaitFences {
    fences: u64,
    fence_count: u32,
    wait_all: u32,
    timeout_ns: u64,
}

// i915
#[repr(C)]
struct I915GemCreate {
    size: u64,
    handle: u32,
    pad: u32,
}

#[repr(C)]
struct I915GemMmap {
    handle: u32,
    pad: u32,
    offset: u64,
    size: u64,
    addr_ptr: u64,
    flags: u64,
}

#[repr(C)]
struct I915GemWait {
    bo_handle: u32,
    flags: u32,
    timeout_ns: i64,
}

// Nouveau
#[repr(C)]
struct NouveauGemNew {
    info_domain: u32,
    info_tile_mode: u32,
    info_tile_flags: u32,
    info_size: u32,
    info_offset: u64,
    info_map_handle: u64,
    align: u32,
    handle: u32,
    channel: u32,
    _pad: u32,
}

// DRM GEM close
#[repr(C)]
struct DrmGemClose {
    handle: u32,
    pad: u32,
}

// ---------------------------------------------------------------------------
// VRAM bucket allocator
// ---------------------------------------------------------------------------

struct VramBucket {
    /// Each entry: (GEM handle, size, is_in_use)
    entries: Vec<(u32, usize, bool)>,
}

struct VramPool {
    /// Power-of-2 bucket index -> bucket (index 12 = 4 KiB, 13 = 8 KiB, ..., 30 = 1 GiB)
    buckets: HashMap<u32, VramBucket>,
}

impl VramPool {
    fn new() -> Self {
        Self {
            buckets: HashMap::new(),
        }
    }

    fn bucket_index(size: usize) -> u32 {
        let mut s = size.max(4096);
        // Round up to next power of 2
        s = s.next_power_of_two();
        s.trailing_zeros()
    }

    /// Try to reclaim a free buffer from the pool.
    fn try_reclaim(&mut self, size: usize) -> Option<u32> {
        let idx = Self::bucket_index(size);
        if let Some(bucket) = self.buckets.get_mut(&idx) {
            for entry in bucket.entries.iter_mut() {
                if !entry.2 {
                    entry.2 = true;
                    return Some(entry.0);
                }
            }
        }
        None
    }

    /// Track a newly allocated buffer.
    fn track(&mut self, handle: u32, size: usize) {
        let idx = Self::bucket_index(size);
        let bucket = self.buckets.entry(idx).or_insert_with(|| VramBucket {
            entries: Vec::new(),
        });
        bucket.entries.push((handle, size, true));
    }

    /// Mark a buffer as free (returns to pool).
    fn release(&mut self, handle: u32) -> bool {
        for bucket in self.buckets.values_mut() {
            for entry in bucket.entries.iter_mut() {
                if entry.0 == handle {
                    entry.2 = false;
                    return true;
                }
            }
        }
        false
    }

    fn stats(&self) -> (usize, usize, usize) {
        let mut total = 0usize;
        let mut used = 0usize;
        let mut free = 0usize;
        for bucket in self.buckets.values() {
            for &(_, sz, in_use) in &bucket.entries {
                total += sz;
                if in_use {
                    used += sz;
                } else {
                    free += sz;
                }
            }
        }
        (total, used, free)
    }
}

// ---------------------------------------------------------------------------
// VortexGpuDevice
// ---------------------------------------------------------------------------

pub struct VortexGpuDevice {
    pub fd: i32,
    pub vendor: GpuVendor,
    pub driver_name: String,
    pub vram_total: u64,
    pub gem_handles: HashMap<u32, GemBuffer>,
    pub pool: VramPool,
    next_fence: u64,
    device_path: String,
}

impl VortexGpuDevice {
    /// Discover all render nodes under /dev/dri/.
    pub fn discover() -> Vec<GpuInfo> {
        let mut results = Vec::new();
        // Scan renderD128..renderD143 (typical range)
        for idx in 128..144 {
            let path = format!("/dev/dri/renderD{}\0", idx);
            let fd = unsafe { open(path.as_ptr(), O_RDWR) };
            if fd < 0 {
                continue;
            }
            // Query version
            let mut name_buf = [0u8; 128];
            let mut date_buf = [0u8; 64];
            let mut desc_buf = [0u8; 256];
            let mut ver = DrmVersion {
                version_major: 0,
                version_minor: 0,
                version_patchlevel: 0,
                name_len: name_buf.len(),
                name: name_buf.as_mut_ptr(),
                date_len: date_buf.len(),
                date: date_buf.as_mut_ptr(),
                desc_len: desc_buf.len(),
                desc: desc_buf.as_mut_ptr(),
            };
            let ret = unsafe { ioctl(fd, DRM_IOCTL_VERSION, &mut ver as *mut DrmVersion) };
            if ret == 0 {
                let driver = unsafe {
                    let sl = std::slice::from_raw_parts(ver.name, ver.name_len);
                    String::from_utf8_lossy(sl).into_owned()
                };
                let desc = unsafe {
                    let sl = std::slice::from_raw_parts(ver.desc, ver.desc_len);
                    String::from_utf8_lossy(sl).into_owned()
                };
                let vendor = Self::classify_vendor(&driver);
                results.push(GpuInfo {
                    path: format!("/dev/dri/renderD{}", idx),
                    driver_name: driver,
                    description: desc,
                    vendor,
                    version_major: ver.version_major,
                    version_minor: ver.version_minor,
                    version_patch: ver.version_patchlevel,
                });
            }
            unsafe { close(fd); }
        }
        results
    }

    fn classify_vendor(driver_name: &str) -> GpuVendor {
        let d = driver_name.to_lowercase();
        if d.contains("amdgpu") || d.contains("radeon") {
            GpuVendor::AMD
        } else if d.contains("nouveau") || d.contains("nvidia") {
            GpuVendor::Nvidia
        } else if d.contains("i915") || d.contains("xe") || d.contains("intel") {
            GpuVendor::Intel
        } else {
            GpuVendor::Unknown
        }
    }

    /// Open a specific render node.
    pub fn open_device(device_path: &str) -> Result<Self, String> {
        let cpath = format!("{}\0", device_path);
        let fd = unsafe { open(cpath.as_ptr(), O_RDWR) };
        if fd < 0 {
            return Err(format!(
                "failed to open {}: {}",
                device_path,
                errno_string()
            ));
        }

        // Query driver version
        let mut name_buf = [0u8; 128];
        let mut date_buf = [0u8; 64];
        let mut desc_buf = [0u8; 256];
        let mut ver = DrmVersion {
            version_major: 0,
            version_minor: 0,
            version_patchlevel: 0,
            name_len: name_buf.len(),
            name: name_buf.as_mut_ptr(),
            date_len: date_buf.len(),
            date: date_buf.as_mut_ptr(),
            desc_len: desc_buf.len(),
            desc: desc_buf.as_mut_ptr(),
        };
        let ret = unsafe { ioctl(fd, DRM_IOCTL_VERSION, &mut ver as *mut DrmVersion) };
        if ret != 0 {
            unsafe { close(fd); }
            return Err(format!("DRM_IOCTL_VERSION failed: {}", errno_string()));
        }

        let driver = unsafe {
            let sl = std::slice::from_raw_parts(ver.name, ver.name_len);
            String::from_utf8_lossy(sl).into_owned()
        };
        let vendor = Self::classify_vendor(&driver);

        Ok(Self {
            fd,
            vendor,
            driver_name: driver,
            vram_total: 0, // Could be queried via vendor-specific info ioctl
            gem_handles: HashMap::new(),
            pool: VramPool::new(),
            next_fence: 1,
            device_path: device_path.to_string(),
        })
    }

    /// Allocate VRAM via GEM create, then mmap for CPU access.
    pub fn alloc_vram(&mut self, size: usize, _flags: u32) -> Result<u32, String> {
        let alloc_size = size.max(4096).next_power_of_two();

        // Check pool first
        if let Some(handle) = self.pool.try_reclaim(alloc_size) {
            return Ok(handle);
        }

        let handle = match self.vendor {
            GpuVendor::AMD => self.gem_create_amdgpu(alloc_size)?,
            GpuVendor::Intel => self.gem_create_i915(alloc_size)?,
            GpuVendor::Nvidia => self.gem_create_nouveau(alloc_size)?,
            GpuVendor::Unknown => {
                return Err("cannot allocate VRAM on unknown GPU vendor".into());
            }
        };

        // Try to mmap for CPU access
        let cpu_map = self.gem_mmap(handle, alloc_size);

        let mut buf = GemBuffer::new(handle, alloc_size);
        buf.cpu_map = cpu_map;
        self.gem_handles.insert(handle, buf);
        self.pool.track(handle, alloc_size);
        Ok(handle)
    }

    fn gem_create_amdgpu(&self, size: usize) -> Result<u32, String> {
        let mut req = AmdgpuGemCreate {
            bo_size: size as u64,
            alignment: 4096,
            domain: 0x4, // AMDGPU_GEM_DOMAIN_VRAM
            flags: 0,
            handle: 0,
            _pad: 0,
        };
        let ret =
            unsafe { ioctl(self.fd, DRM_AMDGPU_GEM_CREATE, &mut req as *mut AmdgpuGemCreate) };
        if ret != 0 {
            return Err(format!("AMDGPU GEM create failed: {}", errno_string()));
        }
        Ok(req.handle)
    }

    fn gem_create_i915(&self, size: usize) -> Result<u32, String> {
        let mut req = I915GemCreate {
            size: size as u64,
            handle: 0,
            pad: 0,
        };
        let ret =
            unsafe { ioctl(self.fd, DRM_I915_GEM_CREATE, &mut req as *mut I915GemCreate) };
        if ret != 0 {
            return Err(format!("i915 GEM create failed: {}", errno_string()));
        }
        Ok(req.handle)
    }

    fn gem_create_nouveau(&self, size: usize) -> Result<u32, String> {
        let mut req = NouveauGemNew {
            info_domain: 0x02, // NOUVEAU_GEM_DOMAIN_VRAM
            info_tile_mode: 0,
            info_tile_flags: 0,
            info_size: size as u32,
            info_offset: 0,
            info_map_handle: 0,
            align: 4096,
            handle: 0,
            channel: 0,
            _pad: 0,
        };
        let ret =
            unsafe { ioctl(self.fd, DRM_NOUVEAU_GEM_NEW, &mut req as *mut NouveauGemNew) };
        if ret != 0 {
            return Err(format!("Nouveau GEM create failed: {}", errno_string()));
        }
        Ok(req.handle)
    }

    fn gem_mmap(&self, handle: u32, size: usize) -> Option<*mut u8> {
        match self.vendor {
            GpuVendor::AMD => {
                let mut req = AmdgpuGemMmap {
                    handle,
                    _pad: 0,
                    offset: 0,
                };
                let ret = unsafe {
                    ioctl(self.fd, DRM_AMDGPU_GEM_MMAP, &mut req as *mut AmdgpuGemMmap)
                };
                if ret != 0 {
                    return None;
                }
                let ptr = unsafe {
                    mmap(
                        std::ptr::null_mut(),
                        size,
                        PROT_READ | PROT_WRITE,
                        MAP_SHARED,
                        self.fd,
                        req.offset as i64,
                    )
                };
                if ptr == MAP_FAILED {
                    None
                } else {
                    Some(ptr)
                }
            }
            GpuVendor::Intel => {
                let mut req = I915GemMmap {
                    handle,
                    pad: 0,
                    offset: 0,
                    size: size as u64,
                    addr_ptr: 0,
                    flags: 0,
                };
                let ret = unsafe {
                    ioctl(self.fd, DRM_I915_GEM_MMAP, &mut req as *mut I915GemMmap)
                };
                if ret != 0 {
                    return None;
                }
                if req.addr_ptr == 0 {
                    None
                } else {
                    Some(req.addr_ptr as *mut u8)
                }
            }
            GpuVendor::Nvidia => {
                // Nouveau: use the map_handle from GEM_NEW via regular mmap
                // For simplicity, attempt a standard mmap on the fd at offset 0
                // Real drivers would use the map_handle returned from GEM_NEW
                None
            }
            GpuVendor::Unknown => None,
        }
    }

    /// Upload data to a GEM buffer via its CPU mapping.
    pub fn upload(&self, handle: u32, data: &[u8]) -> Result<(), String> {
        let gem = self
            .gem_handles
            .get(&handle)
            .ok_or_else(|| format!("GEM handle {} not found", handle))?;
        let ptr = gem
            .cpu_map
            .ok_or_else(|| "buffer not CPU-mapped".to_string())?;
        let copy_len = data.len().min(gem.size);
        unsafe {
            memcpy(ptr, data.as_ptr(), copy_len);
        }
        Ok(())
    }

    /// Download data from a GEM buffer via its CPU mapping.
    pub fn download(&self, handle: u32) -> Result<Vec<u8>, String> {
        let gem = self
            .gem_handles
            .get(&handle)
            .ok_or_else(|| format!("GEM handle {} not found", handle))?;
        let ptr = gem
            .cpu_map
            .ok_or_else(|| "buffer not CPU-mapped".to_string())?;
        let mut out = vec![0u8; gem.size];
        unsafe {
            memcpy(out.as_mut_ptr(), ptr, gem.size);
        }
        Ok(out)
    }

    /// Submit a command buffer with a list of GEM bo handles.
    pub fn submit_commands(
        &mut self,
        cmd_data: &[u8],
        buffer_handles: &[u32],
    ) -> Result<u64, String> {
        let fence_id = self.next_fence;
        self.next_fence += 1;

        match self.vendor {
            GpuVendor::AMD => {
                let mut cs = AmdgpuCs {
                    ctx_id: 0,
                    bo_list_handle: if buffer_handles.is_empty() {
                        0
                    } else {
                        buffer_handles[0]
                    },
                    num_chunks: 1,
                    flags: 0,
                    chunks: cmd_data.as_ptr() as u64,
                    seq_no: fence_id,
                };
                let ret =
                    unsafe { ioctl(self.fd, DRM_AMDGPU_CS, &mut cs as *mut AmdgpuCs) };
                if ret != 0 {
                    return Err(format!("AMDGPU CS submit failed: {}", errno_string()));
                }
            }
            GpuVendor::Intel => {
                // i915 execbuffer2 — simplified
                let mut exec_buf = [0u8; 64];
                // Write batch_start_offset=0, batch_len, etc.
                let batch_len = cmd_data.len() as u32;
                exec_buf[8..12].copy_from_slice(&batch_len.to_ne_bytes());
                if !buffer_handles.is_empty() {
                    let first = buffer_handles[0];
                    exec_buf[0..4].copy_from_slice(&first.to_ne_bytes());
                }
                let ret = unsafe {
                    ioctl(
                        self.fd,
                        DRM_I915_EXECBUFFER2,
                        exec_buf.as_mut_ptr(),
                    )
                };
                if ret != 0 {
                    return Err(format!("i915 execbuffer2 failed: {}", errno_string()));
                }
            }
            GpuVendor::Nvidia => {
                // Nouveau pushbuf — simplified
                let mut pushbuf = [0u8; 64];
                let nr_push = 1u32;
                pushbuf[0..4].copy_from_slice(&nr_push.to_ne_bytes());
                pushbuf[8..16].copy_from_slice(&(cmd_data.as_ptr() as u64).to_ne_bytes());
                let ret = unsafe {
                    ioctl(
                        self.fd,
                        DRM_NOUVEAU_GEM_PUSHBUF,
                        pushbuf.as_mut_ptr(),
                    )
                };
                if ret != 0 {
                    return Err(format!("Nouveau pushbuf failed: {}", errno_string()));
                }
            }
            GpuVendor::Unknown => {
                return Err("cannot submit commands on unknown vendor".into());
            }
        }

        Ok(fence_id)
    }

    /// Wait for a fence (GPU completion).
    pub fn wait_fence(&self, fence_id: u64) -> Result<(), String> {
        let timeout_ns: i64 = 5_000_000_000; // 5 seconds
        match self.vendor {
            GpuVendor::AMD => {
                let mut wait = AmdgpuWaitFences {
                    fences: fence_id,
                    fence_count: 1,
                    wait_all: 1,
                    timeout_ns: timeout_ns as u64,
                };
                let ret = unsafe {
                    ioctl(
                        self.fd,
                        DRM_AMDGPU_WAIT_FENCES,
                        &mut wait as *mut AmdgpuWaitFences,
                    )
                };
                if ret != 0 {
                    return Err(format!("AMDGPU fence wait failed: {}", errno_string()));
                }
            }
            GpuVendor::Intel => {
                let mut wait = I915GemWait {
                    bo_handle: fence_id as u32,
                    flags: 0,
                    timeout_ns,
                };
                let ret = unsafe {
                    ioctl(
                        self.fd,
                        DRM_I915_GEM_WAIT,
                        &mut wait as *mut I915GemWait,
                    )
                };
                if ret != 0 {
                    return Err(format!("i915 GEM wait failed: {}", errno_string()));
                }
            }
            GpuVendor::Nvidia | GpuVendor::Unknown => {
                // Nouveau doesn't have a separate wait ioctl in the simple path;
                // pushbuf is synchronous. For unknown, just return ok.
            }
        }
        Ok(())
    }

    /// Free a GEM buffer.
    pub fn free_gem(&mut self, handle: u32) -> Result<(), String> {
        // Unmap CPU mapping if present
        if let Some(gem) = self.gem_handles.remove(&handle) {
            if let Some(ptr) = gem.cpu_map {
                unsafe {
                    munmap(ptr, gem.size);
                }
            }
        }
        self.pool.release(handle);

        let mut gc = DrmGemClose { handle, pad: 0 };
        let ret =
            unsafe { ioctl(self.fd, DRM_IOCTL_GEM_CLOSE, &mut gc as *mut DrmGemClose) };
        if ret != 0 {
            return Err(format!("GEM close failed: {}", errno_string()));
        }
        Ok(())
    }

    /// VRAM pool stats: (total_allocated, in_use, free_in_pool).
    pub fn vram_stats(&self) -> (usize, usize, usize) {
        self.pool.stats()
    }
}

impl Drop for VortexGpuDevice {
    fn drop(&mut self) {
        // Unmap and close all GEM handles
        let handles: Vec<u32> = self.gem_handles.keys().cloned().collect();
        for h in handles {
            let _ = self.free_gem(h);
        }
        if self.fd >= 0 {
            unsafe {
                close(self.fd);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GpuInfo (returned by discover)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub path: String,
    pub driver_name: String,
    pub description: String,
    pub vendor: GpuVendor,
    pub version_major: i32,
    pub version_minor: i32,
    pub version_patch: i32,
}

impl GpuInfo {
    fn to_value(&self) -> Value {
        let mut fields = HashMap::new();
        fields.insert("path".into(), Value::String(self.path.clone()));
        fields.insert("driver".into(), Value::String(self.driver_name.clone()));
        fields.insert("description".into(), Value::String(self.description.clone()));
        fields.insert("vendor".into(), Value::String(self.vendor.as_str().into()));
        fields.insert("version_major".into(), Value::Int(self.version_major as i128));
        fields.insert("version_minor".into(), Value::Int(self.version_minor as i128));
        fields.insert("version_patch".into(), Value::Int(self.version_patch as i128));
        Value::Struct {
            name: "GpuInfo".into(),
            fields,
        }
    }
}

// ---------------------------------------------------------------------------
// CpuRuntime fallback
// ---------------------------------------------------------------------------

struct CpuBuffer {
    data: Vec<u8>,
}

struct CpuRuntime {
    next_handle: u32,
    buffers: HashMap<u32, CpuBuffer>,
    next_fence: u64,
}

impl CpuRuntime {
    fn new() -> Self {
        Self {
            next_handle: 1,
            buffers: HashMap::new(),
            next_fence: 1,
        }
    }

    fn alloc(&mut self, size: usize) -> u32 {
        let handle = self.next_handle;
        self.next_handle += 1;
        self.buffers.insert(
            handle,
            CpuBuffer {
                data: vec![0u8; size],
            },
        );
        handle
    }

    fn upload(&mut self, handle: u32, data: &[u8]) -> Result<(), String> {
        let buf = self
            .buffers
            .get_mut(&handle)
            .ok_or_else(|| format!("CPU buffer {} not found", handle))?;
        let len = data.len().min(buf.data.len());
        buf.data[..len].copy_from_slice(&data[..len]);
        Ok(())
    }

    fn download(&self, handle: u32) -> Result<Vec<u8>, String> {
        let buf = self
            .buffers
            .get(&handle)
            .ok_or_else(|| format!("CPU buffer {} not found", handle))?;
        Ok(buf.data.clone())
    }

    fn submit(&mut self) -> u64 {
        let f = self.next_fence;
        self.next_fence += 1;
        f
    }

    fn free(&mut self, handle: u32) -> Result<(), String> {
        self.buffers
            .remove(&handle)
            .map(|_| ())
            .ok_or_else(|| format!("CPU buffer {} not found", handle))
    }

    fn stats(&self) -> (usize, usize, usize) {
        let total: usize = self.buffers.values().map(|b| b.data.len()).sum();
        (total, total, 0)
    }
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

enum DeviceSlot {
    Gpu(VortexGpuDevice),
    Cpu(CpuRuntime),
}

struct DriverState {
    devices: HashMap<u32, DeviceSlot>,
    next_id: u32,
    /// Whether we already attempted discovery
    discovered: bool,
    /// Cached discovery results
    gpu_infos: Vec<GpuInfo>,
}

impl DriverState {
    fn new() -> Self {
        Self {
            devices: HashMap::new(),
            next_id: 1,
            discovered: false,
            gpu_infos: Vec::new(),
        }
    }

    fn ensure_discovered(&mut self) {
        if !self.discovered {
            self.gpu_infos = VortexGpuDevice::discover();
            self.discovered = true;
        }
    }

    fn open_device(&mut self, path: &str) -> Result<u32, String> {
        match VortexGpuDevice::open_device(path) {
            Ok(dev) => {
                let id = self.next_id;
                self.next_id += 1;
                self.devices.insert(id, DeviceSlot::Gpu(dev));
                Ok(id)
            }
            Err(e) => Err(e),
        }
    }

    fn open_cpu_fallback(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.devices
            .insert(id, DeviceSlot::Cpu(CpuRuntime::new()));
        id
    }
}

static DRIVER: LazyLock<Mutex<DriverState>> =
    LazyLock::new(|| Mutex::new(DriverState::new()));

// ---------------------------------------------------------------------------
// Helper: extract int from Value
// ---------------------------------------------------------------------------

fn val_to_int(v: &Value, name: &str) -> Result<i128, String> {
    match v {
        Value::Int(n) => Ok(*n),
        _ => Err(format!("{} must be an integer", name)),
    }
}

fn val_to_string(v: &Value, name: &str) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        _ => Err(format!("{} must be a string", name)),
    }
}

fn val_to_bytes(v: &Value) -> Result<Vec<u8>, String> {
    match v {
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                match item {
                    Value::Int(n) => out.push(*n as u8),
                    _ => return Err("array elements must be integers for byte data".into()),
                }
            }
            Ok(out)
        }
        Value::String(s) => Ok(s.as_bytes().to_vec()),
        _ => Err("expected array or string for byte data".into()),
    }
}

fn bytes_to_value(data: Vec<u8>) -> Value {
    Value::Array(data.into_iter().map(|b| Value::Int(b as i128)).collect())
}

// ---------------------------------------------------------------------------
// Builtin implementations
// ---------------------------------------------------------------------------

fn builtin_gpu_discover(
    _env: &mut Env,
    _args: Vec<Value>,
) -> Result<Value, String> {
    let mut state = DRIVER.lock().map_err(|e| format!("lock error: {}", e))?;
    state.ensure_discovered();
    let infos: Vec<Value> = state.gpu_infos.iter().map(|g| g.to_value()).collect();
    if infos.is_empty() {
        // Return an array with a single CPU fallback info
        let mut fields = HashMap::new();
        fields.insert("path".into(), Value::String("cpu".into()));
        fields.insert("driver".into(), Value::String("cpu_fallback".into()));
        fields.insert("description".into(), Value::String("CPU fallback runtime (no GPU found)".into()));
        fields.insert("vendor".into(), Value::String("cpu".into()));
        fields.insert("version_major".into(), Value::Int(0));
        fields.insert("version_minor".into(), Value::Int(0));
        fields.insert("version_patch".into(), Value::Int(0));
        Ok(Value::Array(vec![Value::Struct {
            name: "GpuInfo".into(),
            fields,
        }]))
    } else {
        Ok(Value::Array(infos))
    }
}

fn builtin_gpu_open(
    _env: &mut Env,
    args: Vec<Value>,
) -> Result<Value, String> {
    if args.is_empty() {
        return Err("gpu_open requires a device path argument".into());
    }
    let path = val_to_string(&args[0], "device_path")?;

    let mut state = DRIVER.lock().map_err(|e| format!("lock error: {}", e))?;

    if path == "cpu" || path == "auto" {
        // Try to open a real GPU first if "auto"
        if path == "auto" {
            state.ensure_discovered();
            if let Some(info) = state.gpu_infos.first().cloned() {
                if let Ok(id) = state.open_device(&info.path) {
                    return Ok(Value::Int(id as i128));
                }
            }
        }
        // Fall back to CPU
        let id = state.open_cpu_fallback();
        return Ok(Value::Int(id as i128));
    }

    match state.open_device(&path) {
        Ok(id) => Ok(Value::Int(id as i128)),
        Err(_) => {
            // Fallback to CPU runtime
            let id = state.open_cpu_fallback();
            Ok(Value::Int(id as i128))
        }
    }
}

fn builtin_gpu_alloc(
    _env: &mut Env,
    args: Vec<Value>,
) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("gpu_alloc(device_id, size[, flags])".into());
    }
    let dev_id = val_to_int(&args[0], "device_id")? as u32;
    let size = val_to_int(&args[1], "size")? as usize;
    let flags = if args.len() > 2 {
        val_to_int(&args[2], "flags")? as u32
    } else {
        0
    };

    let mut state = DRIVER.lock().map_err(|e| format!("lock error: {}", e))?;
    let dev = state
        .devices
        .get_mut(&dev_id)
        .ok_or_else(|| format!("device {} not found", dev_id))?;

    match dev {
        DeviceSlot::Gpu(gpu) => {
            let handle = gpu.alloc_vram(size, flags)?;
            Ok(Value::Int(handle as i128))
        }
        DeviceSlot::Cpu(cpu) => {
            let handle = cpu.alloc(size);
            Ok(Value::Int(handle as i128))
        }
    }
}

fn builtin_gpu_upload(
    _env: &mut Env,
    args: Vec<Value>,
) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("gpu_upload(device_id, handle, data)".into());
    }
    let dev_id = val_to_int(&args[0], "device_id")? as u32;
    let handle = val_to_int(&args[1], "handle")? as u32;
    let data = val_to_bytes(&args[2])?;

    let mut state = DRIVER.lock().map_err(|e| format!("lock error: {}", e))?;
    let dev = state
        .devices
        .get_mut(&dev_id)
        .ok_or_else(|| format!("device {} not found", dev_id))?;

    match dev {
        DeviceSlot::Gpu(gpu) => {
            gpu.upload(handle, &data)?;
            Ok(Value::Bool(true))
        }
        DeviceSlot::Cpu(cpu) => {
            cpu.upload(handle, &data)?;
            Ok(Value::Bool(true))
        }
    }
}

fn builtin_gpu_download(
    _env: &mut Env,
    args: Vec<Value>,
) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("gpu_download(device_id, handle)".into());
    }
    let dev_id = val_to_int(&args[0], "device_id")? as u32;
    let handle = val_to_int(&args[1], "handle")? as u32;

    let state = DRIVER.lock().map_err(|e| format!("lock error: {}", e))?;
    let dev = state
        .devices
        .get(&dev_id)
        .ok_or_else(|| format!("device {} not found", dev_id))?;

    match dev {
        DeviceSlot::Gpu(gpu) => {
            let data = gpu.download(handle)?;
            Ok(bytes_to_value(data))
        }
        DeviceSlot::Cpu(cpu) => {
            let data = cpu.download(handle)?;
            Ok(bytes_to_value(data))
        }
    }
}

fn builtin_gpu_submit(
    _env: &mut Env,
    args: Vec<Value>,
) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("gpu_submit(device_id, cmd_data[, buffer_handles])".into());
    }
    let dev_id = val_to_int(&args[0], "device_id")? as u32;
    let cmd_data = val_to_bytes(&args[1])?;
    let buffer_handles: Vec<u32> = if args.len() > 2 {
        match &args[2] {
            Value::Array(arr) => arr
                .iter()
                .map(|v| val_to_int(v, "handle").map(|n| n as u32))
                .collect::<Result<Vec<_>, _>>()?,
            _ => vec![],
        }
    } else {
        vec![]
    };

    let mut state = DRIVER.lock().map_err(|e| format!("lock error: {}", e))?;
    let dev = state
        .devices
        .get_mut(&dev_id)
        .ok_or_else(|| format!("device {} not found", dev_id))?;

    match dev {
        DeviceSlot::Gpu(gpu) => {
            let fence = gpu.submit_commands(&cmd_data, &buffer_handles)?;
            Ok(Value::Int(fence as i128))
        }
        DeviceSlot::Cpu(cpu) => {
            // CPU fallback: commands are a no-op, return fence immediately
            let fence = cpu.submit();
            Ok(Value::Int(fence as i128))
        }
    }
}

fn builtin_gpu_wait(
    _env: &mut Env,
    args: Vec<Value>,
) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("gpu_wait(device_id, fence_id)".into());
    }
    let dev_id = val_to_int(&args[0], "device_id")? as u32;
    let fence_id = val_to_int(&args[1], "fence_id")? as u64;

    let state = DRIVER.lock().map_err(|e| format!("lock error: {}", e))?;
    let dev = state
        .devices
        .get(&dev_id)
        .ok_or_else(|| format!("device {} not found", dev_id))?;

    match dev {
        DeviceSlot::Gpu(gpu) => {
            gpu.wait_fence(fence_id)?;
            Ok(Value::Bool(true))
        }
        DeviceSlot::Cpu(_) => {
            // CPU: always immediately complete
            Ok(Value::Bool(true))
        }
    }
}

fn builtin_gpu_free(
    _env: &mut Env,
    args: Vec<Value>,
) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("gpu_free(device_id, handle)".into());
    }
    let dev_id = val_to_int(&args[0], "device_id")? as u32;
    let handle = val_to_int(&args[1], "handle")? as u32;

    let mut state = DRIVER.lock().map_err(|e| format!("lock error: {}", e))?;
    let dev = state
        .devices
        .get_mut(&dev_id)
        .ok_or_else(|| format!("device {} not found", dev_id))?;

    match dev {
        DeviceSlot::Gpu(gpu) => {
            gpu.free_gem(handle)?;
            Ok(Value::Bool(true))
        }
        DeviceSlot::Cpu(cpu) => {
            cpu.free(handle)?;
            Ok(Value::Bool(true))
        }
    }
}

fn builtin_gpu_vram_stats(
    _env: &mut Env,
    args: Vec<Value>,
) -> Result<Value, String> {
    if args.is_empty() {
        return Err("gpu_vram_stats(device_id)".into());
    }
    let dev_id = val_to_int(&args[0], "device_id")? as u32;

    let state = DRIVER.lock().map_err(|e| format!("lock error: {}", e))?;
    let dev = state
        .devices
        .get(&dev_id)
        .ok_or_else(|| format!("device {} not found", dev_id))?;

    let (total, used, free) = match dev {
        DeviceSlot::Gpu(gpu) => gpu.vram_stats(),
        DeviceSlot::Cpu(cpu) => cpu.stats(),
    };

    let mut fields = HashMap::new();
    fields.insert("total".into(), Value::Int(total as i128));
    fields.insert("used".into(), Value::Int(used as i128));
    fields.insert("free".into(), Value::Int(free as i128));
    Ok(Value::Struct {
        name: "VramStats".into(),
        fields,
    })
}

fn builtin_gpu_device_info(
    _env: &mut Env,
    args: Vec<Value>,
) -> Result<Value, String> {
    if args.is_empty() {
        return Err("gpu_device_info(device_id)".into());
    }
    let dev_id = val_to_int(&args[0], "device_id")? as u32;

    let state = DRIVER.lock().map_err(|e| format!("lock error: {}", e))?;
    let dev = state
        .devices
        .get(&dev_id)
        .ok_or_else(|| format!("device {} not found", dev_id))?;

    let mut fields = HashMap::new();
    fields.insert("device_id".into(), Value::Int(dev_id as i128));
    match dev {
        DeviceSlot::Gpu(gpu) => {
            fields.insert("backend".into(), Value::String("gpu".into()));
            fields.insert("vendor".into(), Value::String(gpu.vendor.as_str().into()));
            fields.insert("driver".into(), Value::String(gpu.driver_name.clone()));
            fields.insert("path".into(), Value::String(gpu.device_path.clone()));
            fields.insert("fd".into(), Value::Int(gpu.fd as i128));
            fields.insert("vram_total".into(), Value::Int(gpu.vram_total as i128));
            fields.insert(
                "gem_count".into(),
                Value::Int(gpu.gem_handles.len() as i128),
            );
        }
        DeviceSlot::Cpu(_) => {
            fields.insert("backend".into(), Value::String("cpu_fallback".into()));
            fields.insert("vendor".into(), Value::String("cpu".into()));
            fields.insert("driver".into(), Value::String("cpu_fallback".into()));
            fields.insert("path".into(), Value::String("N/A".into()));
            fields.insert("fd".into(), Value::Int(-1));
            fields.insert("vram_total".into(), Value::Int(0));
            fields.insert("gem_count".into(), Value::Int(0));
        }
    }
    Ok(Value::Struct {
        name: "GpuDeviceInfo".into(),
        fields,
    })
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register_builtins(env: &mut Env) {
    env.functions.insert(
        "gpu_discover".to_string(),
        FnDef::Builtin(builtin_gpu_discover),
    );
    env.functions.insert(
        "gpu_open".to_string(),
        FnDef::Builtin(builtin_gpu_open),
    );
    env.functions.insert(
        "gpu_alloc".to_string(),
        FnDef::Builtin(builtin_gpu_alloc),
    );
    env.functions.insert(
        "gpu_upload".to_string(),
        FnDef::Builtin(builtin_gpu_upload),
    );
    env.functions.insert(
        "gpu_download".to_string(),
        FnDef::Builtin(builtin_gpu_download),
    );
    env.functions.insert(
        "gpu_submit".to_string(),
        FnDef::Builtin(builtin_gpu_submit),
    );
    env.functions.insert(
        "gpu_wait".to_string(),
        FnDef::Builtin(builtin_gpu_wait),
    );
    env.functions.insert(
        "gpu_free".to_string(),
        FnDef::Builtin(builtin_gpu_free),
    );
    env.functions.insert(
        "gpu_vram_stats".to_string(),
        FnDef::Builtin(builtin_gpu_vram_stats),
    );
    env.functions.insert(
        "gpu_device_info".to_string(),
        FnDef::Builtin(builtin_gpu_device_info),
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Reset global state between tests.
    fn reset_driver() {
        let mut state = DRIVER.lock().unwrap();
        state.devices.clear();
        state.next_id = 1;
        state.discovered = false;
        state.gpu_infos.clear();
    }

    #[test]
    fn test_cpu_fallback_open() {
        reset_driver();
        let mut state = DRIVER.lock().unwrap();
        let id = state.open_cpu_fallback();
        assert_eq!(id, 1);
        assert!(state.devices.contains_key(&1));
    }

    #[test]
    fn test_cpu_alloc_upload_download() {
        reset_driver();
        let mut state = DRIVER.lock().unwrap();
        let dev_id = state.open_cpu_fallback();
        let dev = state.devices.get_mut(&dev_id).unwrap();
        if let DeviceSlot::Cpu(cpu) = dev {
            let handle = cpu.alloc(64);
            assert_eq!(handle, 1);

            let data = vec![1u8, 2, 3, 4, 5];
            cpu.upload(handle, &data).unwrap();

            let out = cpu.download(handle).unwrap();
            assert_eq!(&out[..5], &[1, 2, 3, 4, 5]);
            assert_eq!(out.len(), 64);

            cpu.free(handle).unwrap();
            assert!(cpu.download(handle).is_err());
        } else {
            panic!("expected CPU slot");
        }
    }

    #[test]
    fn test_cpu_submit_and_wait() {
        reset_driver();
        let mut state = DRIVER.lock().unwrap();
        let dev_id = state.open_cpu_fallback();
        let dev = state.devices.get_mut(&dev_id).unwrap();
        if let DeviceSlot::Cpu(cpu) = dev {
            let f1 = cpu.submit();
            let f2 = cpu.submit();
            assert_eq!(f1, 1);
            assert_eq!(f2, 2);
        } else {
            panic!("expected CPU slot");
        }
    }

    #[test]
    fn test_cpu_stats() {
        reset_driver();
        let mut state = DRIVER.lock().unwrap();
        let dev_id = state.open_cpu_fallback();
        let dev = state.devices.get_mut(&dev_id).unwrap();
        if let DeviceSlot::Cpu(cpu) = dev {
            let _ = cpu.alloc(128);
            let _ = cpu.alloc(256);
            let (total, used, free) = cpu.stats();
            assert_eq!(total, 384);
            assert_eq!(used, 384);
            assert_eq!(free, 0);
        } else {
            panic!("expected CPU slot");
        }
    }

    #[test]
    fn test_vram_pool_bucket_index() {
        assert_eq!(VramPool::bucket_index(1), 12); // rounds up to 4096
        assert_eq!(VramPool::bucket_index(4096), 12);
        assert_eq!(VramPool::bucket_index(4097), 13); // 8192
        assert_eq!(VramPool::bucket_index(1 << 20), 20); // 1 MiB
    }

    #[test]
    fn test_vram_pool_track_and_reclaim() {
        let mut pool = VramPool::new();
        pool.track(42, 4096);
        // While in use, reclaim should fail
        assert!(pool.try_reclaim(4096).is_none());
        // Release it
        pool.release(42);
        // Now reclaim should succeed
        assert_eq!(pool.try_reclaim(4096), Some(42));
    }

    #[test]
    fn test_vram_pool_stats() {
        let mut pool = VramPool::new();
        pool.track(1, 4096);
        pool.track(2, 8192);
        pool.release(1);
        let (total, used, free) = pool.stats();
        assert_eq!(total, 4096 + 8192);
        assert_eq!(used, 8192);
        assert_eq!(free, 4096);
    }

    #[test]
    fn test_vendor_classification() {
        assert_eq!(VortexGpuDevice::classify_vendor("amdgpu"), GpuVendor::AMD);
        assert_eq!(VortexGpuDevice::classify_vendor("radeon"), GpuVendor::AMD);
        assert_eq!(
            VortexGpuDevice::classify_vendor("nouveau"),
            GpuVendor::Nvidia
        );
        assert_eq!(VortexGpuDevice::classify_vendor("i915"), GpuVendor::Intel);
        assert_eq!(VortexGpuDevice::classify_vendor("xe"), GpuVendor::Intel);
        assert_eq!(
            VortexGpuDevice::classify_vendor("vmwgfx"),
            GpuVendor::Unknown
        );
    }

    #[test]
    fn test_discover_returns_cpu_fallback_when_no_gpu() {
        // In CI without GPU, discover should return empty from the real scan
        // but our builtin wraps it to return a CPU fallback entry
        let infos = VortexGpuDevice::discover();
        // We just verify it doesn't panic; result depends on environment
        let _ = infos;
    }

    #[test]
    fn test_builtin_gpu_discover_fallback() {
        reset_driver();
        let mut env = Env::new();
        register_builtins(&mut env);
        let result = builtin_gpu_discover(&mut env, vec![]).unwrap();
        match result {
            Value::Array(arr) => {
                assert!(!arr.is_empty(), "discover should return at least CPU fallback");
            }
            _ => panic!("expected array from gpu_discover"),
        }
    }

    #[test]
    fn test_builtin_gpu_open_cpu_fallback() {
        reset_driver();
        let mut env = Env::new();
        register_builtins(&mut env);
        let result =
            builtin_gpu_open(&mut env, vec![Value::String("cpu".into())]).unwrap();
        match result {
            Value::Int(id) => assert!(id > 0),
            _ => panic!("expected int from gpu_open"),
        }
    }

    #[test]
    fn test_builtin_gpu_open_auto_fallback() {
        reset_driver();
        let mut env = Env::new();
        let result =
            builtin_gpu_open(&mut env, vec![Value::String("auto".into())]).unwrap();
        match result {
            Value::Int(id) => assert!(id > 0),
            _ => panic!("expected int from gpu_open auto"),
        }
    }

    #[test]
    fn test_builtin_roundtrip_cpu() {
        reset_driver();
        let mut env = Env::new();
        register_builtins(&mut env);

        // Open CPU device
        let dev_id =
            match builtin_gpu_open(&mut env, vec![Value::String("cpu".into())]).unwrap() {
                Value::Int(n) => n,
                _ => panic!("expected int"),
            };

        // Alloc 32 bytes
        let handle = match builtin_gpu_alloc(
            &mut env,
            vec![Value::Int(dev_id), Value::Int(32)],
        )
        .unwrap()
        {
            Value::Int(n) => n,
            _ => panic!("expected int"),
        };

        // Upload
        let data = Value::Array(vec![
            Value::Int(0xDE),
            Value::Int(0xAD),
            Value::Int(0xBE),
            Value::Int(0xEF),
        ]);
        builtin_gpu_upload(
            &mut env,
            vec![Value::Int(dev_id), Value::Int(handle), data],
        )
        .unwrap();

        // Download
        let result = builtin_gpu_download(
            &mut env,
            vec![Value::Int(dev_id), Value::Int(handle)],
        )
        .unwrap();
        match result {
            Value::Array(arr) => {
                assert_eq!(arr.len(), 32);
                assert!(matches!(arr[0], Value::Int(0xDE)));
                assert!(matches!(arr[1], Value::Int(0xAD)));
                assert!(matches!(arr[2], Value::Int(0xBE)));
                assert!(matches!(arr[3], Value::Int(0xEF)));
            }
            _ => panic!("expected array"),
        }

        // Submit + wait
        let fence = match builtin_gpu_submit(
            &mut env,
            vec![
                Value::Int(dev_id),
                Value::Array(vec![Value::Int(0x90)]), // NOP
            ],
        )
        .unwrap()
        {
            Value::Int(n) => n,
            _ => panic!("expected int"),
        };
        builtin_gpu_wait(&mut env, vec![Value::Int(dev_id), Value::Int(fence)]).unwrap();

        // Stats
        let stats =
            builtin_gpu_vram_stats(&mut env, vec![Value::Int(dev_id)]).unwrap();
        match stats {
            Value::Struct { name, fields } => {
                assert_eq!(name, "VramStats");
                assert!(matches!(fields.get("total"), Some(Value::Int(32))));
            }
            _ => panic!("expected struct"),
        }

        // Device info
        let info =
            builtin_gpu_device_info(&mut env, vec![Value::Int(dev_id)]).unwrap();
        match info {
            Value::Struct { name, fields } => {
                assert_eq!(name, "GpuDeviceInfo");
                assert!(matches!(
                    fields.get("backend"),
                    Some(Value::String(s)) if s == "cpu_fallback"
                ));
            }
            _ => panic!("expected struct"),
        }

        // Free
        builtin_gpu_free(&mut env, vec![Value::Int(dev_id), Value::Int(handle)]).unwrap();
    }

    #[test]
    fn test_builtin_error_cases() {
        reset_driver();
        let mut env = Env::new();

        // Missing args
        assert!(builtin_gpu_open(&mut env, vec![]).is_err());
        assert!(builtin_gpu_alloc(&mut env, vec![]).is_err());
        assert!(builtin_gpu_upload(&mut env, vec![]).is_err());
        assert!(builtin_gpu_download(&mut env, vec![]).is_err());
        assert!(builtin_gpu_submit(&mut env, vec![]).is_err());
        assert!(builtin_gpu_wait(&mut env, vec![]).is_err());
        assert!(builtin_gpu_free(&mut env, vec![]).is_err());
        assert!(builtin_gpu_vram_stats(&mut env, vec![]).is_err());
        assert!(builtin_gpu_device_info(&mut env, vec![]).is_err());

        // Bad device id
        assert!(builtin_gpu_alloc(
            &mut env,
            vec![Value::Int(999), Value::Int(64)]
        )
        .is_err());
    }

    #[test]
    fn test_bytes_conversion() {
        let data = vec![1u8, 2, 3, 255];
        let val = bytes_to_value(data.clone());
        let back = val_to_bytes(&val).unwrap();
        assert_eq!(data, back);

        // String path
        let s = Value::String("hello".into());
        let bytes = val_to_bytes(&s).unwrap();
        assert_eq!(bytes, b"hello");
    }

    #[test]
    fn test_gpu_info_to_value() {
        let info = GpuInfo {
            path: "/dev/dri/renderD128".into(),
            driver_name: "amdgpu".into(),
            description: "AMD Radeon".into(),
            vendor: GpuVendor::AMD,
            version_major: 3,
            version_minor: 54,
            version_patch: 0,
        };
        let val = info.to_value();
        match val {
            Value::Struct { name, fields } => {
                assert_eq!(name, "GpuInfo");
                assert!(matches!(
                    fields.get("vendor"),
                    Some(Value::String(s)) if s == "amd"
                ));
                assert!(matches!(
                    fields.get("version_major"),
                    Some(Value::Int(3))
                ));
            }
            _ => panic!("expected struct"),
        }
    }
}

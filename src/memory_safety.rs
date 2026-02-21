use std::collections::{HashMap, HashSet};

/// Memory regions for tracking buffer lifetimes
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Region {
    Stack(usize),
    Heap(usize),
    Device(usize),
    Static,
}

/// Access mode
#[derive(Clone, Debug, PartialEq)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
    Exclusive,
}

/// A tracked reference to a memory region
#[derive(Clone, Debug)]
pub struct MemRef {
    pub region: Region,
    pub mode: AccessMode,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub alias_set: usize,
    pub region_id: usize,
}

/// Memory safety checker
pub struct MemorySafetyChecker {
    regions: HashMap<usize, RegionInfo>,
    active_refs: Vec<Option<MemRef>>,
    alias_sets: HashMap<usize, Vec<usize>>,
    errors: Vec<MemoryError>,
    next_region: usize,
    next_alias: usize,
    next_ref: usize,
}

#[derive(Clone, Debug)]
pub struct RegionInfo {
    pub id: usize,
    pub kind: Region,
    pub size: usize,
    pub is_alive: bool,
    pub initialized: bool,
    pub created_at: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MemoryError {
    UseAfterFree { region: usize, location: String },
    DoubleFree { region: usize, location: String },
    DataRace { region: usize, writer: String, reader: String },
    BufferOverflow { region: usize, index: usize, size: usize, location: String },
    UninitializedRead { region: usize, location: String },
    LeakedBuffer { region: usize, created_at: String },
    HostDeviceMismatch { expected: String, actual: String, location: String },
}

impl MemorySafetyChecker {
    pub fn new() -> Self {
        Self {
            regions: HashMap::new(),
            active_refs: Vec::new(),
            alias_sets: HashMap::new(),
            errors: Vec::new(),
            next_region: 0,
            next_alias: 0,
            next_ref: 0,
        }
    }

    /// Allocate a new region, returns region id
    pub fn alloc(&mut self, kind: Region, size: usize, location: &str) -> usize {
        let id = self.next_region;
        self.next_region += 1;
        self.regions.insert(id, RegionInfo {
            id,
            kind,
            size,
            is_alive: true,
            initialized: false,
            created_at: location.to_string(),
        });
        id
    }

    /// Free a region
    pub fn free(&mut self, region_id: usize, location: &str) -> Result<(), MemoryError> {
        let region = self.regions.get_mut(&region_id);
        match region {
            Some(r) if !r.is_alive => {
                let err = MemoryError::DoubleFree { region: region_id, location: location.to_string() };
                self.errors.push(err.clone());
                Err(err)
            }
            Some(r) => {
                r.is_alive = false;
                Ok(())
            }
            None => {
                let err = MemoryError::UseAfterFree { region: region_id, location: location.to_string() };
                self.errors.push(err.clone());
                Err(err)
            }
        }
    }

    /// Create a reference to a region, returns ref id
    pub fn borrow(&mut self, region_id: usize, mode: AccessMode, location: &str) -> Result<usize, MemoryError> {
        // Check region is alive
        let region = self.regions.get(&region_id);
        match region {
            Some(r) if !r.is_alive => {
                let err = MemoryError::UseAfterFree { region: region_id, location: location.to_string() };
                self.errors.push(err.clone());
                return Err(err);
            }
            None => {
                let err = MemoryError::UseAfterFree { region: region_id, location: location.to_string() };
                self.errors.push(err.clone());
                return Err(err);
            }
            _ => {}
        }

        // Check for conflicts with existing borrows
        self.check_race(region_id, mode.clone(), location)?;

        let region_info = self.regions.get(&region_id).unwrap();
        let alias = self.next_alias;
        self.next_alias += 1;

        let ref_id = self.next_ref;
        self.next_ref += 1;

        let memref = MemRef {
            region: region_info.kind.clone(),
            mode,
            dtype: String::new(),
            shape: Vec::new(),
            alias_set: alias,
            region_id,
        };

        // Extend active_refs if needed
        while self.active_refs.len() <= ref_id {
            self.active_refs.push(None);
        }
        self.active_refs[ref_id] = Some(memref);
        self.alias_sets.entry(alias).or_default().push(ref_id);

        Ok(ref_id)
    }

    /// Release a reference
    pub fn release(&mut self, ref_id: usize) {
        if ref_id < self.active_refs.len() {
            self.active_refs[ref_id] = None;
        }
    }

    /// Check if an access is safe
    pub fn check_access(&self, region_id: usize, index: usize, _mode: AccessMode, location: &str) -> Result<(), MemoryError> {
        match self.regions.get(&region_id) {
            Some(r) if !r.is_alive => {
                Err(MemoryError::UseAfterFree { region: region_id, location: location.to_string() })
            }
            Some(r) => {
                if index >= r.size {
                    Err(MemoryError::BufferOverflow { region: region_id, index, size: r.size, location: location.to_string() })
                } else {
                    Ok(())
                }
            }
            None => {
                Err(MemoryError::UseAfterFree { region: region_id, location: location.to_string() })
            }
        }
    }

    /// Check for data races
    pub fn check_race(&self, region_id: usize, mode: AccessMode, location: &str) -> Result<(), MemoryError> {
        let existing: Vec<&MemRef> = self.active_refs.iter()
            .filter_map(|r| r.as_ref())
            .filter(|r| r.region_id == region_id)
            .collect();

        for existing_ref in &existing {
            // Exclusive blocks everything
            if existing_ref.mode == AccessMode::Exclusive {
                let err = MemoryError::DataRace {
                    region: region_id,
                    writer: format!("exclusive borrow"),
                    reader: location.to_string(),
                };
                return Err(err);
            }
            // New exclusive blocked by anything
            if mode == AccessMode::Exclusive {
                let err = MemoryError::DataRace {
                    region: region_id,
                    writer: location.to_string(),
                    reader: format!("existing borrow"),
                };
                return Err(err);
            }
            // Write conflicts with any existing
            if mode == AccessMode::Write || mode == AccessMode::ReadWrite {
                let err = MemoryError::DataRace {
                    region: region_id,
                    writer: location.to_string(),
                    reader: format!("existing borrow"),
                };
                return Err(err);
            }
            // Read conflicts with existing write
            if (existing_ref.mode == AccessMode::Write || existing_ref.mode == AccessMode::ReadWrite)
                && mode == AccessMode::Read
            {
                let err = MemoryError::DataRace {
                    region: region_id,
                    writer: format!("existing write borrow"),
                    reader: location.to_string(),
                };
                return Err(err);
            }
        }
        Ok(())
    }

    /// Check host/device consistency
    pub fn check_device_access(&self, ref_id: usize, expected_device: &str, location: &str) -> Result<(), MemoryError> {
        if let Some(Some(memref)) = self.active_refs.get(ref_id) {
            let actual = match &memref.region {
                Region::Device(_) => "device",
                Region::Stack(_) | Region::Heap(_) => "host",
                Region::Static => "static",
            };
            if actual != expected_device {
                return Err(MemoryError::HostDeviceMismatch {
                    expected: expected_device.to_string(),
                    actual: actual.to_string(),
                    location: location.to_string(),
                });
            }
        }
        Ok(())
    }

    /// End of scope: check for leaked buffers
    pub fn end_scope(&mut self) -> Vec<MemoryError> {
        let mut leaks = Vec::new();
        for (_, region) in &self.regions {
            if region.is_alive {
                let err = MemoryError::LeakedBuffer {
                    region: region.id,
                    created_at: region.created_at.clone(),
                };
                leaks.push(err);
            }
        }
        self.errors.extend(leaks.clone());
        leaks
    }

    /// Report all accumulated errors
    pub fn report(&self) -> Vec<MemoryError> {
        self.errors.clone()
    }

    /// Pretty-print errors
    pub fn format_errors(&self) -> String {
        let mut out = String::new();
        for err in &self.errors {
            match err {
                MemoryError::UseAfterFree { region, location } =>
                    out.push_str(&format!("error: use after free of region {} at {}\n", region, location)),
                MemoryError::DoubleFree { region, location } =>
                    out.push_str(&format!("error: double free of region {} at {}\n", region, location)),
                MemoryError::DataRace { region, writer, reader } =>
                    out.push_str(&format!("error: data race on region {}: {} conflicts with {}\n", region, writer, reader)),
                MemoryError::BufferOverflow { region, index, size, location } =>
                    out.push_str(&format!("error: buffer overflow on region {} (index {} >= size {}) at {}\n", region, index, size, location)),
                MemoryError::UninitializedRead { region, location } =>
                    out.push_str(&format!("error: uninitialized read of region {} at {}\n", region, location)),
                MemoryError::LeakedBuffer { region, created_at } =>
                    out.push_str(&format!("error: leaked buffer region {} allocated at {}\n", region, created_at)),
                MemoryError::HostDeviceMismatch { expected, actual, location } =>
                    out.push_str(&format!("error: host/device mismatch at {}: expected {} got {}\n", location, expected, actual)),
            }
        }
        out
    }
}

/// GPU-specific memory tracker
pub struct GpuMemoryTracker {
    host_buffers: HashSet<usize>,
    device_buffers: HashSet<usize>,
    transfers_in_flight: Vec<Transfer>,
    device_ready: HashSet<usize>,
}

#[derive(Clone, Debug)]
pub struct Transfer {
    pub src_region: usize,
    pub dst_region: usize,
    pub direction: TransferDirection,
    pub size: usize,
    pub is_complete: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

impl GpuMemoryTracker {
    pub fn new() -> Self {
        Self {
            host_buffers: HashSet::new(),
            device_buffers: HashSet::new(),
            transfers_in_flight: Vec::new(),
            device_ready: HashSet::new(),
        }
    }

    pub fn alloc_host(&mut self, id: usize) {
        self.host_buffers.insert(id);
    }

    pub fn alloc_device(&mut self, id: usize) {
        self.device_buffers.insert(id);
    }

    pub fn transfer(&mut self, src: usize, dst: usize, dir: TransferDirection) {
        self.transfers_in_flight.push(Transfer {
            src_region: src,
            dst_region: dst,
            direction: dir,
            size: 0,
            is_complete: false,
        });
    }

    pub fn complete_transfer(&mut self, src: usize, dst: usize) {
        for t in &mut self.transfers_in_flight {
            if t.src_region == src && t.dst_region == dst && !t.is_complete {
                t.is_complete = true;
                self.device_ready.insert(dst);
                break;
            }
        }
    }

    /// Check: accessing device buffer that hasn't been transferred to yet
    pub fn check_device_ready(&self, id: usize) -> Result<(), MemoryError> {
        if self.device_buffers.contains(&id) && !self.device_ready.contains(&id) {
            return Err(MemoryError::UninitializedRead {
                region: id,
                location: format!("device buffer {} not yet transferred", id),
            });
        }
        Ok(())
    }

    /// Check: reading host buffer while transfer to device is in flight
    pub fn check_transfer_conflict(&self, id: usize) -> Result<(), MemoryError> {
        for t in &self.transfers_in_flight {
            if !t.is_complete && (t.src_region == id || t.dst_region == id) {
                return Err(MemoryError::DataRace {
                    region: id,
                    writer: format!("in-flight transfer"),
                    reader: format!("buffer access"),
                });
            }
        }
        Ok(())
    }
}

/// Analyze an AST for memory safety issues
pub fn analyze_memory_safety(_program: &[crate::ast::Item]) -> Vec<MemoryError> {
    // Placeholder: walk AST and check kernel functions for memory issues
    // Full implementation requires tracking allocations through the AST
    let checker = MemorySafetyChecker::new();
    checker.report()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_use_after_free() {
        let mut checker = MemorySafetyChecker::new();
        let r = checker.alloc(Region::Heap(0), 10, "line 1");
        checker.free(r, "line 2").unwrap();
        let result = checker.check_access(r, 0, AccessMode::Read, "line 3");
        assert!(matches!(result, Err(MemoryError::UseAfterFree { .. })));
    }

    #[test]
    fn test_double_free() {
        let mut checker = MemorySafetyChecker::new();
        let r = checker.alloc(Region::Heap(0), 10, "line 1");
        checker.free(r, "line 2").unwrap();
        let result = checker.free(r, "line 3");
        assert!(matches!(result, Err(MemoryError::DoubleFree { .. })));
    }

    #[test]
    fn test_data_race_read_write() {
        let mut checker = MemorySafetyChecker::new();
        let r = checker.alloc(Region::Heap(0), 10, "line 1");
        let _ref1 = checker.borrow(r, AccessMode::Read, "line 2").unwrap();
        let result = checker.borrow(r, AccessMode::Write, "line 3");
        assert!(matches!(result, Err(MemoryError::DataRace { .. })));
    }

    #[test]
    fn test_buffer_overflow() {
        let checker = MemorySafetyChecker::new();
        let mut checker = checker;
        let r = checker.alloc(Region::Heap(0), 10, "line 1");
        let result = checker.check_access(r, 15, AccessMode::Read, "line 2");
        assert!(matches!(result, Err(MemoryError::BufferOverflow { region: _, index: 15, size: 10, .. })));
    }

    #[test]
    fn test_leaked_buffer() {
        let mut checker = MemorySafetyChecker::new();
        let _r = checker.alloc(Region::Heap(0), 10, "line 1");
        let leaks = checker.end_scope();
        assert_eq!(leaks.len(), 1);
        assert!(matches!(leaks[0], MemoryError::LeakedBuffer { .. }));
    }

    #[test]
    fn test_valid_read_read_pattern() {
        let mut checker = MemorySafetyChecker::new();
        let r = checker.alloc(Region::Heap(0), 10, "line 1");
        let ref1 = checker.borrow(r, AccessMode::Read, "line 2").unwrap();
        let ref2 = checker.borrow(r, AccessMode::Read, "line 3").unwrap();
        checker.release(ref1);
        checker.release(ref2);
        assert!(checker.free(r, "line 4").is_ok());
    }

    #[test]
    fn test_gpu_device_not_ready() {
        let mut gpu = GpuMemoryTracker::new();
        gpu.alloc_device(1);
        let result = gpu.check_device_ready(1);
        assert!(matches!(result, Err(MemoryError::UninitializedRead { .. })));
    }

    #[test]
    fn test_gpu_transfer_conflict() {
        let mut gpu = GpuMemoryTracker::new();
        gpu.alloc_host(0);
        gpu.alloc_device(1);
        gpu.transfer(0, 1, TransferDirection::HostToDevice);
        let result = gpu.check_transfer_conflict(0);
        assert!(matches!(result, Err(MemoryError::DataRace { .. })));
    }

    #[test]
    fn test_exclusive_borrow_blocks_all() {
        let mut checker = MemorySafetyChecker::new();
        let r = checker.alloc(Region::Device(0), 100, "kernel");
        let _exc = checker.borrow(r, AccessMode::Exclusive, "line 1").unwrap();
        let result = checker.borrow(r, AccessMode::Read, "line 2");
        assert!(matches!(result, Err(MemoryError::DataRace { .. })));
    }

    #[test]
    fn test_release_then_reborrow_different_mode() {
        let mut checker = MemorySafetyChecker::new();
        let r = checker.alloc(Region::Heap(0), 10, "line 1");
        let ref1 = checker.borrow(r, AccessMode::Write, "line 2").unwrap();
        checker.release(ref1);
        let result = checker.borrow(r, AccessMode::Read, "line 3");
        assert!(result.is_ok());
    }
}

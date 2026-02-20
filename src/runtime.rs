//! GPU Runtime for Vortex.
//!
//! Provides GPU device management, kernel launching, and memory transfer
//! using CUDA Driver API (via FFI) or falling back to a simulation mode.
//!
//! Architecture:
//! - `GpuDevice`: represents a GPU device
//! - `GpuBuffer`: device memory allocation
//! - `GpuKernel`: a loaded kernel ready for launch
//! - `GpuRuntime`: top-level runtime managing devices and launches

use std::collections::HashMap;
use std::path::Path;

/// GPU backend type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuBackend {
    /// NVIDIA CUDA
    Cuda,
    /// AMD ROCm / HIP
    Rocm,
    /// Software simulation (no GPU needed)
    Simulated,
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackend::Cuda => write!(f, "CUDA"),
            GpuBackend::Rocm => write!(f, "ROCm"),
            GpuBackend::Simulated => write!(f, "Simulated"),
        }
    }
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub id: usize,
    pub name: String,
    pub backend: GpuBackend,
    pub compute_capability: (u32, u32),
    pub total_memory: u64,
    pub multiprocessors: u32,
    pub max_threads_per_block: u32,
    pub warp_size: u32,
    pub max_shared_memory: u64,
}

impl GpuDevice {
    /// Create a simulated device for testing
    pub fn simulated() -> Self {
        Self {
            id: 0,
            name: "Vortex Simulated GPU".to_string(),
            backend: GpuBackend::Simulated,
            compute_capability: (8, 0),
            total_memory: 16 * 1024 * 1024 * 1024, // 16 GB
            multiprocessors: 108,
            max_threads_per_block: 1024,
            warp_size: 32,
            max_shared_memory: 164 * 1024,
        }
    }
}

/// A GPU memory buffer
#[derive(Debug)]
pub struct GpuBuffer {
    pub id: u64,
    pub size: usize,
    pub device_id: usize,
    /// In simulation mode, we store data in CPU memory
    sim_data: Option<Vec<u8>>,
}

impl GpuBuffer {
    fn new_simulated(id: u64, size: usize, device_id: usize) -> Self {
        Self {
            id,
            size,
            device_id,
            sim_data: Some(vec![0u8; size]),
        }
    }

    /// Write data to the buffer (host -> device)
    pub fn write(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() > self.size {
            return Err(format!(
                "data size ({}) exceeds buffer size ({})",
                data.len(),
                self.size
            ));
        }
        if let Some(ref mut sim) = self.sim_data {
            sim[..data.len()].copy_from_slice(data);
            Ok(())
        } else {
            Err("buffer not in simulation mode".to_string())
        }
    }

    /// Read data from the buffer (device -> host)
    pub fn read(&self, size: usize) -> Result<Vec<u8>, String> {
        if size > self.size {
            return Err(format!(
                "read size ({}) exceeds buffer size ({})",
                size, self.size
            ));
        }
        if let Some(ref sim) = self.sim_data {
            Ok(sim[..size].to_vec())
        } else {
            Err("buffer not in simulation mode".to_string())
        }
    }

    /// Read as f64 array
    pub fn read_f64(&self, count: usize) -> Result<Vec<f64>, String> {
        let bytes = self.read(count * 8)?;
        Ok(bytes
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    /// Read as i64 array
    pub fn read_i64(&self, count: usize) -> Result<Vec<i64>, String> {
        let bytes = self.read(count * 8)?;
        Ok(bytes
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }
}

/// A loaded GPU kernel
#[derive(Debug, Clone)]
pub struct GpuKernel {
    pub name: String,
    pub source_path: String,
    pub param_count: usize,
    /// Grid dimensions (blocks_x, blocks_y, blocks_z)
    pub grid: (u32, u32, u32),
    /// Block dimensions (threads_x, threads_y, threads_z)
    pub block: (u32, u32, u32),
    /// Shared memory size in bytes
    pub shared_mem: u32,
}

/// Kernel launch parameters
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
}

impl LaunchConfig {
    pub fn new_1d(grid_x: u32, block_x: u32) -> Self {
        Self {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem: 0,
        }
    }

    pub fn new_2d(grid_x: u32, grid_y: u32, block_x: u32, block_y: u32) -> Self {
        Self {
            grid: (grid_x, grid_y, 1),
            block: (block_x, block_y, 1),
            shared_mem: 0,
        }
    }

    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem = bytes;
        self
    }
}

/// Kernel argument types
#[derive(Debug, Clone)]
pub enum KernelArg {
    Buffer(u64),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    UInt64(u64),
}

/// Launch result from kernel execution
#[derive(Debug)]
pub struct LaunchResult {
    pub kernel_name: String,
    pub elapsed_us: u64,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
}

/// The GPU Runtime — manages devices, memory, and kernel execution
pub struct GpuRuntime {
    backend: GpuBackend,
    devices: Vec<GpuDevice>,
    buffers: HashMap<u64, GpuBuffer>,
    kernels: HashMap<String, GpuKernel>,
    next_buffer_id: u64,
    verbose: bool,
}

impl GpuRuntime {
    /// Create a new runtime with auto-detected backend
    pub fn new(verbose: bool) -> Self {
        let backend = detect_backend();
        let devices = match backend {
            GpuBackend::Simulated => vec![GpuDevice::simulated()],
            _ => vec![GpuDevice::simulated()], // TODO: real device enumeration via FFI
        };

        if verbose {
            eprintln!(
                "[runtime] initialized {} backend with {} device(s)",
                backend,
                devices.len()
            );
            for dev in &devices {
                eprintln!(
                    "[runtime]   device {}: {} ({}MB, {} SMs)",
                    dev.id,
                    dev.name,
                    dev.total_memory / (1024 * 1024),
                    dev.multiprocessors
                );
            }
        }

        Self {
            backend,
            devices,
            buffers: HashMap::new(),
            kernels: HashMap::new(),
            next_buffer_id: 1,
            verbose,
        }
    }

    /// Create a simulated runtime (no GPU required)
    pub fn simulated() -> Self {
        Self {
            backend: GpuBackend::Simulated,
            devices: vec![GpuDevice::simulated()],
            buffers: HashMap::new(),
            kernels: HashMap::new(),
            next_buffer_id: 1,
            verbose: false,
        }
    }

    /// Get available devices
    pub fn devices(&self) -> &[GpuDevice] {
        &self.devices
    }

    /// Get the backend type
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    /// Allocate a device buffer
    pub fn alloc(&mut self, size: usize) -> Result<u64, String> {
        self.alloc_on(0, size)
    }

    /// Allocate a device buffer on a specific device
    pub fn alloc_on(&mut self, device_id: usize, size: usize) -> Result<u64, String> {
        if device_id >= self.devices.len() {
            return Err(format!("device {} not found", device_id));
        }

        let buf_id = self.next_buffer_id;
        self.next_buffer_id += 1;

        let buffer = GpuBuffer::new_simulated(buf_id, size, device_id);
        if self.verbose {
            eprintln!(
                "[runtime] allocated buffer {} ({} bytes) on device {}",
                buf_id, size, device_id
            );
        }

        self.buffers.insert(buf_id, buffer);
        Ok(buf_id)
    }

    /// Free a device buffer
    pub fn free(&mut self, buf_id: u64) -> Result<(), String> {
        if self.buffers.remove(&buf_id).is_some() {
            if self.verbose {
                eprintln!("[runtime] freed buffer {}", buf_id);
            }
            Ok(())
        } else {
            Err(format!("buffer {} not found", buf_id))
        }
    }

    /// Write data to a buffer (host -> device)
    pub fn memcpy_h2d(&mut self, buf_id: u64, data: &[u8]) -> Result<(), String> {
        let buffer = self
            .buffers
            .get_mut(&buf_id)
            .ok_or_else(|| format!("buffer {} not found", buf_id))?;
        buffer.write(data)
    }

    /// Read data from a buffer (device -> host)
    pub fn memcpy_d2h(&self, buf_id: u64, size: usize) -> Result<Vec<u8>, String> {
        let buffer = self
            .buffers
            .get(&buf_id)
            .ok_or_else(|| format!("buffer {} not found", buf_id))?;
        buffer.read(size)
    }

    /// Upload i64 array to device
    pub fn upload_i64(&mut self, data: &[i64]) -> Result<u64, String> {
        let size = data.len() * 8;
        let buf_id = self.alloc(size)?;
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.memcpy_h2d(buf_id, &bytes)?;
        Ok(buf_id)
    }

    /// Upload f64 array to device
    pub fn upload_f64(&mut self, data: &[f64]) -> Result<u64, String> {
        let size = data.len() * 8;
        let buf_id = self.alloc(size)?;
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.memcpy_h2d(buf_id, &bytes)?;
        Ok(buf_id)
    }

    /// Download i64 array from device
    pub fn download_i64(&self, buf_id: u64, count: usize) -> Result<Vec<i64>, String> {
        let buffer = self
            .buffers
            .get(&buf_id)
            .ok_or_else(|| format!("buffer {} not found", buf_id))?;
        buffer.read_i64(count)
    }

    /// Download f64 array from device
    pub fn download_f64(&self, buf_id: u64, count: usize) -> Result<Vec<f64>, String> {
        let buffer = self
            .buffers
            .get(&buf_id)
            .ok_or_else(|| format!("buffer {} not found", buf_id))?;
        buffer.read_f64(count)
    }

    /// Load a kernel from a PTX file
    pub fn load_kernel(&mut self, name: &str, ptx_path: &Path) -> Result<(), String> {
        if !ptx_path.exists() {
            return Err(format!("PTX file not found: {}", ptx_path.display()));
        }

        let kernel = GpuKernel {
            name: name.to_string(),
            source_path: ptx_path.display().to_string(),
            param_count: 0,
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem: 0,
        };

        if self.verbose {
            eprintln!(
                "[runtime] loaded kernel '{}' from {}",
                name,
                ptx_path.display()
            );
        }

        self.kernels.insert(name.to_string(), kernel);
        Ok(())
    }

    /// Launch a kernel with given configuration and arguments
    pub fn launch(
        &mut self,
        name: &str,
        config: &LaunchConfig,
        args: &[KernelArg],
    ) -> Result<LaunchResult, String> {
        let _kernel = self
            .kernels
            .get(name)
            .ok_or_else(|| format!("kernel '{}' not loaded", name))?;

        if self.verbose {
            eprintln!(
                "[runtime] launching kernel '{}' grid=({},{},{}) block=({},{},{}) shared={}",
                name,
                config.grid.0,
                config.grid.1,
                config.grid.2,
                config.block.0,
                config.block.1,
                config.block.2,
                config.shared_mem,
            );
        }

        let start = std::time::Instant::now();

        match self.backend {
            GpuBackend::Simulated => {
                self.simulate_kernel(name, config, args)?;
            }
            _ => {
                // TODO: real CUDA/ROCm kernel launch via FFI
                self.simulate_kernel(name, config, args)?;
            }
        }

        let elapsed = start.elapsed().as_micros() as u64;

        if self.verbose {
            eprintln!("[runtime] kernel '{}' completed in {}us", name, elapsed);
        }

        Ok(LaunchResult {
            kernel_name: name.to_string(),
            elapsed_us: elapsed,
            grid: config.grid,
            block: config.block,
        })
    }

    /// Simulate kernel execution on CPU (for testing without GPU)
    fn simulate_kernel(
        &mut self,
        name: &str,
        config: &LaunchConfig,
        args: &[KernelArg],
    ) -> Result<(), String> {
        let total_threads = (config.grid.0 * config.grid.1 * config.grid.2)
            as u64
            * (config.block.0 * config.block.1 * config.block.2) as u64;

        if self.verbose {
            eprintln!(
                "[runtime] simulating kernel '{}' with {} threads, {} args",
                name,
                total_threads,
                args.len()
            );
        }

        // For simulation, we do a simple element-wise operation
        // based on the kernel name convention
        if name.contains("add") || name.contains("vadd") {
            self.simulate_elementwise_add(args, total_threads as usize)?;
        } else if name.contains("mul") || name.contains("vmul") {
            self.simulate_elementwise_mul(args, total_threads as usize)?;
        } else if name.contains("scale") {
            self.simulate_scale(args, total_threads as usize)?;
        }
        // Unknown kernels just succeed silently in simulation

        Ok(())
    }

    fn simulate_elementwise_add(&mut self, args: &[KernelArg], n: usize) -> Result<(), String> {
        if args.len() < 3 {
            return Ok(());
        }
        let (a_id, b_id, c_id) = match (&args[0], &args[1], &args[2]) {
            (KernelArg::Buffer(a), KernelArg::Buffer(b), KernelArg::Buffer(c)) => (*a, *b, *c),
            _ => return Ok(()),
        };

        let a_data = self.download_f64(a_id, n)?;
        let b_data = self.download_f64(b_id, n)?;
        let c_data: Vec<f64> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a + b).collect();
        let bytes: Vec<u8> = c_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.memcpy_h2d(c_id, &bytes)
    }

    fn simulate_elementwise_mul(&mut self, args: &[KernelArg], n: usize) -> Result<(), String> {
        if args.len() < 3 {
            return Ok(());
        }
        let (a_id, b_id, c_id) = match (&args[0], &args[1], &args[2]) {
            (KernelArg::Buffer(a), KernelArg::Buffer(b), KernelArg::Buffer(c)) => (*a, *b, *c),
            _ => return Ok(()),
        };

        let a_data = self.download_f64(a_id, n)?;
        let b_data = self.download_f64(b_id, n)?;
        let c_data: Vec<f64> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a * b).collect();
        let bytes: Vec<u8> = c_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.memcpy_h2d(c_id, &bytes)
    }

    fn simulate_scale(&mut self, args: &[KernelArg], n: usize) -> Result<(), String> {
        if args.len() < 3 {
            return Ok(());
        }
        let (a_id, scale, c_id) = match (&args[0], &args[1], &args[2]) {
            (KernelArg::Buffer(a), KernelArg::Float64(s), KernelArg::Buffer(c)) => (*a, *s, *c),
            _ => return Ok(()),
        };

        let a_data = self.download_f64(a_id, n)?;
        let c_data: Vec<f64> = a_data.iter().map(|a| a * scale).collect();
        let bytes: Vec<u8> = c_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.memcpy_h2d(c_id, &bytes)
    }

    /// Print runtime status
    pub fn print_status(&self) {
        println!("Vortex GPU Runtime Status:");
        println!("==========================");
        println!("  Backend: {}", self.backend);
        println!("  Devices: {}", self.devices.len());
        for dev in &self.devices {
            println!(
                "    [{}] {} (CC {}.{}, {}MB, {} SMs)",
                dev.id,
                dev.name,
                dev.compute_capability.0,
                dev.compute_capability.1,
                dev.total_memory / (1024 * 1024),
                dev.multiprocessors
            );
        }
        println!("  Allocated buffers: {}", self.buffers.len());
        println!("  Loaded kernels: {}", self.kernels.len());
        for (name, k) in &self.kernels {
            println!("    {} (from {})", name, k.source_path);
        }
    }
}

/// Detect available GPU backend
fn detect_backend() -> GpuBackend {
    // Check for CUDA
    if std::process::Command::new("nvidia-smi")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return GpuBackend::Cuda;
    }

    // Check for ROCm
    if std::process::Command::new("rocm-smi")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return GpuBackend::Rocm;
    }

    GpuBackend::Simulated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulated_runtime() {
        let rt = GpuRuntime::simulated();
        assert_eq!(rt.backend(), GpuBackend::Simulated);
        assert_eq!(rt.devices().len(), 1);
        assert_eq!(rt.devices()[0].name, "Vortex Simulated GPU");
    }

    #[test]
    fn test_buffer_alloc_free() {
        let mut rt = GpuRuntime::simulated();
        let buf = rt.alloc(1024).unwrap();
        assert!(buf > 0);
        rt.free(buf).unwrap();
        assert!(rt.free(buf).is_err()); // double free
    }

    #[test]
    fn test_memcpy_roundtrip() {
        let mut rt = GpuRuntime::simulated();
        let data = vec![1i64, 2, 3, 4, 5];
        let buf = rt.upload_i64(&data).unwrap();
        let result = rt.download_i64(buf, 5).unwrap();
        assert_eq!(data, result);
    }

    #[test]
    fn test_f64_roundtrip() {
        let mut rt = GpuRuntime::simulated();
        let data = vec![1.5f64, 2.7, 3.14, 4.0, 5.5];
        let buf = rt.upload_f64(&data).unwrap();
        let result = rt.download_f64(buf, 5).unwrap();
        assert_eq!(data, result);
    }

    #[test]
    fn test_simulated_vector_add() {
        let mut rt = GpuRuntime::simulated();
        let n = 4;

        let a = rt.upload_f64(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = rt.upload_f64(&[10.0, 20.0, 30.0, 40.0]).unwrap();
        let c = rt.alloc(n * 8).unwrap();

        // Load a fake kernel
        let kernel_path = std::env::temp_dir().join("test_vadd.ptx");
        std::fs::write(&kernel_path, "// fake PTX").unwrap();
        rt.load_kernel("vadd", &kernel_path).unwrap();

        let config = LaunchConfig::new_1d(1, n as u32);
        let result = rt
            .launch(
                "vadd",
                &config,
                &[
                    KernelArg::Buffer(a),
                    KernelArg::Buffer(b),
                    KernelArg::Buffer(c),
                ],
            )
            .unwrap();

        assert_eq!(result.kernel_name, "vadd");

        let output = rt.download_f64(c, n).unwrap();
        assert_eq!(output, vec![11.0, 22.0, 33.0, 44.0]);

        let _ = std::fs::remove_file(kernel_path);
    }

    #[test]
    fn test_launch_config() {
        let config = LaunchConfig::new_1d(256, 256);
        assert_eq!(config.grid, (256, 1, 1));
        assert_eq!(config.block, (256, 1, 1));

        let config2d = LaunchConfig::new_2d(16, 16, 32, 32).with_shared_mem(4096);
        assert_eq!(config2d.grid, (16, 16, 1));
        assert_eq!(config2d.shared_mem, 4096);
    }

    #[test]
    fn test_detect_backend() {
        let backend = detect_backend();
        // In CI/test environments, this will likely be Simulated
        assert!(matches!(
            backend,
            GpuBackend::Cuda | GpuBackend::Rocm | GpuBackend::Simulated
        ));
    }
}

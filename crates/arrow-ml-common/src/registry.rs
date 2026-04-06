//! Runtime backend discovery and dispatch.
//!
//! On first access, the registry scans well-known directories for shared
//! libraries matching `libarrow_ml_backend_*{.dylib,.so,.dll}`, loads
//! each one, and sorts them by priority (highest first).  Kernel dispatch
//! functions iterate the list and call the first backend that succeeds.

use crate::backend::{AkMatmulF32Fn, AkMatmulF64Fn, Backend, AK_OK};
use std::path::PathBuf;
use std::sync::OnceLock;

/// Returns the glob pattern for backend shared libraries on the current OS.
fn dylib_glob_pattern() -> &'static str {
    if cfg!(target_os = "macos") {
        "libarrow_ml_backend_*.dylib"
    } else if cfg!(target_os = "windows") {
        "arrow_ml_backend_*.dll"
    } else {
        "libarrow_ml_backend_*.so"
    }
}

/// Global backend registry, initialized once on first use.
static REGISTRY: OnceLock<BackendRegistry> = OnceLock::new();

pub struct BackendRegistry {
    backends: Vec<Backend>,
}

impl BackendRegistry {
    /// Returns a reference to the lazily-initialized global registry.
    pub fn global() -> &'static BackendRegistry {
        REGISTRY.get_or_init(BackendRegistry::discover)
    }

    /// Scan the search paths, load every valid backend, sort by priority.
    fn discover() -> BackendRegistry {
        let mut backends = Vec::new();

        for dir in Self::search_dirs() {
            let pattern = dylib_glob_pattern();
            let glob = match glob::glob(dir.join(&pattern).to_string_lossy().as_ref()) {
                Ok(g) => g,
                Err(_) => continue,
            };
            for entry in glob.flatten() {
                if let Some(backend) = Backend::load(&entry) {
                    eprintln!(
                        "[arrow-ml] loaded backend {:?} (priority {}) from {}",
                        backend.name,
                        backend.priority,
                        entry.display()
                    );
                    backends.push(backend);
                }
            }
        }

        // Highest priority first
        backends.sort_by(|a, b| b.priority.cmp(&a.priority));

        if backends.is_empty() {
            eprintln!("[arrow-ml] no GPU backends found, using CPU kernels");
        }

        BackendRegistry { backends }
    }

    /// Directories to scan for backend shared libraries.
    fn search_dirs() -> Vec<PathBuf> {
        let mut dirs = Vec::new();

        // 1. Explicit env var
        if let Ok(dir) = std::env::var("ARROW_ML_BACKEND_DIR") {
            dirs.push(PathBuf::from(dir));
        }

        // 2. Next to the running executable
        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                dirs.push(exe_dir.to_path_buf());
            }
        }

        // 3. Cargo target dirs (dev convenience)
        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            let workspace_root = PathBuf::from(manifest_dir)
                .ancestors()
                .nth(2) // crates/<name>/ -> workspace root
                .map(|p| p.to_path_buf());
            if let Some(root) = workspace_root {
                dirs.push(root.join("target").join("debug"));
                dirs.push(root.join("target").join("release"));
            }
        }

        dirs
    }

    /// Returns the names of all loaded backends, in priority order.
    pub fn loaded_backends(&self) -> Vec<&str> {
        self.backends.iter().map(|b| b.name.as_str()).collect()
    }

    /// Returns the highest-priority `matmul_f32` function pointer, if any
    /// backend provides one.
    pub fn matmul_f32(&self) -> Option<AkMatmulF32Fn> {
        self.backends.iter().find_map(|b| b.matmul_f32)
    }

    /// Returns the highest-priority `matmul_f64` function pointer, if any
    /// backend provides one.
    pub fn matmul_f64(&self) -> Option<AkMatmulF64Fn> {
        self.backends.iter().find_map(|b| b.matmul_f64)
    }

    /// Convenience: run matmul_f32 through the best backend. Returns `None`
    /// if no backend supports it or the backend returned an error.
    pub fn try_matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<()> {
        let f = self.matmul_f32()?;
        let rc = unsafe {
            f(
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                m as u32,
                k as u32,
                n as u32,
            )
        };
        if rc == AK_OK {
            Some(())
        } else {
            None
        }
    }

    /// Convenience: run matmul_f64 through the best backend.
    pub fn try_matmul_f64(
        &self,
        a: &[f64],
        b: &[f64],
        c: &mut [f64],
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<()> {
        let f = self.matmul_f64()?;
        let rc = unsafe {
            f(
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                m as u32,
                k as u32,
                n as u32,
            )
        };
        if rc == AK_OK {
            Some(())
        } else {
            None
        }
    }
}

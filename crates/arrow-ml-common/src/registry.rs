//! Runtime backend discovery and dispatch.
//!
//! On first access, the registry scans well-known directories for backends.
//! Two discovery mechanisms run in order:
//!
//! 1. **Manifest scan.** JSON files matching `*.json` in any manifest
//!    search directory are parsed as [`crate::BackendManifest`]s and the
//!    libraries they describe are loaded.
//! 2. **Glob fallback.** Any shared libraries matching
//!    `libarrow_ml_backend_*{.dylib,.so,.dll}` in a library search directory
//!    are loaded directly. Libraries already loaded via a manifest are skipped.
//!
//! After discovery, backends are sorted by priority (highest first). Kernel
//! dispatch functions iterate the list and call the first backend that succeeds.

use crate::backend::{AmMatmulF32Fn, AmMatmulF64Fn, Backend, AM_OK};
use crate::manifest::BackendManifest;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Per-user backend directory:
/// - Unix: `$HOME/.arrow-ml/backends`
/// - Windows: `%LOCALAPPDATA%\arrow-ml\backends`
fn user_backend_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var_os("LOCALAPPDATA").map(|p| PathBuf::from(p).join("arrow-ml").join("backends"))
    }
    #[cfg(not(windows))]
    {
        std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".arrow-ml").join("backends"))
    }
}

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
        let mut loaded_paths: HashSet<PathBuf> = HashSet::new();

        // 1. Manifest-driven discovery
        for dir in Self::manifest_search_dirs() {
            let pattern = dir.join("*.json");
            let glob = match glob::glob(pattern.to_string_lossy().as_ref()) {
                Ok(g) => g,
                Err(_) => continue,
            };
            for manifest_path in glob.flatten() {
                let manifest = match BackendManifest::from_path(&manifest_path) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!(
                            "[arrow-ml] skipping malformed manifest {}: {}",
                            manifest_path.display(),
                            e
                        );
                        continue;
                    }
                };
                let lib_path = manifest.resolve_library_path(&manifest_path);
                let canonical =
                    std::fs::canonicalize(&lib_path).unwrap_or_else(|_| lib_path.clone());
                if !loaded_paths.insert(canonical) {
                    continue;
                }
                if let Some(backend) = Backend::load(&lib_path) {
                    eprintln!(
                        "[arrow-ml] loaded backend {:?} (priority {}) from manifest {}",
                        backend.name,
                        backend.priority,
                        manifest_path.display()
                    );
                    backends.push(backend);
                }
            }
        }

        // 2. Glob fallback
        for dir in Self::library_search_dirs() {
            let pattern = dylib_glob_pattern();
            let glob = match glob::glob(dir.join(pattern).to_string_lossy().as_ref()) {
                Ok(g) => g,
                Err(_) => continue,
            };
            for entry in glob.flatten() {
                let canonical = std::fs::canonicalize(&entry).unwrap_or_else(|_| entry.clone());
                if !loaded_paths.insert(canonical) {
                    continue;
                }
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
        backends.sort_by_key(|b| std::cmp::Reverse(b.priority));

        if backends.is_empty() {
            eprintln!("[arrow-ml] no GPU backends found, using CPU kernels");
        }

        BackendRegistry { backends }
    }

    /// Directories to scan for `*.json` backend manifests.
    fn manifest_search_dirs() -> Vec<PathBuf> {
        let mut dirs = Vec::new();

        if let Ok(dir) = std::env::var("ARROW_ML_BACKEND_MANIFEST_DIR") {
            dirs.push(PathBuf::from(dir));
        }

        if let Some(user) = user_backend_dir() {
            dirs.push(user);
        }

        #[cfg(unix)]
        dirs.push(PathBuf::from("/etc/arrow-ml/backends"));

        dirs.extend(Self::shared_runtime_dirs());

        dirs
    }

    /// Directories to scan for backend shared libraries (glob fallback).
    fn library_search_dirs() -> Vec<PathBuf> {
        let mut dirs = Vec::new();

        if let Ok(dir) = std::env::var("ARROW_ML_BACKEND_DIR") {
            dirs.push(PathBuf::from(dir));
        }

        if let Some(user) = user_backend_dir() {
            dirs.push(user);
        }

        #[cfg(unix)]
        dirs.push(PathBuf::from("/etc/arrow-ml/backends"));

        dirs.extend(Self::shared_runtime_dirs());

        dirs
    }

    /// Directories searched by both manifest and library discovery: next to
    /// the running executable, plus workspace `target/{debug,release}` for
    /// dev convenience.
    fn shared_runtime_dirs() -> Vec<PathBuf> {
        let mut dirs = Vec::new();

        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                dirs.push(exe_dir.to_path_buf());
            }
        }

        if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            let workspace_root = PathBuf::from(manifest_dir)
                .ancestors()
                .nth(2)
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
    pub fn matmul_f32(&self) -> Option<AmMatmulF32Fn> {
        self.backends.iter().find_map(|b| b.matmul_f32)
    }

    /// Returns the highest-priority `matmul_f64` function pointer, if any
    /// backend provides one.
    pub fn matmul_f64(&self) -> Option<AmMatmulF64Fn> {
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
        if rc == AM_OK {
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
        if rc == AM_OK {
            Some(())
        } else {
            None
        }
    }
}

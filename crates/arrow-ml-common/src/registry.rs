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
//! After discovery, backends are sorted by priority (highest first). Per-kernel
//! "best backend" selectors (e.g. [`BackendRegistry::best_matmul`]) return the
//! highest-priority backend that exports a given kernel's symbol family.

use crate::backend::Backend;
use crate::device_tensor::AmDeviceType;
use crate::manifest::BackendManifest;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

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
    backends: Vec<Arc<Backend>>,
}

impl BackendRegistry {
    /// Returns a reference to the lazily-initialized global registry.
    pub fn global() -> &'static BackendRegistry {
        REGISTRY.get_or_init(BackendRegistry::discover)
    }

    /// Scan the search paths, load every valid backend, sort by priority.
    fn discover() -> BackendRegistry {
        let mut backends: Vec<Arc<Backend>> = Vec::new();
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
                    backends.push(Arc::new(backend));
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
                    backends.push(Arc::new(backend));
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

    /// Returns the highest-priority backend that exports the matmul symbol
    /// family, or `None` if no loaded backend implements matmul.
    ///
    /// This does **not** check dtype/device support. Prefer
    /// [`BackendRegistry::best_matmul_for`] if you care which dtype and
    /// device the kernel needs to handle.
    pub fn best_matmul(&self) -> Option<Arc<Backend>> {
        self.backends.iter().find(|b| b.matmul.is_some()).cloned()
    }

    /// Returns the highest-priority backend whose matmul kernel reports
    /// support for `(dtype, device_type)`, or `None` if no loaded backend
    /// supports that combination.
    ///
    /// Iterates through every loaded backend in priority order — so if the
    /// top-priority backend exports matmul but only supports a different
    /// dtype, the next backend is tried, and so on.
    pub fn best_matmul_for(&self, dtype: i32, device_type: i32) -> Option<Arc<Backend>> {
        self.backends
            .iter()
            .find(|b| {
                b.matmul
                    .as_ref()
                    .is_some_and(|ops| ops.supports_dtype(dtype, device_type))
            })
            .cloned()
    }

    pub fn best_softmax(&self) -> Option<Arc<Backend>> {
        self.backends.iter().find(|b| b.softmax.is_some()).cloned()
    }

    pub fn best_softmax_for(&self, dtype: i32, device_type: i32) -> Option<Arc<Backend>> {
        self.backends
            .iter()
            .find(|b| {
                b.softmax
                    .as_ref()
                    .is_some_and(|ops| ops.supports_dtype(dtype, device_type))
            })
            .cloned()
    }

    /// Returns the highest-priority backend whose `device_type` matches
    /// the requested device, or `None` if no loaded backend serves it.
    ///
    /// Repeated calls return the same `Arc<Backend>` (verifiable via
    /// [`Arc::ptr_eq`]) — the underlying backend handle is interned in
    /// the registry's storage and clones share the same allocation.
    pub fn best_for_device(
        &self,
        device_type: AmDeviceType,
        _device_id: i64,
    ) -> Option<Arc<Backend>> {
        let target = device_type as i32;
        self.backends
            .iter()
            .find(|b| b.device_type == target)
            .cloned()
    }
}

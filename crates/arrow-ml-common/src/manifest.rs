//! Backend manifest format.
//!
//! Manifests are small JSON files that describe a backend without requiring
//! the loader to `dlopen` the library just to read its name or priority.
//! Inspired by Vulkan's ICD manifest layout.
//!
//! # Example
//!
//! ```json
//! {
//!     "file_format_version": "1.0.0",
//!     "backend": {
//!         "name": "metal",
//!         "library_path": "libarrow_ml_backend_metal.dylib",
//!         "abi_version": 1,
//!         "priority": 100,
//!         "description": "Apple Metal GPU backend"
//!     }
//! }
//! ```
//!
//! `library_path` may be:
//! - absolute (e.g. `/usr/local/lib/libarrow_ml_backend_metal.dylib`)
//! - relative to the manifest file's directory (e.g. `./libarrow_ml_backend_metal.dylib`)
//! - a bare filename, resolved by the system loader's search rules

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Top-level manifest document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendManifest {
    /// Manifest schema version. Currently `"1.0.0"`.
    pub file_format_version: String,
    pub backend: BackendDescriptor,
}

/// Per-backend metadata stored in a manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendDescriptor {
    /// Human-readable backend name (e.g. `"metal"`).
    pub name: String,
    /// Path to the cdylib. May be absolute, relative to the manifest, or a bare filename.
    pub library_path: String,
    /// ABI version this backend was built against. Used for an early reject
    /// before `dlopen` is called.
    pub abi_version: u32,
    /// Higher means more preferred when multiple backends are present.
    #[serde(default)]
    pub priority: u32,
    /// Optional human-readable description.
    #[serde(default)]
    pub description: Option<String>,
}

impl BackendManifest {
    /// Read and parse a manifest file from disk.
    pub fn from_path(path: &Path) -> Result<Self, ManifestError> {
        let bytes = std::fs::read(path).map_err(ManifestError::Io)?;
        serde_json::from_slice(&bytes).map_err(ManifestError::Parse)
    }

    /// Resolve `backend.library_path` against the directory containing the
    /// given manifest file. Absolute paths are returned unchanged; relative
    /// paths are joined to the manifest's parent directory; bare filenames
    /// are returned as-is so the OS loader can search standard library paths.
    pub fn resolve_library_path(&self, manifest_path: &Path) -> PathBuf {
        let lib = Path::new(&self.backend.library_path);
        if lib.is_absolute() {
            return lib.to_path_buf();
        }
        if lib.components().count() == 1 {
            return lib.to_path_buf();
        }
        match manifest_path.parent() {
            Some(dir) => dir.join(lib),
            None => lib.to_path_buf(),
        }
    }
}

/// Errors that can occur when reading or parsing a manifest file.
#[derive(Debug)]
pub enum ManifestError {
    Io(std::io::Error),
    Parse(serde_json::Error),
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManifestError::Io(e) => write!(f, "i/o error reading manifest: {}", e),
            ManifestError::Parse(e) => write!(f, "failed to parse manifest: {}", e),
        }
    }
}

impl std::error::Error for ManifestError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_manifest() {
        let json = r#"{
            "file_format_version": "1.0.0",
            "backend": {
                "name": "metal",
                "library_path": "libarrow_ml_backend_metal.dylib",
                "abi_version": 1
            }
        }"#;
        let m: BackendManifest = serde_json::from_str(json).unwrap();
        assert_eq!(m.backend.name, "metal");
        assert_eq!(m.backend.abi_version, 1);
        assert_eq!(m.backend.priority, 0);
        assert!(m.backend.description.is_none());
    }

    #[test]
    fn parses_full_manifest() {
        let json = r#"{
            "file_format_version": "1.0.0",
            "backend": {
                "name": "metal",
                "library_path": "/usr/local/lib/libarrow_ml_backend_metal.dylib",
                "abi_version": 1,
                "priority": 100,
                "description": "Apple Metal GPU backend"
            }
        }"#;
        let m: BackendManifest = serde_json::from_str(json).unwrap();
        assert_eq!(m.backend.priority, 100);
        assert_eq!(m.backend.description.as_deref(), Some("Apple Metal GPU backend"));
    }

    #[test]
    fn resolves_absolute_path_unchanged() {
        let m = BackendManifest {
            file_format_version: "1.0.0".into(),
            backend: BackendDescriptor {
                name: "x".into(),
                library_path: "/abs/path/lib.so".into(),
                abi_version: 1,
                priority: 0,
                description: None,
            },
        };
        let resolved = m.resolve_library_path(Path::new("/some/dir/manifest.json"));
        assert_eq!(resolved, PathBuf::from("/abs/path/lib.so"));
    }

    #[test]
    fn resolves_relative_path_against_manifest_dir() {
        let m = BackendManifest {
            file_format_version: "1.0.0".into(),
            backend: BackendDescriptor {
                name: "x".into(),
                library_path: "./sub/lib.so".into(),
                abi_version: 1,
                priority: 0,
                description: None,
            },
        };
        let resolved = m.resolve_library_path(Path::new("/some/dir/manifest.json"));
        assert_eq!(resolved, PathBuf::from("/some/dir/./sub/lib.so"));
    }

    #[test]
    fn bare_filename_returned_as_is() {
        let m = BackendManifest {
            file_format_version: "1.0.0".into(),
            backend: BackendDescriptor {
                name: "x".into(),
                library_path: "libfoo.so".into(),
                abi_version: 1,
                priority: 0,
                description: None,
            },
        };
        let resolved = m.resolve_library_path(Path::new("/some/dir/manifest.json"));
        assert_eq!(resolved, PathBuf::from("libfoo.so"));
    }
}

//! GPU architecture and ROCm toolchain detection.

use crate::error::KernelError;
use std::collections::HashMap;
use std::process::Command;
use std::sync::{Mutex, OnceLock};

/// Resolve the target architecture for device `ordinal`, e.g. "gfx1101".
///
/// Queried per device rather than once per machine: on a heterogeneous box —
/// an iGPU next to a dGPU, or two different dGPUs — a global answer would build
/// device 1's code objects for device 0's architecture, which only surfaces at
/// the first launch as an opaque "invalid device function". `gcnArchName` also
/// carries the target features (`gfx1101:xnack-`) that `--offload-arch` wants.
///
/// Guessing here is worse than failing, so a device that reports something
/// other than a `gfx*` target is an error rather than a fallback.
/// Memoised per ordinal: `RocmDevice::new` runs on every test and short-lived
/// process, and the answer cannot change while the process lives.
pub(crate) fn gpu_arch(ordinal: usize) -> Result<String, KernelError> {
    if let Ok(arch) = std::env::var("CANDLE_ROCM_ARCH") {
        return Ok(arch);
    }

    static ARCHES: OnceLock<Mutex<HashMap<usize, String>>> = OnceLock::new();
    let arches = ARCHES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(cached) = arches.lock() {
        if let Some(arch) = cached.get(&ordinal) {
            return Ok(arch.clone());
        }
    }

    let arch = gcn_arch_name(ordinal)?;
    if !arch.starts_with("gfx") {
        return Err(KernelError::Compilation(format!(
            "device {ordinal} reports architecture `{arch}`, which is not an AMD GPU target; \
             set CANDLE_ROCM_ARCH to build kernels anyway"
        )));
    }
    if let Ok(mut cached) = arches.lock() {
        cached.insert(ordinal, arch.clone());
    }
    Ok(arch)
}

/// Which set of MMQ tile constants `quantized.cu` is compiled with.
///
/// The kernel carries one geometry per architecture and picks between them on
/// the `RDNA2`/`RDNA3` defines this module supplies. The host has to launch the
/// *same* geometry — the grid and block are computed from `mmq_x`, `mmq_y` and
/// `nwarps` — so the choice cannot stay inside the compiler flags.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MmqTiles {
    /// `MMQ_*_AMPERE`, `nwarps = 4`. Every non-RDNA target.
    Ampere,
    /// `MMQ_*_RDNA2`, `nwarps = 8`. RDNA2 and later.
    Rdna2,
}

/// The `RDNA*` define that selects `arch`'s MMQ geometry, or `None` where the
/// kernel's default (Ampere) set stands.
///
/// gfx12xx (RDNA4) gets its own `RDNA4` define rather than reusing `RDNA3`.
/// `quantized.cu`'s dp4a tile geometry is identical under both (see
/// [`mmq_tiles`]), but the WMMA kernels in `candle-kernels/src/mmq_gguf`
/// branch on `RDNA3` vs `RDNA4` for the matrix-core intrinsic and the input
/// data layout (`I_MAJOR_MIRRORED` vs `I_MAJOR`). Collapsing them here would
/// make RDNA4 silently compile RDNA3's WMMA path. gfx101x (RDNA1) is
/// deliberately absent. Its tile set is a third variant, and no candle CI or
/// developer machine has the hardware to check it.
pub(crate) fn rdna_define(arch: &str) -> Option<&'static str> {
    match arch_number(arch)? {
        1030..=1039 => Some("RDNA2"),
        1100..=1199 => Some("RDNA3"),
        1200..=1299 => Some("RDNA4"),
        _ => None,
    }
}

/// [`MmqTiles`] for `arch`, in lockstep with [`rdna_define`].
pub(crate) fn mmq_tiles(arch: &str) -> MmqTiles {
    match rdna_define(arch) {
        Some(_) => MmqTiles::Rdna2,
        None => MmqTiles::Ampere,
    }
}

/// Whether `arch` has AMD WMMA (matrix-core) support for the `mmq_gguf`
/// kernels. True for RDNA3 (gfx11xx) and RDNA4 (gfx12xx).
pub(crate) fn has_wmma(arch: &str) -> bool {
    matches!(rdna_define(arch), Some("RDNA3" | "RDNA4"))
}

/// The AMD `cc` value `mmq_gguf`'s host-side functions expect
/// (`amd_wmma_available`, `get_mmq_y_host`, `get_mmq_x_max_host`,
/// `mmq_get_granularity_host` in `mmq_common.cuh`/`mmq_gguf.cuh`), or `None`
/// if `arch` cannot be parsed this way.
///
/// This is deliberately not [`arch_number`]'s plain decimal parse.
/// `mmq_common.cuh` defines e.g. `GGML_CUDA_CC_RDNA4` as
/// `GGML_CUDA_CC_OFFSET_AMD + 0x1200`, and the comparisons against it
/// (`GGML_CUDA_CC_IS_RDNA4`) only land correctly if the `cc` value this
/// function returns uses the same encoding: the last two characters of the
/// stripped `gfxNNNN` name parsed as a *hex* minor version, the remainder
/// parsed as a *hex* major version, combined as
/// `OFFSET_AMD + major * 0x100 + minor`. This mirrors upstream llama.cpp's
/// own `ggml_cuda_parse_id` (`ggml-cuda.cu`) bit for bit; a decimal parse of
/// "1201" would give a smaller number than `GGML_CUDA_CC_RDNA4` and make
/// `amd_wmma_available` wrongly report `false` for this card's gfx1201.
pub(crate) fn amd_cc(arch: &str) -> Option<i32> {
    const OFFSET_AMD: i32 = 0x0100_0000;
    let name = arch.split(':').next()?.strip_prefix("gfx")?;
    if name.len() < 3 {
        return None;
    }
    if let Some(generic) = name.strip_suffix("-generic") {
        let mut parts = generic.splitn(2, '-');
        let major = i32::from_str_radix(parts.next()?, 16).ok()?;
        let minor = match parts.next() {
            Some(m) => i32::from_str_radix(m, 16).ok()? * 0x10,
            None => 0,
        };
        return Some(OFFSET_AMD + major * 0x100 + minor);
    }
    let (major_str, minor_str) = name.split_at(name.len() - 2);
    let minor = i32::from_str_radix(minor_str, 16).ok()?;
    let major = i32::from_str_radix(major_str, 16).ok()?;
    Some(OFFSET_AMD + major * 0x100 + minor)
}

/// The numeric part of a `gfx` target, e.g. `gfx1101:xnack-` -> 1101, or a
/// ROCm generic target name synthesized into the same decimal scheme, e.g.
/// `gfx12-generic` -> 1200.
///
/// `None` for anything that is neither: `gfx90a` must not read as "90" and
/// land in a range it was never meant to match.
fn arch_number(arch: &str) -> Option<u32> {
    let name = arch.split(':').next()?.strip_prefix("gfx")?;
    if let Ok(n) = name.parse::<u32>() {
        return Some(n);
    }
    let generic = name.strip_suffix("-generic")?;
    let mut parts = generic.splitn(2, '-');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = match parts.next() {
        Some(m) => m.parse().ok()?,
        None => 0,
    };
    Some(major * 100 + minor * 10)
}

fn gcn_arch_name(ordinal: usize) -> Result<String, KernelError> {
    use rocm_rs::hip::ffi;

    let device = i32::try_from(ordinal)
        .map_err(|_| KernelError::Rocm(format!("device ordinal {ordinal} is out of range")))?;

    // SAFETY: `hipDeviceProp_tR0600` is a plain-data struct, so an all-zero
    // value is a valid one to hand to the driver. The call only writes into it,
    // and its status is checked before any field is read.
    let (status, props) = unsafe {
        let mut props = std::mem::zeroed::<ffi::hipDeviceProp_tR0600>();
        let status = ffi::hipGetDevicePropertiesR0600(&mut props, device);
        (status, props)
    };

    if status != ffi::hipError_t_hipSuccess {
        return Err(KernelError::Rocm(format!(
            "hipGetDeviceProperties failed for device {ordinal} (HIP error {status}); \
             set CANDLE_ROCM_ARCH to build kernels anyway"
        )));
    }

    // The driver is not required to NUL-terminate a full 256-byte name, so stop
    // at the first NUL rather than handing the array to `CStr`.
    let name: Vec<u8> = props
        .gcnArchName
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    Ok(String::from_utf8_lossy(&name).trim().to_string())
}

/// The ROCm toolchain identity, as `(directory tag, full version string)`.
///
/// The tag stays short so cache directories remain readable. The full
/// `hipcc --version` output is what goes into the cache key: a patch-level LLVM
/// bump inside one minor release can change the code-object ABI, so keying on
/// "7.2" alone would reuse objects the new loader may reject.
///
/// Memoised, because the answer costs a process spawn — around 80 ms of the
/// ~90 ms every `RocmDevice::new` used to take — and cannot change under a
/// running process.
pub(crate) fn rocm_version() -> Result<(String, String), KernelError> {
    if let Ok(version) = std::env::var("CANDLE_ROCM_VERSION") {
        return Ok((version.clone(), version));
    }

    static VERSION: OnceLock<Result<(String, String), String>> = OnceLock::new();
    VERSION
        .get_or_init(|| probe_rocm_version().map_err(|e| e.to_string()))
        .clone()
        .map_err(KernelError::Compilation)
}

fn probe_rocm_version() -> Result<(String, String), KernelError> {
    let output = Command::new("hipcc")
        .arg("--version")
        .output()
        .map_err(|e| KernelError::Compilation(format!("could not run hipcc: {e}")))?;

    let full = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let tag = full.lines().find_map(short_version).ok_or_else(|| {
        KernelError::Compilation("could not parse the HIP version from hipcc --version".to_string())
    })?;
    Ok((tag, full))
}

/// "HIP version: 7.2.4-53211" -> "7.2".
fn short_version(line: &str) -> Option<String> {
    let version = line.strip_prefix("HIP version:")?.trim();
    let short = version.split('.').take(2).collect::<Vec<_>>().join(".");
    Some(short.split('-').next().unwrap_or(&short).to_string())
}

#[cfg(test)]
mod tests {
    use super::{amd_cc, has_wmma, mmq_tiles, rdna_define, short_version, MmqTiles};

    /// `GGML_CUDA_CC_OFFSET_AMD` from `mmq_common.cuh`, so these tests can
    /// check `amd_cc` against the real constants it has to agree with
    /// rather than hand-picked numbers.
    const OFFSET_AMD: i32 = 0x0100_0000;

    #[test]
    fn amd_cc_matches_the_ggml_cuda_cc_constants() {
        // GGML_CUDA_CC_RDNA2 = OFFSET_AMD + 0x1030
        assert_eq!(amd_cc("gfx1030"), Some(OFFSET_AMD + 0x1030));
        // GGML_CUDA_CC_RDNA3 = OFFSET_AMD + 0x1100
        assert_eq!(amd_cc("gfx1100"), Some(OFFSET_AMD + 0x1100));
        assert_eq!(amd_cc("gfx1101"), Some(OFFSET_AMD + 0x1101));
        // GGML_CUDA_CC_RDNA4 = OFFSET_AMD + 0x1200; gfx1201's encoding must
        // land >= this, which a decimal parse of "1201" would not.
        let rdna4 = OFFSET_AMD + 0x1200;
        assert!(amd_cc("gfx1201").unwrap() >= rdna4);
        assert_eq!(amd_cc("gfx1201"), Some(OFFSET_AMD + 0x1201));
        // A target feature suffix must not change the encoding.
        assert_eq!(amd_cc("gfx1101:xnack-"), Some(OFFSET_AMD + 0x1101));
    }

    #[test]
    fn amd_cc_rejects_short_or_empty_names() {
        assert_eq!(amd_cc("gfx10"), None);
        assert_eq!(amd_cc("gfx"), None);
        assert_eq!(amd_cc("not-a-gfx-target"), None);
    }

    #[test]
    fn selects_the_rdna_mmq_geometry_per_architecture() {
        assert_eq!(rdna_define("gfx1101"), Some("RDNA3"));
        assert_eq!(rdna_define("gfx1101:xnack-"), Some("RDNA3"));
        assert_eq!(rdna_define("gfx1151"), Some("RDNA3"));
        assert_eq!(rdna_define("gfx1201"), Some("RDNA4"));
        assert_eq!(rdna_define("gfx1030"), Some("RDNA2"));
        // RDNA1 and the CDNA/Vega parts keep the kernel's default set. `gfx90a`
        // must not be read as architecture 90.
        assert_eq!(rdna_define("gfx1010"), None);
        assert_eq!(rdna_define("gfx90a:sramecc+:xnack-"), None);
        assert_eq!(rdna_define("gfx942"), None);
        assert_eq!(rdna_define("not-a-gfx-target"), None);
    }

    #[test]
    fn the_tile_set_follows_the_define() {
        assert_eq!(mmq_tiles("gfx1101"), MmqTiles::Rdna2);
        assert_eq!(mmq_tiles("gfx1201"), MmqTiles::Rdna2);
        assert_eq!(mmq_tiles("gfx1030"), MmqTiles::Rdna2);
        assert_eq!(mmq_tiles("gfx90a"), MmqTiles::Ampere);
    }

    #[test]
    fn wmma_is_rdna3_and_rdna4_only() {
        assert!(has_wmma("gfx1101"));
        assert!(has_wmma("gfx1201"));
        assert!(!has_wmma("gfx1030"));
        assert!(!has_wmma("gfx90a"));
        assert!(!has_wmma("gfx942"));
    }

    #[test]
    fn arch_number_handles_generic_targets() {
        assert_eq!(rdna_define("gfx12-generic"), Some("RDNA4"));
        assert_eq!(rdna_define("gfx11-generic"), Some("RDNA3"));
        assert_eq!(rdna_define("gfx10-3-generic"), Some("RDNA2"));
        // gfx9-generic -> 900, outside every RDNA range.
        assert_eq!(rdna_define("gfx9-generic"), None);
    }

    #[test]
    fn has_wmma_accepts_generic_targets() {
        assert!(has_wmma("gfx12-generic"));
        assert!(has_wmma("gfx11-generic"));
        assert!(!has_wmma("gfx10-3-generic"));
        assert!(!has_wmma("gfx9-generic"));
    }

    #[test]
    fn amd_cc_handles_generic_targets() {
        assert_eq!(amd_cc("gfx12-generic"), Some(OFFSET_AMD + 0x1200));
        assert_eq!(amd_cc("gfx11-generic"), Some(OFFSET_AMD + 0x1100));
        assert_eq!(amd_cc("gfx10-3-generic"), Some(OFFSET_AMD + 0x1030));
    }

    #[test]
    fn parses_the_hip_version_line() {
        assert_eq!(
            short_version("HIP version: 7.2.4-53211"),
            Some("7.2".to_string())
        );
        assert_eq!(short_version("HIP version: 6.2"), Some("6.2".to_string()));
        assert_eq!(short_version("clang version 20.0.0"), None);
    }
}

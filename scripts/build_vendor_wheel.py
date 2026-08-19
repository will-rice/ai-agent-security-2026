"""Build the vendored aicomp-sdk wheel DETERMINISTICALLY from an extracted bundle.

The SDK wheel is gitignored (competition redistribution), so CI rebuilds it from the
Kaggle bundle on every run and ``uv.lock`` pins its hash. A non-deterministic zip (file
mtimes, zlib-version-dependent DEFLATE) makes that hash differ between the local build
and CI -> ``uv sync`` fails with "Hash mismatch for aicomp-sdk". This writes a
byte-identical wheel regardless of environment: fixed zip timestamps and ``ZIP_STORED``
(no compression, so no zlib-version drift). Local build and CI build produce the SAME
bytes -> the same hash -> ``uv.lock`` matches.

Both the local re-lock and the CI ``Rebuild vendored SDK wheel`` step call this, so
there is one source of truth for the wheel bytes.

Usage:
    uv run python scripts/build_vendor_wheel.py <bundle_dir> <out_whl>

``bundle_dir`` contains ``aicomp_sdk/`` and ``aicomp_sdk-3.1.2.dist-info/``.
"""

import sys
import zipfile
from pathlib import Path

_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)  # zip epoch; deterministic across environments
_SUBDIRS = ("aicomp_sdk", "aicomp_sdk-3.1.2.dist-info")


def build_wheel(bundle_dir: Path, out_whl: Path) -> Path:
    """Write a byte-deterministic wheel of ``bundle_dir`` to ``out_whl``.

    Args:
        bundle_dir: Directory holding the SDK package + dist-info.
        out_whl: Destination wheel path.

    Returns:
        The written wheel path.
    """
    members = sorted(
        (p, p.relative_to(bundle_dir).as_posix())
        for sub in _SUBDIRS
        for p in (bundle_dir / sub).rglob("*")
        if p.is_file()
    )
    with zipfile.ZipFile(out_whl, "w", zipfile.ZIP_STORED) as z:
        for path, arcname in members:
            info = zipfile.ZipInfo(arcname, date_time=_FIXED_DATE_TIME)
            info.external_attr = 0o644 << 16  # stable file mode
            z.writestr(info, path.read_bytes())
    return out_whl


def main() -> None:
    """Build the wheel from the two positional args."""
    bundle_dir = Path(sys.argv[1])
    out_whl = Path(sys.argv[2])
    build_wheel(bundle_dir, out_whl)
    print(f"wrote {out_whl} ({out_whl.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

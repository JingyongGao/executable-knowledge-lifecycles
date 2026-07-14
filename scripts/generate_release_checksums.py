"""Write and verify SHA-256 checksums for public release assets."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "SHA256SUMS"
ASSETS = (
    Path("paper/main.pdf"),
    Path("artifacts/v0.2.6-exp.1-final-evidence.zip"),
    Path("artifacts/release_manifest.json"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    missing = [path.as_posix() for path in ASSETS if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"release assets missing: {', '.join(missing)}")
    lines = [f"{sha256(ROOT / path)}  {path.as_posix()}" for path in ASSETS]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()

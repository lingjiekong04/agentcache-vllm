"""Copy same-version binary artifacts into the isolated source tree only."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "v1/core/block_pool.py": (
        "82076397ed6eb65127cc8cca9ee8fd669fe39ca8e7d6ee87381e2771782a7748"
    ),
    "v1/core/kv_cache_utils.py": (
        "82857a94f1b54748287acabd3c3dd8791cbb374a300cb961bab8cc0213c61e63"
    ),
    "v1/core/kv_cache_manager.py": (
        "577529cb7ed73e0b882e969f1a355fcca734d621f6b085532d65b6e4ef5e9e46"
    ),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources():
    manifest = json.loads((ROOT / "agentcache/source_manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Engineering source changed: {relative}")


def installed_root():
    distribution = importlib.metadata.distribution("vllm")
    if distribution.version != "0.8.3":
        raise ValueError("Only installed vLLM 0.8.3 extensions are supported")
    root = Path(distribution.locate_file("vllm")).resolve()
    if root == ROOT / "vllm":
        raise ValueError("Expected the old installed package, not this tree")
    for file, expected in EXPECTED.items():
        if digest(root / file) != expected:
            raise ValueError(f"Installed baseline differs: {file}")
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux" or sys.version_info[:2] != (3, 11):
        raise ValueError("Runtime preparation requires the existing Linux Python 3.11")
    verify_sources()
    old = installed_root()
    manifest_path = ROOT / "agentcache/runtime_manifest.json"
    if args.verify:
        manifest = json.loads(manifest_path.read_text())
        if manifest["installed_root"] != str(old):
            raise ValueError("Runtime location changed")
        for item in manifest["artifacts"]:
            for root in (old, ROOT / "vllm"):
                if digest(root / item["path"]) != item["sha256"]:
                    raise ValueError(f"Binary artifact changed: {item['path']}")
        import vllm
        from vllm.v1.core import block_pool, gpu_eviction_policy
        from vllm import _C

        for module in (vllm, block_pool, gpu_eviction_policy, _C):
            if not Path(module.__file__).resolve().is_relative_to(ROOT / "vllm"):
                raise ValueError(f"Non-engineering import: {module.__file__}")
        print(
            json.dumps(
                {
                    "runtime_verified": True,
                    "python": sys.executable,
                    "imports": [
                        m.__file__ for m in (vllm, block_pool, gpu_eviction_policy, _C)
                    ],
                }
            )
        )
        return
    if manifest_path.exists():
        raise FileExistsError("Runtime already prepared; use --verify")
    relatives = sorted(p.relative_to(old) for p in old.rglob("*.so"))
    relatives.extend(
        Path(p)
        for p in (
            "_version.py",
            "vllm_flash_attn/__init__.py",
            "vllm_flash_attn/flash_attn_interface.py",
        )
    )
    required = ("_C.abi3.so", "_moe_C.abi3.so", "vllm_flash_attn/_vllm_fa2_C.abi3.so")
    if any(Path(p) not in relatives for p in required):
        raise ValueError("Required same-version A100 extensions missing")
    copies = []
    for relative in relatives:
        source, target = old / relative, ROOT / "vllm" / relative
        if not source.is_file() or target.exists():
            raise ValueError(f"Missing source or existing target: {relative}")
        copies.append((relative, source, target))
    # Preflight all paths before copying. Partial preparation is preserved on failure.
    for relative, source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as inp, target.open("xb") as out:
            shutil.copyfileobj(inp, out)
    with manifest_path.open("x", encoding="utf-8") as out:
        json.dump(
            {
                "installed_root": str(old),
                "python": sys.executable,
                "artifacts": [
                    {"path": p.as_posix(), "sha256": digest(src)}
                    for p, src, _ in copies
                ],
            },
            out,
            indent=2,
        )
    print("Prepared isolated source artifacts; old site-packages unchanged")


if __name__ == "__main__":
    main()

"""Rebuild the L1 source baseline from Git layers and the existing Drive bundle.

Copies only the declared dependency closure into a fresh independent directory.
No pilot scene, episode initializer, source edit, symlink or GPU is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "resources/house_scene_l1/manifest.json"
ENTRY = "demo_sim/scene_room_baseline.usda"


def safe_child(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError("Unsafe relative asset path: " + str(relative))
    result = root / rel
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("Asset path escapes source directory")
    return result


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized(layer):
    from pxr import Sdf, UsdUtils
    clone = Sdf.Layer.CreateAnonymous(".usda")
    clone.TransferContent(layer)
    UsdUtils.ModifyAssetPaths(clone, lambda v: "__ASSET__" if v else v)
    return hashlib.sha256(clone.ExportToString().encode()).hexdigest()


def validate_receipt(base):
    base = Path(base).resolve(strict=True)
    root = base.parent.parent
    receipt = json.loads((root / "bundle.json").read_text())
    if (receipt.get("schema") != "arena.house_scene_l1.bundle.v1"
            or root / ENTRY != base or receipt.get("entrypoint") != ENTRY
            or receipt.get("manifest_sha256") != digest(MANIFEST)):
        raise ValueError("Install the clean L1 baseline; old pilot entrypoints are not supported")
    for row in receipt["outputs"]:
        path = safe_child(root, row["path"])
        if digest(path) != row["sha256"]:
            raise ValueError("Installed L1 dependency changed: " + row["path"])
    return {k: v for k, v in receipt.items() if k != "outputs"}


def install(bundle, output, *, manifest=MANIFEST, repo=ROOT):
    from pxr import Sdf, Usd, UsdUtils
    bundle, output = Path(bundle).resolve(strict=True), Path(output).resolve()
    data = json.loads(Path(manifest).read_text())
    if data["schema"] != "arena.house_scene_l1.bundle.v1":
        raise ValueError("Unknown baseline manifest")
    if output.exists() or output.is_relative_to(bundle) or bundle.is_relative_to(output):
        raise ValueError("Use a fresh, separate output; no existing assets are overwritten")
    sources, destinations, input_stats = [], set(), {}
    for row in data["files"]:
        origin = row["source"]
        if origin["kind"] not in ("bundle", "repo"):
            raise ValueError("Unknown source kind")
        source = safe_child(bundle if origin["kind"] == "bundle" else Path(repo), origin["path"])
        destination = safe_child(output, row["destination"])
        if destination in destinations:
            raise ValueError("Duplicate output path")
        destinations.add(destination)
        if source.stat().st_size != origin["bytes"] or digest(source) != origin["sha256"]:
            raise ValueError("Source differs from the imported L1 dependency: " + str(source))
        sources.append((row, source, destination))
        input_stats[source] = (source.stat().st_size, source.stat().st_mtime_ns)
    # 10 GiB remains free even when reflinks are unavailable.
    disk = next(p for p in (output.parent, *output.parents) if p.exists())
    needed = sum(row["source"]["bytes"] for row, _, _ in sources)
    if shutil.disk_usage(disk).free < needed + 10 * 1024**3:
        raise ValueError("Insufficient free space including the 10 GiB reserve")
    output.mkdir(parents=True, exist_ok=False)
    for row, source, target in sources:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not row["layer"]:
            # An independent copy; never share mutable inodes with the old bundle.
            subprocess.run(["cp", "--reflink=auto", "--", str(source), str(target)], check=True)
            continue
        original = Sdf.Layer.FindOrOpen(str(source))
        if normalized(original) != row["normalized_sha256"]:
            raise ValueError("Non-asset USD content differs from L1: " + row["destination"])
        layer = Sdf.Layer.CreateNew(str(target))
        layer.TransferContent(original)
        mapping = row["asset_paths"]
        def rewrite(value):
            if not value:
                return value
            if value not in mapping:
                raise ValueError("Unrecorded USD asset path: " + value)
            return mapping[value]
        UsdUtils.ModifyAssetPaths(layer, rewrite)
        layer.Save()
        if normalized(layer) != row["normalized_sha256"]:
            raise ValueError("USD content changed during path localization")
    stage = Usd.Stage.Open(str(output / data["entrypoint"]))
    if not stage or stage.GetTimeCodesPerSecond() != 60:
        raise ValueError("Expected the L1 60-time-code source baseline")
    foreign = [l.realPath for l in stage.GetUsedLayers()
               if l.realPath and not Path(l.realPath.split("[", 1)[0]).is_relative_to(output)]
    if foreign:
        raise ValueError("Clean baseline still loads external layers: " + repr(foreign))
    if any("pilot" in l.realPath.lower() for l in stage.GetUsedLayers()):
        raise ValueError("Pilot layer must not be in the clean baseline")
    if any((p.stat().st_size, p.stat().st_mtime_ns) != stat for p, stat in input_stats.items()):
        raise ValueError("Source changed during installation")
    receipt = dict(schema=data["schema"], source_commit=data["source_commit"],
                   entrypoint=data["entrypoint"], files=len(sources),
                   active_prims=sum(1 for _ in stage.Traverse()),
                   source_assets_unchanged=True, symlinks_used=False,
                   manifest_sha256=digest(Path(manifest)),
                   outputs=[dict(path=row["destination"], sha256=digest(target))
                            for row, _, target in sources],
                   known_baseline_unresolved=data["known_baseline_unresolved"])
    (output / "bundle.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, default=ROOT / "assets/house_scene_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/house_scene_l1")
    args = parser.parse_args()
    print(json.dumps(install(args.source_bundle, args.output), indent=2))


if __name__ == "__main__":
    main()

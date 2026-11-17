"""Validated, persistent simulation datasets with explicit regeneration."""
import hashlib
import json
from pathlib import Path
import numpy as np
import heat


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_dataset(output, name, cases, regenerate=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}_frames.npy"
    manifest_path = output / f"{name}_dataset.json"
    signature = {"version": 1, "simulation_sha256": digest(Path(heat.__file__)),
                 "cases_sha256": hashlib.sha256(cases.tobytes()).hexdigest(),
                 "shape": [len(cases), heat.STEPS + 1, heat.N, heat.N], "dtype": "float32"}
    frames = None
    reason = "explicit regeneration" if regenerate else "missing or incompatible cache"
    if path.exists() and not regenerate:
        candidate = None
        try:
            candidate = np.load(path, mmap_mode="r", allow_pickle=False)
            valid_shape = list(candidate.shape) == signature["shape"] and candidate.dtype == np.float32
            if valid_shape and manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("signature") == signature and manifest.get("file_sha256") == digest(path):
                    frames = candidate
            elif valid_shape:
                # One-time migration: verify every existing simulation before trusting it.
                print(f"Verifying legacy {name} dataset once...", flush=True)
                if all(np.array_equal(candidate[i], heat.simulate(*case).astype(np.float32))
                       for i, case in enumerate(cases)):
                    frames = candidate
            if frames is None:
                candidate._mmap.close()
        except (ValueError, OSError, EOFError, TypeError):
            if candidate is not None:
                candidate._mmap.close()
            frames = None
    if frames is None:
        print(f"Generating {name} dataset: {reason}", flush=True)
        temporary = output / f"{name}_frames.pending.npy"
        values = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=tuple(signature["shape"]))
        for index, case in enumerate(cases):
            values[index] = heat.simulate(*case)
        values.flush()
        values._mmap.close()
        temporary.replace(path)
        frames = np.load(path, mmap_mode="r", allow_pickle=False)
        status = "generated"
    else:
        status = "reused"
    manifest = {"signature": signature, "file_sha256": digest(path)}
    try:
        existing_manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        existing_manifest = None
    if existing_manifest != manifest:
        temporary_manifest = manifest_path.with_suffix(".pending.json")
        temporary_manifest.write_text(json.dumps(manifest, indent=2))
        temporary_manifest.replace(manifest_path)
    print(f"{name}: {status} {len(cases)} configurations", flush=True)
    return frames, status

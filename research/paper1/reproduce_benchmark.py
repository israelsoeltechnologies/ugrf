#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path
import urllib.request

import numpy as np

from ugrf import UtilityGatedRenewal

CORE_VERSION = "UGRF_FROZEN_CORE_2026-09-27_V3_EXACT"
HORIZON = 6

SPECS = {
    "CarParts": {"utility_origins": (30, 33, 36, 39), "final_origin": 45, "shape": (2674, 51)},
    "AUTO": {"utility_origins": (12,), "final_origin": 18, "shape": (3000, 24)},
    "RAF": {"utility_origins": (48, 54, 60, 66, 72), "final_origin": 78, "shape": (5000, 84)},
}

TARGETS = {
    "CarParts": 0.9918971759288252,
    "AUTO": 0.9761486034991342,
    "RAF": 0.8905672729887494,
}

AUTO_URL = "https://raw.githubusercontent.com/canerturkmen/gluon-ts/intermittent-datasets/datasets/intermittent_auto/test/data.json"
RAF_URL = "https://raw.githubusercontent.com/canerturkmen/gluon-ts/intermittent-datasets/datasets/intermittent_raf/test/data.json"
CARPARTS_BLOB_API = "https://api.github.com/repos/brunoklein99/deepar/git/blobs/821266e21541c59ceb00166753486ca5fdc56674"


def download_url(url: str, path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "ugrf/0.1.0"})
    with urllib.request.urlopen(req) as response:
        path.write_bytes(response.read())


def download_carparts(path: Path) -> None:
    req = urllib.request.Request(
        CARPARTS_BLOB_API,
        headers={"User-Agent": "ugrf/0.1.0", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("encoding") != "base64":
        raise RuntimeError("Expected base64-encoded GitHub blob")
    path.write_bytes(base64.b64decode(payload["content"]))


def ensure_data(data_dir: Path, download: bool) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "CarParts": data_dir / "carparts.csv",
        "AUTO": data_dir / "auto.json",
        "RAF": data_dir / "raf.json",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing and not download:
        raise FileNotFoundError("Missing: " + ", ".join(missing) + ". Re-run with --download.")
    if not paths["AUTO"].exists():
        download_url(AUTO_URL, paths["AUTO"])
    if not paths["RAF"].exists():
        download_url(RAF_URL, paths["RAF"])
    if not paths["CarParts"].exists():
        download_carparts(paths["CarParts"])
    return paths


def load_gluonts_json(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8").strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = [payload]
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    return np.vstack([np.asarray(rec["target"], dtype=float) for rec in payload])


def load_carparts(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            vals = []
            for value in row[1:]:
                try:
                    x = float(value)
                    vals.append(x if np.isfinite(x) and x >= 0 else 0.0)
                except (TypeError, ValueError):
                    vals.append(0.0)
            rows.append(vals)
    return np.asarray(rows, dtype=float)


def load_all(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    data = {
        "CarParts": load_carparts(paths["CarParts"]),
        "AUTO": load_gluonts_json(paths["AUTO"]),
        "RAF": load_gluonts_json(paths["RAF"]),
    }
    for name, arr in data.items():
        arr[~np.isfinite(arr)] = 0.0
        arr[arr < 0] = 0.0
        expected = SPECS[name]["shape"]
        if arr.shape != expected:
            raise ValueError(f"{name}: expected {expected}, got {arr.shape}")
    return data


def ratio_for(name: str, panel: np.ndarray) -> float:
    spec = SPECS[name]
    model = UtilityGatedRenewal(horizon=HORIZON).fit(
        panel,
        utility_origins=spec["utility_origins"],
        final_origin=spec["final_origin"],
    )
    actual = panel[:, spec["final_origin"] : spec["final_origin"] + HORIZON].sum(axis=1)
    valid = np.any(panel[:, : spec["final_origin"]] > 0, axis=1) & np.isfinite(model.renewal_forecast_)
    err_tsb = np.abs(actual[valid] - model.tsb_forecast_[valid]).sum()
    err_ugrf = np.abs(actual[valid] - model.forecast_[valid]).sum()
    return float(err_ugrf / err_tsb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    data = load_all(ensure_data(args.data_dir, args.download))
    print(f"core={CORE_VERSION}")
    for name in ("CarParts", "AUTO", "RAF"):
        got = ratio_for(name, data[name])
        target = TARGETS[name]
        ok = np.isclose(got, target, rtol=0.0, atol=5e-6)
        print(f"{name:9s} got={got:.12f} target={target:.12f} passed={ok}")
        if not ok:
            raise RuntimeError(f"Frozen reproduction failed for {name}")


if __name__ == "__main__":
    main()

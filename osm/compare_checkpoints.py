# Compares EMRDM prediction metrics with optional OSM-aware subsets.

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


PATCH_PATTERN = re.compile(
    r"ROIs\d+_(?P<season>spring|summer|fall|winter)_s2(?:_cloudy)?_"
    r"(?P<roi>\d+)_p(?P<patch>\d+)\.(?:tif|png)$"
)

LOWER_IS_BETTER = ("RMSE", "MAE", "SAM")
HIGHER_IS_BETTER = ("PSNR", "SSIM")
DEFAULT_METRICS = LOWER_IS_BETTER + HIGHER_IS_BETTER


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two EMRDM metrics.csv files and optionally report subsets "
            "where OSM road/water/building masks are present."
        )
    )
    parser.add_argument("--baseline", required=True, help="Baseline metrics.csv path.")
    parser.add_argument("--candidate", required=True, help="Candidate metrics.csv path.")
    parser.add_argument(
        "--osm-root",
        default=None,
        help="Optional OSM output root containing <season>_s2_<roi>/ masks.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=["road", "water", "building"],
        help="OSM channel names in masks.npy order.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Summary is always printed.",
    )
    return parser.parse_args()


def load_metrics(path, label):
    df = pd.read_csv(path)
    unnamed = [col for col in df.columns if col.startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    if "image_path" not in df.columns:
        raise ValueError(f"{path} does not contain an image_path column.")

    df["image_path"] = df["image_path"].map(lambda p: os.path.basename(str(p)))
    keep = ["image_path"] + [m for m in DEFAULT_METRICS if m in df.columns]
    df = df[keep].copy()

    for metric in keep:
        if metric == "image_path":
            continue
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    if df["image_path"].duplicated().any():
        dupes = df.loc[df["image_path"].duplicated(), "image_path"].head().tolist()
        raise ValueError(f"{label} has duplicate image_path entries, for example {dupes}.")

    return df


def parse_patch_name(image_path):
    match = PATCH_PATTERN.match(os.path.basename(str(image_path)))
    if match is None:
        return None
    return {
        "season": match.group("season"),
        "roi": match.group("roi"),
        "patch": match.group("patch"),
    }


class OSMPresenceReader:
    def __init__(self, root, types, height=256, width=256):
        self.root = Path(root)
        self.types = list(types)
        self.height = int(height)
        self.width = int(width)
        self._cache = {}

    def _load_roi(self, season, roi):
        key = (season, roi)
        if key in self._cache:
            return self._cache[key]

        roi_dir = self.root / f"{season}_s2_{roi}"
        manifest_path = roi_dir / "manifest.json"
        masks_path = roi_dir / "masks.npy"
        if not manifest_path.is_file() or not masks_path.is_file():
            self._cache[key] = None
            return None

        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        entry = {
            "manifest": manifest,
            "masks": np.load(masks_path, mmap_mode="r"),
            "patch_id_to_index": manifest.get("patch_id_to_index", {}),
            "bitorder": manifest.get("bitorder", "little"),
        }
        self._cache[key] = entry
        return entry

    def _patch_index(self, entry, patch):
        patch_id_to_index = entry["patch_id_to_index"]
        for candidate in (patch, str(patch), f"p{patch}"):
            if candidate in patch_id_to_index:
                return int(patch_id_to_index[candidate])

        patch_ids = entry["manifest"].get("patch_ids", [])
        for idx, patch_id in enumerate(patch_ids):
            if str(patch_id) in {str(patch), f"p{patch}"}:
                return idx
        return None

    def get_presence(self, image_path):
        parsed = parse_patch_name(image_path)
        if parsed is None:
            return {name: False for name in self.types}

        entry = self._load_roi(parsed["season"], parsed["roi"])
        if entry is None:
            return {name: False for name in self.types}

        patch_index = self._patch_index(entry, parsed["patch"])
        if patch_index is None:
            return {name: False for name in self.types}

        packed = np.asarray(entry["masks"][patch_index])
        unpacked = np.unpackbits(
            packed,
            axis=-1,
            count=self.height * self.width,
            bitorder=entry["bitorder"],
        )
        presence = {}
        for channel, name in enumerate(self.types):
            presence[name] = channel < unpacked.shape[0] and bool(unpacked[channel].any())
        return presence


def add_osm_presence(df, osm_root, types):
    reader = OSMPresenceReader(osm_root, types)
    rows = [reader.get_presence(path) for path in df["image_path"]]
    presence = pd.DataFrame(rows)
    for name in types:
        df[f"osm_{name}"] = presence[name].fillna(False).astype(bool)
    df["osm_any"] = df[[f"osm_{name}" for name in types]].any(axis=1)
    return df


def mean_or_none(series):
    value = series.mean()
    if pd.isna(value):
        return None
    return float(value)


def summarize_subset(df, name, mask, metrics):
    subset = df.loc[mask]
    summary = {"subset": name, "n": int(len(subset))}
    for metric in metrics:
        base_col = f"{metric}_baseline"
        cand_col = f"{metric}_candidate"
        if base_col not in subset or cand_col not in subset:
            continue

        base_mean = mean_or_none(subset[base_col])
        cand_mean = mean_or_none(subset[cand_col])
        delta = None
        relative = None
        improved = None
        if base_mean is not None and cand_mean is not None:
            delta = cand_mean - base_mean
            if base_mean != 0:
                relative = delta / abs(base_mean)
            if metric in LOWER_IS_BETTER:
                improved = cand_mean < base_mean
            elif metric in HIGHER_IS_BETTER:
                improved = cand_mean > base_mean

        summary[metric] = {
            "baseline": base_mean,
            "candidate": cand_mean,
            "delta": delta,
            "relative_delta": relative,
            "improved": improved,
        }
    return summary


def compare(baseline_path, candidate_path, osm_root=None, types=None):
    types = types or ["road", "water", "building"]
    baseline = load_metrics(baseline_path, "baseline")
    candidate = load_metrics(candidate_path, "candidate")

    metrics = [metric for metric in DEFAULT_METRICS if metric in baseline and metric in candidate]
    merged = baseline.merge(
        candidate,
        on="image_path",
        suffixes=("_baseline", "_candidate"),
        how="inner",
    )
    if len(merged) == 0:
        raise ValueError("No matching image_path rows found between the two metrics files.")

    subsets = [("all", pd.Series(True, index=merged.index))]
    if osm_root is not None:
        merged = add_osm_presence(merged, osm_root, types)
        subsets.append(("osm_any", merged["osm_any"]))
        subsets.append(("osm_empty", ~merged["osm_any"]))
        for name in types:
            subsets.append((name, merged[f"osm_{name}"]))

    return {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "osm_root": str(osm_root) if osm_root is not None else None,
        "n_baseline": int(len(baseline)),
        "n_candidate": int(len(candidate)),
        "n_matched": int(len(merged)),
        "metrics": metrics,
        "subsets": [
            summarize_subset(merged, subset_name, subset_mask, metrics)
            for subset_name, subset_mask in subsets
        ],
    }


def print_summary(summary):
    print(f"Matched samples: {summary['n_matched']}")
    print(f"Baseline: {summary['baseline']}")
    print(f"Candidate: {summary['candidate']}")
    if summary["osm_root"] is not None:
        print(f"OSM root: {summary['osm_root']}")
    print()

    for subset in summary["subsets"]:
        print(f"[{subset['subset']}] n={subset['n']}")
        for metric in summary["metrics"]:
            values = subset.get(metric)
            if not values:
                continue
            base = values["baseline"]
            cand = values["candidate"]
            delta = values["delta"]
            improved = values["improved"]
            if base is None or cand is None:
                continue
            sign = "+" if delta is not None and delta >= 0 else ""
            mark = "improved" if improved else "worse/same"
            print(
                f"  {metric}: baseline={base:.6f} "
                f"candidate={cand:.6f} delta={sign}{delta:.6f} ({mark})"
            )
        print()


def main():
    args = parse_args()
    summary = compare(
        baseline_path=Path(args.baseline),
        candidate_path=Path(args.candidate),
        osm_root=Path(args.osm_root) if args.osm_root else None,
        types=args.types,
    )
    print_summary(summary)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Wrote JSON summary: {output}")


if __name__ == "__main__":
    main()

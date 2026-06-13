# OSM conditioning extension for SEN12MS-CR.
# Original Sentinel loaders remain unchanged in sentinel.py.

import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import torch as th

from sgm.data.sentinel.sentinel import SEN12MSCRInterface

SEN12MSCR_OSM_PATTERN = re.compile(
    r"ROIs\d+_(?P<season>spring|summer|fall|winter)_s2(?:_cloudy)?_"
    r"(?P<roi>\d+)_p(?P<patch>\d+)\.tif$"
)


def parse_sen12mscr_s2_patch(path):
    if path is None:
        return None
    match = SEN12MSCR_OSM_PATTERN.match(os.path.basename(path))
    if match is None:
        return None
    return {
        "season": match.group("season"),
        "roi": match.group("roi"),
        "patch": match.group("patch"),
    }


class OSMMaskReader:
    def __init__(self, root, channels=3, height=256, width=256):
        self.root = Path(root)
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self._cache = {}

    def _roi_dir(self, season, roi):
        return self.root / f"{season}_s2_{roi}"

    def _load_roi(self, season, roi):
        key = (season, roi)
        if key in self._cache:
            return self._cache[key]

        roi_dir = self._roi_dir(season, roi)
        manifest_path = roi_dir / "manifest.json"
        masks_path = roi_dir / "masks.npy"
        if not manifest_path.is_file() or not masks_path.is_file():
            self._cache[key] = None
            return None

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        masks = np.load(masks_path, mmap_mode="r")
        entry = {
            "manifest": manifest,
            "masks": masks,
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

    def zeros(self):
        return th.zeros((self.channels, self.height, self.width), dtype=th.float32)

    def get(self, image_path):
        parsed = parse_sen12mscr_s2_patch(image_path)
        if parsed is None:
            return self.zeros()

        entry = self._load_roi(parsed["season"], parsed["roi"])
        if entry is None:
            return self.zeros()

        patch_index = self._patch_index(entry, parsed["patch"])
        if patch_index is None:
            return self.zeros()

        packed = np.asarray(entry["masks"][patch_index])
        unpacked = np.unpackbits(
            packed,
            axis=-1,
            count=self.height * self.width,
            bitorder=entry["bitorder"],
        )
        unpacked = unpacked.reshape(packed.shape[0], self.height, self.width)

        n_unpacked_channels = unpacked.shape[0]
        if n_unpacked_channels != self.channels:
            warnings.warn(
                f"OSM channel count mismatch for {image_path}: "
                f"expected {self.channels}, got {n_unpacked_channels}.",
                stacklevel=2,
            )

        if n_unpacked_channels >= self.channels:
            mask = unpacked[: self.channels]
        else:
            mask = np.zeros(
                (self.channels, self.height, self.width), dtype=unpacked.dtype
            )
            mask[:n_unpacked_channels] = unpacked

        return th.as_tensor(mask, dtype=th.float32)


class SEN12MSCROSMInterface(SEN12MSCRInterface):
    def __init__(
        self,
        root,
        split="all",
        region="all",
        cloud_masks="s2cloudless_mask",
        sample_type="pretrain",
        rescale_method="default",
        rescale=True,
        return_target=True,
        osm_root=None,
        osm_channels=3,
        concat_osm=True,
        osm_key="OSM",
    ):
        super().__init__(
            root=root,
            split=split,
            region=region,
            cloud_masks=cloud_masks,
            sample_type=sample_type,
            rescale_method=rescale_method,
            rescale=rescale,
            return_target=return_target,
        )
        if osm_root is None:
            raise ValueError("SEN12MSCROSMInterface requires osm_root.")

        self.osm_key = osm_key
        self.concat_osm = bool(concat_osm)
        self.osm_store = OSMMaskReader(osm_root, channels=osm_channels)

    def __getitem__(self, pdx):
        sample = super().__getitem__(pdx)
        osm = self.osm_store.get(sample.get("image_path"))
        sample[self.osm_key] = osm

        if self.concat_osm and "S1S2" in sample:
            sample["S1S2"] = th.cat([sample["S1S2"], osm], dim=0)

        return sample
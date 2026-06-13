# Converts a pretrained SEN12MS-CR checkpoint for the OSM-conditioned config.

import argparse
from pathlib import Path

import torch


PATCH_IN_WEIGHT = "model.diffusion_model.patch_in.proj.weight"
EMA_PATCH_IN_WEIGHT = "model_ema.diffusion_modelpatch_inprojweight"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Expand the Sentinel input projection from 28 to 31 channels so a "
            "pretrained checkpoint can initialize the OSM-conditioned model."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the original pretrained .ckpt file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where the converted .ckpt file will be written.",
    )
    parser.add_argument(
        "--old-channels",
        type=int,
        default=28,
        help="Original model input channel count.",
    )
    parser.add_argument(
        "--new-channels",
        type=int,
        default=31,
        help="OSM-conditioned model input channel count.",
    )
    parser.add_argument(
        "--init",
        choices=("zero", "mean"),
        default="zero",
        help="How to initialize the added input-channel weights.",
    )
    return parser.parse_args()


def expand_input_weight(weight, old_channels, new_channels, init):
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D input projection weight, got {weight.shape}.")
    if weight.shape[1] != old_channels:
        raise ValueError(
            f"Expected {old_channels} input channels, got weight shape {weight.shape}."
        )
    if new_channels <= old_channels:
        raise ValueError(
            f"new_channels must be greater than old_channels, got "
            f"{new_channels} <= {old_channels}."
        )

    expanded = weight.new_zeros((weight.shape[0], new_channels))
    expanded[:, :old_channels] = weight

    if init == "mean":
        expanded[:, old_channels:] = weight.mean(dim=1, keepdim=True)

    return expanded


def convert_checkpoint(input_path, output_path, old_channels, new_channels, init):
    checkpoint = torch.load(input_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(f"{input_path} does not contain a 'state_dict' entry.")

    state_dict = checkpoint["state_dict"]
    changed = []
    for key in (PATCH_IN_WEIGHT, EMA_PATCH_IN_WEIGHT):
        if key not in state_dict:
            continue
        old_weight = state_dict[key]
        state_dict[key] = expand_input_weight(
            old_weight,
            old_channels=old_channels,
            new_channels=new_channels,
            init=init,
        )
        changed.append((key, tuple(old_weight.shape), tuple(state_dict[key].shape)))

    if not changed:
        expected = ", ".join((PATCH_IN_WEIGHT, EMA_PATCH_IN_WEIGHT))
        raise KeyError(f"No input projection weights found. Expected one of: {expected}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return changed


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    changed = convert_checkpoint(
        input_path=input_path,
        output_path=output_path,
        old_channels=args.old_channels,
        new_channels=args.new_channels,
        init=args.init,
    )

    print(f"Wrote converted checkpoint: {output_path}")
    for key, old_shape, new_shape in changed:
        print(f"{key}: {old_shape} -> {new_shape}")
    print(f"Added channels initialized with: {args.init}")


if __name__ == "__main__":
    main()
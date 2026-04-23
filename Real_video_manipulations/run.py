"""
run.py — command-line interface for the video augmentation pipeline.

Usage examples:
# Apply a preset to one video
python run.py input.mp4 output.mp4 --preset handheld

# Apply specific augmentations
python run.py input.mp4 output.mp4 --augs camera_shake film_grain compression

# Process a whole directory
python run.py data/real/ data/augmented/ --preset bad_stream --batch

# Lower memory usage (default chunk=64; lower = less RAM)
python run.py input.mp4 output.mp4 --preset handheld --chunk 16

# List all available augmentations and presets, then exit
python run.py --list
"""

import argparse
import sys
from augmentations import AUGMENTATION_REGISTRY, PRESETS
from pipeline import process_video, batch_process


# HELPERS

def print_available():
    print("\n── Augmentations (" + str(len(AUGMENTATION_REGISTRY)) + " total) ──")
    categories = {
        "Lighting"       : ["brightness", "contrast", "gamma", "flicker",
                            "vignette", "color_shift", "exposure_burst"],
        "Shaking"        : ["camera_shake", "rotation_jitter", "zoom_pulse", "earthquake"],
        "Poor connection": ["compression", "packet_loss", "bitrate_drop", "blocking",
                            "interlacing", "pixelation", "network_noise", "horizontal_tearing"],
        "Sensor / optics": ["film_grain", "motion_blur", "lens_distortion", "chromatic_aberration"],
    }
    for cat, names in categories.items():
        print(f"\n  {cat}:")
        for n in names:
            print(f"    {n}")

    print("\n── Presets ──")
    for name, augs in PRESETS.items():
        print(f"  {name:<14}  {augs}")
    print()


# ARGUMENT PARSER

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description="Video augmentation pipeline for synthetic video detector testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py input.mp4 output.mp4 --preset handheld
  python run.py input.mp4 output.mp4 --augs flicker camera_shake film_grain
  python run.py data/real/ data/aug/  --preset bad_stream --batch
  python run.py input.mp4 output.mp4 --preset old_footage --chunk 16
  python run.py --list
        """,
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Input video file (or directory when --batch is used).",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output video file (or directory when --batch is used).",
    )

    # What to apply
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--preset", "-p",
        metavar="NAME",
        choices=list(PRESETS.keys()),
        help="Use a built-in preset. Choices: " + ", ".join(PRESETS.keys()),
    )
    group.add_argument(
        "--augs", "-a",
        nargs="+",
        metavar="AUG",
        help="One or more augmentation names to apply in order.",
    )

    # Mode
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Process all videos in INPUT directory and write to OUTPUT directory.",
    )

    # Memory / performance
    parser.add_argument(
        "--chunk", "-c",
        type=int,
        default=64,
        metavar="N",
        help=(
            "Frames per processing chunk. "
            "Lower = less RAM, same result. Default: 64. "
            "RAM ≈ N × H × W × 3 bytes  (e.g. 64 frames @ 1080p ≈ 400 MB)."
        ),
    )

    # Info
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Print all available augmentation names and presets, then exit.",
    )

    return parser


# MAIN

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print_available()
        sys.exit(0)

    # Validate required positional args
    if not args.input or not args.output:
        parser.print_help()
        print("\nError: INPUT and OUTPUT are required (unless --list is used).")
        sys.exit(1)

    if not args.preset and not args.augs:
        parser.print_help()
        print("\nError: provide --preset NAME or --augs AUG [AUG ...].")
        sys.exit(1)

    # Validate custom augmentation names
    if args.augs:
        bad = [a for a in args.augs if a not in AUGMENTATION_REGISTRY]
        if bad:
            print(f"Error: unknown augmentation(s): {bad}")
            print("Run  python run.py --list  to see valid names.")
            sys.exit(1)

    if args.batch:
        batch_process(
            input_dir=args.input,
            output_dir=args.output,
            augmentations=args.augs,
            preset=args.preset,
            chunk_size=args.chunk,
        )
    else:
        process_video(
            input_path=args.input,
            output_path=args.output,
            augmentations=args.augs,
            preset=args.preset,
            chunk_size=args.chunk,
        )


if __name__ == "__main__":
    main()

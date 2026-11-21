"""Command-line entry point.

Run from the project directory:

    python -m image2mesh samples/sample.png -o relief.stl --base 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from image2mesh.converter import EXPORT_SUFFIXES, MAX_RESOLUTION, convert


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="image2mesh",
        description=(
            "Convert a 2D image into a 3D height-map mesh. "
            "Pixel brightness becomes height. It is not measured depth."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m image2mesh photo.png -o relief.stl --height 0.2 --base 0.05\n"
            "  python -m image2mesh logo.png -o logo.obj --invert --smooth-radius 2\n"
            "  python -m image2mesh photo.png -o relief.glb --screenshot preview.png\n"
        ),
    )
    parser.add_argument("image", help="Input image (PNG, JPEG, BMP, and other Pillow formats)")
    parser.add_argument(
        "-o",
        "--output",
        default="mesh.stl",
        help="Output path. Suffix picks the format: " + ", ".join(EXPORT_SUFFIXES),
    )
    parser.add_argument(
        "--height",
        type=float,
        default=1.0,
        help="Relief amplitude. The long side of the mesh has length 1, in the same units",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=f"Samples along the longer side (2–{MAX_RESOLUTION}). The other side keeps the image aspect",
    )
    parser.add_argument(
        "--square",
        action="store_true",
        help="Stretch the image to a square grid. By default the aspect ratio is preserved",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        default=True,
        help="Apply Gaussian smoothing",
    )
    parser.add_argument("--no-smooth", dest="smooth", action="store_false", help="Skip Gaussian smoothing")
    parser.add_argument(
        "--smooth-radius",
        type=float,
        default=1.0,
        help="Gaussian sigma, in pixels of the resized grid",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Dark pixels become high. Useful for a lithophane, together with --base",
    )
    parser.add_argument(
        "--fill-holes",
        action="store_true",
        default=True,
        help="Fill transparent pixels from the nearest opaque pixel",
    )
    parser.add_argument(
        "--no-fill-holes",
        dest="fill_holes",
        action="store_false",
        help="Treat transparent pixels as black instead of borrowing a neighbor",
    )
    parser.add_argument(
        "--base",
        type=float,
        default=0.0,
        metavar="THICKNESS",
        help="Thickness of a solid base. Above 0 the mesh is a watertight solid; 0 leaves an open sheet",
    )
    parser.add_argument(
        "--no-center",
        dest="center",
        action="store_false",
        default=True,
        help=(
            "Leave the mesh in the positive octant. "
            "By default the footprint is centered on X and Y and the base stays on z = 0"
        ),
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive PyVista window")
    parser.add_argument(
        "--screenshot",
        nargs="?",
        const="preview.png",
        default=None,
        metavar="PNG",
        help="Save an off-screen PNG preview. With no path, writes preview.png",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        print(
            f"Converting {args.image} -> {args.output} "
            f"(resolution={args.resolution}, height={args.height}, base={args.base})"
        )
        mesh = convert(
            args.image,
            args.output,
            height_scale=args.height,
            resolution=args.resolution,
            smooth=args.smooth,
            smooth_radius=args.smooth_radius,
            invert=args.invert,
            fill_holes=args.fill_holes,
            center=args.center,
            square=args.square,
            base_thickness=args.base,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rows = mesh.metadata.get("grid_rows")
    cols = mesh.metadata.get("grid_cols")
    low, high = mesh.bounds
    print(f"grid:       {rows} x {cols}")
    print(f"vertices:   {len(mesh.vertices):,}")
    print(f"faces:      {len(mesh.faces):,}")
    print(f"bounds:     {low.tolist()} -> {high.tolist()}")
    print(f"watertight: {'yes' if mesh.is_watertight else 'no'}")
    if mesh.is_watertight:
        print(f"volume:     {mesh.volume:.6g}")
    print(f"exported:   {args.output}")

    if args.screenshot:
        try:
            from image2mesh.visualizer import save_screenshot

            saved = save_screenshot(mesh, args.screenshot)
        except ImportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"error: could not render the screenshot: {exc}", file=sys.stderr)
            return 1
        print(f"screenshot: {saved}")

    if args.show:
        try:
            from image2mesh.visualizer import show_mesh

            show_mesh(mesh, title=f"Mesh preview — {Path(args.image).name}")
        except ImportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"error: could not open the preview window: {exc}", file=sys.stderr)
            return 1
    return 0

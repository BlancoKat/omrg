"""Image luminance to a triangle height map, optionally closed into a solid.

The long side of the mesh is 1 unit wide. ``height_scale`` is the relief
amplitude after the height map has been contrast-stretched to 0..1, and
``base_thickness`` is a slab under that relief. Pixel brightness is not
physical depth; see the README.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageOps

# trimesh writes these directly. 3MF also needs lxml and networkx, which
# export_mesh checks for so the error names the missing packages.
EXPORT_SUFFIXES = (".stl", ".obj", ".ply", ".glb", ".gltf", ".3mf")
MAX_RESOLUTION = 4096

# Pillow's ITU-R 601-2 luma, so opaque images match Image.convert("L").
_LUMA_R = 299
_LUMA_G = 587
_LUMA_B = 114


@dataclass
class ConversionParams:
    """Knobs for one conversion. Invalid combinations fail in ``__post_init__``."""

    height_scale: float = 1.0
    smooth: bool = True
    smooth_radius: float = 1.0
    resolution: int = 256
    invert: bool = False
    fill_holes: bool = True
    center: bool = True
    square: bool = False
    base_thickness: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.resolution, bool):
            raise ValueError("resolution must be an integer")
        self.resolution = int(round(float(self.resolution)))
        self.height_scale = float(self.height_scale)
        self.smooth_radius = float(self.smooth_radius)
        self.base_thickness = float(self.base_thickness)
        self.smooth = bool(self.smooth)
        self.invert = bool(self.invert)
        self.fill_holes = bool(self.fill_holes)
        self.center = bool(self.center)
        self.square = bool(self.square)

        if self.resolution < 2 or self.resolution > MAX_RESOLUTION:
            raise ValueError(f"resolution must be between 2 and {MAX_RESOLUTION}")
        for name in ("height_scale", "smooth_radius", "base_thickness"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite number >= 0")


def target_size(width: int, height: int, resolution: int, square: bool) -> tuple[int, int]:
    """Return ``(width, height)`` of the resized sample grid.

    ``resolution`` is the longer side. The other side follows the source
    aspect ratio, and both sides stay at least 2 samples so a mesh exists.
    ``square=True`` stretches the image to ``resolution`` by ``resolution``.
    """

    if width < 1 or height < 1:
        raise ValueError("image width and height must be positive")
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    if square:
        return resolution, resolution
    if width >= height:
        target_w = resolution
        target_h = max(2, int(round(height * resolution / width)))
    else:
        target_h = resolution
        target_w = max(2, int(round(width * resolution / height)))
    return target_w, target_h


def load_height_map(image: str | Path | Image.Image, params: ConversionParams) -> np.ndarray:
    """Return a float array, shape ``(rows, cols)``, values in 0..1.

    Row 0 is the top of the image. Transparent pixels (alpha under 128 after
    the resize) are missing samples: filled from the nearest opaque pixel
    when ``params.fill_holes`` is set, otherwise treated as black.
    """

    rgba = _open_rgba(image)
    dst_w, dst_h = target_size(rgba.size[0], rgba.size[1], params.resolution, params.square)
    if (dst_w, dst_h) != rgba.size:
        rgba = rgba.resize((dst_w, dst_h), Image.Resampling.LANCZOS)

    channels = np.asarray(rgba, dtype=np.float64)
    luminance = (
        channels[..., 0] * _LUMA_R
        + channels[..., 1] * _LUMA_G
        + channels[..., 2] * _LUMA_B
    ) / (1000.0 * 255.0)
    missing = channels[..., 3] < 128.0
    field = _fill_missing(luminance, missing, params.fill_holes)

    if params.smooth and params.smooth_radius > 0:
        from scipy.ndimage import gaussian_filter

        field = gaussian_filter(field, sigma=params.smooth_radius, mode="reflect")
    if params.invert:
        field = 1.0 - field
    return _contrast_stretch(field)


def triangulate_grid(rows: int, cols: int) -> np.ndarray:
    """Two triangles per grid cell, wound so the normal points up the Z axis.

    The height map this is applied to must already be flipped so row 0 lies
    at minimum Y. Index of row ``i``, column ``j`` is ``i * cols + j``.
    """

    if rows < 2 or cols < 2:
        raise ValueError("a mesh grid needs at least 2 samples on each side")
    ii, jj = np.indices((rows - 1, cols - 1))
    v00 = ii * cols + jj
    v01 = v00 + 1
    v10 = v00 + cols
    v11 = v10 + 1
    lower = np.stack((v00, v01, v10), axis=-1).reshape(-1, 3)
    upper = np.stack((v01, v11, v10), axis=-1).reshape(-1, 3)
    return np.ascontiguousarray(np.vstack((lower, upper)), dtype=np.int64)


def build_mesh(height_map: np.ndarray, params: ConversionParams) -> trimesh.Trimesh:
    """Triangulate ``height_map`` and, when a base is requested, close it.

    ``height_map`` values are clamped at 0. With ``base_thickness == 0`` the
    result is an open sheet. A positive base adds a flat bottom at z = 0 and
    walls around the boundary, which is a watertight solid of positive volume.
    """

    field = np.asarray(height_map, dtype=np.float64)
    if field.ndim != 2 or min(field.shape) < 2:
        raise ValueError("height map must be a 2D array of at least 2x2")
    if not np.isfinite(field).all():
        raise ValueError("height map contains non-finite values")

    # Row 0 of the incoming map is the top of the picture. Flip it so +Y in
    # the mesh points toward the top of the image when +Z is up.
    field = np.maximum(np.flipud(field), 0.0)
    rows, cols = field.shape
    long_side = float(max(rows, cols))
    xs = np.linspace(0.0, cols / long_side, cols)
    ys = np.linspace(0.0, rows / long_side, rows)
    xx, yy = np.meshgrid(xs, ys)
    zz = params.base_thickness + field * params.height_scale
    top = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    faces = triangulate_grid(rows, cols)

    if params.base_thickness > 0:
        count = rows * cols
        bottom = np.column_stack((top[:, 0], top[:, 1], np.zeros(count, dtype=np.float64)))
        vertices = np.vstack((top, bottom))
        bottom_faces = np.ascontiguousarray(faces[:, ::-1] + count)
        wall_faces = _wall_faces(rows, cols, count)
        faces = np.vstack((faces, bottom_faces, wall_faces))
    else:
        vertices = top

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    _place_on_bed(mesh, params.center)
    mesh.metadata["grid_rows"] = int(rows)
    mesh.metadata["grid_cols"] = int(cols)
    mesh.metadata["base_thickness"] = float(params.base_thickness)
    return mesh


def export_mesh(mesh: trimesh.Trimesh, output_path: str | Path) -> Path:
    """Write ``mesh`` to ``output_path``. The suffix picks the format."""

    out = Path(output_path)
    suffix = out.suffix.lower()
    if suffix not in EXPORT_SUFFIXES:
        supported = ", ".join(EXPORT_SUFFIXES)
        raise ValueError(f"unsupported output format {out.suffix!r}; use one of {supported}")
    if suffix == ".3mf":
        _require_3mf()
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out)
    return out


def convert(
    image: str | Path | Image.Image,
    output_path: str | Path,
    **kwargs,
) -> trimesh.Trimesh:
    """Run the full pipeline and write the mesh. Returns the in-memory mesh."""

    params = ConversionParams(**kwargs)
    height_map = load_height_map(image, params)
    mesh = build_mesh(height_map, params)
    export_mesh(mesh, output_path)
    return mesh


def _open_rgba(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGBA")
    path = Path(image)
    if not path.is_file():
        raise FileNotFoundError(f"input image not found: {path}")
    with Image.open(path) as opened:
        rgba = ImageOps.exif_transpose(opened).convert("RGBA")
        rgba.load()
    return rgba


def _fill_missing(luminance: np.ndarray, missing: np.ndarray, fill: bool) -> np.ndarray:
    """Honor ``fill_holes`` for transparent samples.

    Filling copies the nearest opaque luminance (a real inpaint, not a mesh
    repair). Leaving holes unfilled stores black there so the grid stays
    complete and a later solid base can still be watertight.
    """

    if not missing.any():
        return luminance
    if missing.all():
        return np.zeros_like(luminance)
    if not fill:
        output = np.array(luminance, copy=True)
        output[missing] = 0.0
        return output
    from scipy.ndimage import distance_transform_edt

    indices = distance_transform_edt(missing, return_distances=False, return_indices=True)
    return luminance[tuple(indices)]


def _contrast_stretch(field: np.ndarray) -> np.ndarray:
    lo = float(field.min())
    hi = float(field.max())
    if hi > lo:
        return (field - lo) / (hi - lo)
    return np.zeros_like(field)


def _boundary_edges(rows: int, cols: int) -> np.ndarray:
    """Boundary vertex pairs walked counter-clockwise when viewed from +Z.

    The interior is on the left of each edge, which is the walk the wall
    faces use to point their normals outward.
    """

    j = np.arange(cols - 1, dtype=np.int64)
    south = np.column_stack((j, j + 1))
    i = np.arange(rows - 1, dtype=np.int64)
    east = np.column_stack((i * cols + (cols - 1), (i + 1) * cols + (cols - 1)))
    jn = np.arange(cols - 1, 0, -1, dtype=np.int64)
    north = np.column_stack(((rows - 1) * cols + jn, (rows - 1) * cols + jn - 1))
    iw = np.arange(rows - 1, 0, -1, dtype=np.int64)
    west = np.column_stack((iw * cols, (iw - 1) * cols))
    return np.vstack((south, east, north, west))


def _wall_faces(rows: int, cols: int, count: int) -> np.ndarray:
    edges = _boundary_edges(rows, cols)
    start = edges[:, 0]
    end = edges[:, 1]
    below_start = start + count
    below_end = end + count
    first = np.column_stack((start, below_start, below_end))
    second = np.column_stack((start, below_end, end))
    return np.vstack((first, second))


def _place_on_bed(mesh: trimesh.Trimesh, center: bool) -> None:
    """Keep the lowest vertex on z = 0. Optionally center X and Y.

    Centering Z as well would sink half of a solid through the build plate.
    ``center=False`` leaves the minimum corner at the origin.
    """

    bounds = mesh.bounds
    offset = np.zeros(3, dtype=np.float64)
    offset[2] = bounds[0, 2]
    if center:
        offset[0] = (bounds[0, 0] + bounds[1, 0]) / 2.0
        offset[1] = (bounds[0, 1] + bounds[1, 1]) / 2.0
    if np.any(offset):
        mesh.apply_translation(-offset)


def _require_3mf() -> None:
    missing = []
    try:
        import lxml  # noqa: F401
    except ImportError:
        missing.append("lxml")
    try:
        import networkx  # noqa: F401
    except ImportError:
        missing.append("networkx")
    if missing:
        names = " and ".join(missing)
        raise RuntimeError(
            f"3MF export needs {names}, which trimesh imports when writing a .3mf file. "
            "They are listed in requirements.txt."
        )

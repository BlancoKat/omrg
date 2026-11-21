"""Geometry, height-map options, and file export. No display required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from image2mesh.converter import (
    EXPORT_SUFFIXES,
    ConversionParams,
    build_mesh,
    convert,
    export_mesh,
    load_height_map,
    target_size,
    triangulate_grid,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "sample.png"


def _params(**kwargs) -> ConversionParams:
    values = dict(
        resolution=16,
        smooth=False,
        smooth_radius=1.0,
        height_scale=0.5,
        base_thickness=0.0,
        center=False,
        square=False,
        invert=False,
        fill_holes=True,
    )
    values.update(kwargs)
    return ConversionParams(**values)


def _gradient(width: int, height: int) -> Image.Image:
    row = np.linspace(0, 255, width, dtype=np.uint8)
    return Image.fromarray(np.tile(row, (height, 1)), mode="L")


def _reference_faces(rows: int, cols: int) -> np.ndarray:
    """The same triangulation, written as the obvious loop, for one small grid."""

    lower = []
    upper = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            v00 = i * cols + j
            v01 = v00 + 1
            v10 = v00 + cols
            v11 = v10 + 1
            lower.append((v00, v01, v10))
            upper.append((v01, v11, v10))
    return np.array(lower + upper, dtype=np.int64)


def _extents(mesh: trimesh.Trimesh) -> np.ndarray:
    return mesh.bounds[1] - mesh.bounds[0]


def test_sample_image_is_a_wide_png():
    assert SAMPLE.is_file()
    with Image.open(SAMPLE) as image:
        assert image.size == (160, 100)
        assert image.mode in {"L", "RGB", "RGBA"}


def test_triangulation_matches_the_cell_loop_and_scales():
    for rows, cols in ((2, 2), (3, 5), (8, 4)):
        faces = triangulate_grid(rows, cols)
        assert np.array_equal(faces, _reference_faces(rows, cols))
        assert len(faces) == 2 * (rows - 1) * (cols - 1)
    # A few hundred thousand triangles should be a straight numpy build.
    wide = triangulate_grid(200, 360)
    assert len(wide) == 2 * 199 * 359


def test_aspect_ratio_is_preserved_unless_square_is_set():
    image = _gradient(40, 20)
    kept = load_height_map(image, _params(resolution=20, square=False))
    stretched = load_height_map(image, _params(resolution=20, square=True))
    assert kept.shape == (10, 20)
    assert stretched.shape == (20, 20)
    assert target_size(40, 20, 20, square=False) == (20, 10)
    assert target_size(40, 20, 20, square=True) == (20, 20)

    kept_mesh = build_mesh(kept, _params(height_scale=0.2))
    square_mesh = build_mesh(stretched, _params(height_scale=0.2))
    kept_size = _extents(kept_mesh)
    square_size = _extents(square_mesh)
    assert kept_size[0] / kept_size[1] == pytest.approx(40 / 20)
    assert square_size[0] == pytest.approx(square_size[1])


def test_very_thin_image_still_has_two_samples_on_the_short_side():
    image = Image.new("L", (100, 1), color=128)
    field = load_height_map(image, _params(resolution=50, square=False))
    assert field.shape == (2, 50)


def test_image_top_is_positive_y():
    image = Image.new("L", (8, 8), color=0)
    for y in range(3):
        for x in range(8):
            image.putpixel((x, y), 255)
    field = load_height_map(image, _params(resolution=8, smooth=False))
    assert field[0].mean() > field[-1].mean()
    mesh = build_mesh(field, _params(height_scale=1.0, base_thickness=0.0))
    top = mesh.vertices[np.argmax(mesh.vertices[:, 1])]
    bottom = mesh.vertices[np.argmin(mesh.vertices[:, 1])]
    assert top[2] > bottom[2]


def test_invert_flips_the_stretched_height_map():
    image = _gradient(16, 8)
    plain = load_height_map(image, _params(resolution=16))
    flipped = load_height_map(image, _params(resolution=16, invert=True))
    assert plain[0, 0] == pytest.approx(0.0)
    assert plain[0, -1] == pytest.approx(1.0)
    assert np.allclose(flipped, 1.0 - plain)


def test_smoothing_reduces_the_steepest_step():
    image = Image.new("L", (32, 16), color=0)
    for x in range(16, 32):
        for y in range(16):
            image.putpixel((x, y), 255)
    sharp = load_height_map(image, _params(resolution=32, smooth=False))
    soft = load_height_map(image, _params(resolution=32, smooth=True, smooth_radius=2.0))
    sharp_step = np.abs(np.diff(sharp.mean(axis=0))).max()
    soft_step = np.abs(np.diff(soft.mean(axis=0))).max()
    assert sharp.max() == pytest.approx(1.0)
    assert soft.max() == pytest.approx(1.0)
    assert soft_step < sharp_step


def test_fill_holes_inpaints_transparent_pixels():
    from image2mesh.converter import _fill_missing

    luminance = np.array(
        [
            [0.2, 0.2, np.nan, 0.8],
            [0.2, np.nan, np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    missing = np.isnan(luminance)
    filled = _fill_missing(luminance, missing, fill=True)
    # (1, 1) is one step from a 0.2 sample. (1, 3) is one step from 0.8.
    assert filled[1, 1] == pytest.approx(0.2)
    assert filled[1, 3] == pytest.approx(0.8)
    assert np.isfinite(filled).all()
    punched = _fill_missing(luminance, missing, fill=False)
    assert punched[1, 1] == pytest.approx(0.0)
    assert punched[1, 3] == pytest.approx(0.0)
    assert punched[1, 0] == pytest.approx(0.2)

    rgba = np.zeros((8, 12, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = 255
    rgba[:, 4:8, 3] = 0
    image = Image.fromarray(rgba, mode="RGBA")
    common = dict(resolution=12, square=False, smooth=False)
    inpainted = load_height_map(image, _params(**common, fill_holes=True))
    left_open = load_height_map(image, _params(**common, fill_holes=False))
    # One opaque color, once the hole is filled, has no contrast left.
    assert np.allclose(inpainted, 0.0)
    assert left_open[:, 4:8].mean() == pytest.approx(0.0)
    assert left_open[:, :4].mean() == pytest.approx(1.0)


def test_open_sheet_is_not_a_solid_and_faces_point_up():
    flat = build_mesh(np.zeros((4, 4)), _params(height_scale=0.3, base_thickness=0.0))
    assert flat.metadata["grid_rows"] == 4
    assert len(flat.vertices) == 16
    assert len(flat.faces) == 2 * 3 * 3
    assert not flat.is_watertight
    assert np.allclose(flat.face_normals[:, 2], 1.0, atol=1e-6)
    assert flat.vertex_normals.shape == flat.vertices.shape

    sloped = np.zeros((4, 4))
    sloped[:, 2:] = 1.0
    mesh = build_mesh(sloped, _params(height_scale=0.3, base_thickness=0.0))
    assert not mesh.is_watertight
    assert np.all(mesh.face_normals[:, 2] > 0.0)
    assert mesh.bounds[0, 2] == pytest.approx(0.0)
    assert _extents(mesh)[2] == pytest.approx(0.3)


def test_base_is_watertight_with_positive_volume():
    field = np.zeros((5, 9))
    field[:, 4:] = 1.0
    mesh = build_mesh(field, _params(height_scale=0.4, base_thickness=0.1, center=True))
    top = 2 * 4 * 8
    walls = 2 * 2 * (5 + 9 - 2)
    assert len(mesh.vertices) == 2 * 5 * 9
    assert len(mesh.faces) == 2 * top + walls
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0
    assert mesh.is_volume
    size = _extents(mesh)
    assert mesh.bounds[0, 2] == pytest.approx(0.0)
    assert size[2] == pytest.approx(0.5)
    assert abs(mesh.bounds[0, 0] + mesh.bounds[1, 0]) < 1e-8
    assert abs(mesh.bounds[0, 1] + mesh.bounds[1, 1]) < 1e-8
    # Relief sits on the base, so the volume is above a slab of the footprint
    # and below a box as tall as the highest point.
    footprint = size[0] * size[1]
    assert mesh.volume > 0.1 * footprint
    assert mesh.volume < 0.5 * footprint


def test_flat_image_with_base_is_a_slab():
    field = np.zeros((4, 4))
    mesh = build_mesh(field, _params(height_scale=1.0, base_thickness=0.2, center=False))
    assert mesh.is_volume
    assert mesh.volume == pytest.approx(0.2, rel=1e-6)
    assert mesh.bounds[0, 0] == pytest.approx(0.0)
    assert mesh.bounds[0, 1] == pytest.approx(0.0)


def test_random_relief_stays_a_volume():
    field = np.random.default_rng(0).random((24, 36))
    mesh = build_mesh(field, _params(height_scale=0.25, base_thickness=0.04, center=True))
    assert mesh.is_watertight
    assert mesh.is_volume
    assert mesh.bounds[0, 2] == pytest.approx(0.0)


def test_no_center_keeps_the_minimum_corner_at_the_origin():
    field = np.linspace(0.0, 1.0, 12).reshape(3, 4)
    mesh = build_mesh(field, _params(base_thickness=0.05, center=False))
    assert mesh.bounds[0] == pytest.approx([0.0, 0.0, 0.0])


@pytest.mark.parametrize("suffix", EXPORT_SUFFIXES)
def test_export_roundtrip_keeps_a_watertight_solid(suffix, tmp_path):
    image = _gradient(20, 10)
    destination = tmp_path / f"relief{suffix}"
    mesh = convert(
        image,
        destination,
        resolution=10,
        smooth=False,
        height_scale=0.3,
        base_thickness=0.08,
        square=False,
        center=True,
    )
    assert destination.is_file()
    assert destination.stat().st_size > 0
    assert mesh.is_watertight
    loaded = trimesh.load(destination, force="mesh")
    assert len(loaded.faces) == len(mesh.faces)
    assert loaded.is_watertight
    assert loaded.volume > 0


def test_open_sheet_export_is_not_watertight(tmp_path):
    image = _gradient(12, 8)
    destination = tmp_path / "sheet.stl"
    mesh = convert(image, destination, resolution=8, smooth=False, base_thickness=0.0, height_scale=0.2)
    assert not mesh.is_watertight
    loaded = trimesh.load(destination, force="mesh")
    assert not loaded.is_watertight


def test_export_rejects_an_unknown_suffix(tmp_path):
    mesh = build_mesh(np.zeros((4, 4)), _params(base_thickness=0.1))
    with pytest.raises(ValueError, match="unsupported"):
        export_mesh(mesh, tmp_path / "relief.fbx")


def test_missing_image_and_bad_resolution(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert(tmp_path / "missing.png", tmp_path / "out.stl")
    with pytest.raises(ValueError):
        ConversionParams(resolution=1)
    with pytest.raises(ValueError):
        ConversionParams(base_thickness=-0.1)
    with pytest.raises(ValueError):
        ConversionParams(height_scale=float("nan"))

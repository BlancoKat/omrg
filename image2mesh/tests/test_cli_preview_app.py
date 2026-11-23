"""CLI, off-screen preview, and the Gradio upload helper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from image2mesh.cli import parse_args
from image2mesh.converter import build_mesh
from image2mesh.visualizer import PREVIEW_SIZE, trimesh_to_polydata

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "sample.png"


def _relief():
    from image2mesh.converter import ConversionParams

    field = np.zeros((6, 10))
    field[1:5, 2:8] = 1.0
    return build_mesh(
        field,
        ConversionParams(
            height_scale=0.3,
            base_thickness=0.05,
            smooth=False,
            center=True,
            resolution=10,
        ),
    )


def test_parser_defaults_and_flags():
    defaults = parse_args(["photo.png"])
    assert defaults.output == "mesh.stl"
    assert defaults.height == 1.0
    assert defaults.resolution == 256
    assert defaults.smooth is True
    assert defaults.smooth_radius == 1.0
    assert defaults.invert is False
    assert defaults.fill_holes is True
    assert defaults.center is True
    assert defaults.square is False
    assert defaults.base == 0.0
    assert defaults.show is False
    assert defaults.screenshot is None

    flags = parse_args(
        [
            "photo.png",
            "-o",
            "out.glb",
            "--height",
            "0.2",
            "--resolution",
            "64",
            "--no-smooth",
            "--smooth-radius",
            "2.5",
            "--invert",
            "--no-fill-holes",
            "--no-center",
            "--square",
            "--base",
            "0.04",
            "--screenshot",
            "shot.png",
            "--show",
        ]
    )
    assert flags.output == "out.glb"
    assert flags.height == 0.2
    assert flags.resolution == 64
    assert flags.smooth is False
    assert flags.smooth_radius == 2.5
    assert flags.invert is True
    assert flags.fill_holes is False
    assert flags.center is False
    assert flags.square is True
    assert flags.base == 0.04
    assert flags.screenshot == "shot.png"
    assert flags.show is True
    bare = parse_args(["photo.png", "--screenshot"])
    assert bare.screenshot == "preview.png"


def test_pyvista_faces_are_padded():
    pv = pytest.importorskip("pyvista")
    from pyvista.core.errors import CellSizeError

    mesh = _relief()
    with pytest.raises(CellSizeError):
        pv.PolyData(np_vertices(mesh), mesh.faces)
    poly = trimesh_to_polydata(mesh)
    assert poly.n_cells == len(mesh.faces)
    assert int(poly.faces[0]) == 3
    assert len(poly.faces) == len(mesh.faces) * 4


def np_vertices(mesh):
    import numpy as np

    return np.asarray(mesh.vertices)


def test_screenshot_is_written_off_screen(tmp_path):
    pytest.importorskip("pyvista")
    from image2mesh.visualizer import save_screenshot

    destination = tmp_path / "preview.png"
    save_screenshot(_relief(), destination)
    assert destination.is_file()
    assert destination.stat().st_size > 0
    with Image.open(destination) as image:
        assert image.size == PREVIEW_SIZE
        colors = np.asarray(image)
    # A flat frame would mean the mesh never landed in the picture.
    assert colors.std() > 0


def test_cli_smoke_on_the_sample_image(tmp_path):
    output = tmp_path / "sample.stl"
    preview = tmp_path / "sample.png"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "image2mesh",
            str(SAMPLE),
            "-o",
            str(output),
            "--height",
            "0.2",
            "--resolution",
            "48",
            "--base",
            "0.05",
            "--screenshot",
            str(preview),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "watertight: yes" in result.stdout
    assert output.is_file()
    assert preview.is_file()
    loaded = trimesh.load(output, force="mesh")
    assert loaded.is_watertight
    assert loaded.is_volume
    size = loaded.bounds[1] - loaded.bounds[0]
    assert size[0] / size[1] == pytest.approx(160 / 100, rel=1e-3)
    assert loaded.bounds[0, 2] == pytest.approx(0.0, abs=1e-4)
    with Image.open(preview) as image:
        assert image.size == PREVIEW_SIZE


def test_cli_reports_a_missing_image(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "image2mesh", str(tmp_path / "nope.png"), "-o", str(tmp_path / "out.stl")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "not found" in result.stderr.lower()


def test_cli_usage_error_without_an_image():
    result = subprocess.run(
        [sys.executable, "-m", "image2mesh"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2


def test_upload_writes_unique_files_not_data_urls():
    from image2mesh.app import convert_upload

    image = Image.new("L", (18, 10), color=0)
    for x in range(9, 18):
        for y in range(10):
            image.putpixel((x, y), 220)
    preview_a, mesh_a = convert_upload(
        image,
        output_format="stl",
        height=0.25,
        resolution=12,
        base=0.06,
        smooth=False,
    )
    preview_b, mesh_b = convert_upload(
        image,
        output_format="obj",
        height=0.25,
        resolution=12,
        base=0.0,
        smooth=False,
    )
    assert mesh_a != mesh_b
    assert Path(mesh_a).is_file()
    assert Path(mesh_b).is_file()
    assert not mesh_a.startswith("data:")
    assert not mesh_b.startswith("data:")
    solid = trimesh.load(mesh_a, force="mesh")
    sheet = trimesh.load(mesh_b, force="mesh")
    assert solid.is_watertight
    assert solid.is_volume
    assert not sheet.is_watertight
    if preview_a is not None:
        assert preview_a != preview_b
        assert preview_a.endswith(".png")
        assert Path(preview_a).is_file()
        assert not preview_a.startswith("data:")


def test_gradio_demo_builds():
    pytest.importorskip("gradio")
    from image2mesh.app import build_demo

    demo = build_demo()
    assert demo is not None

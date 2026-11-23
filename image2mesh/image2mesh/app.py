"""Gradio UI. Launch with ``python -m image2mesh.app`` from the project directory.

Each conversion writes a unique file under a process-local temporary
directory and hands that path to Gradio. The download is a real file,
which is what ``gr.File`` serves.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from image2mesh.converter import convert

_WORKDIR = tempfile.TemporaryDirectory(prefix="image2mesh-")


def convert_upload(
    image,
    output_format: str = "stl",
    height: float = 1.0,
    resolution: int = 128,
    base: float = 0.0,
    smooth: bool = True,
    smooth_radius: float = 1.0,
    invert: bool = False,
    center: bool = True,
    square: bool = False,
    fill_holes: bool = True,
) -> tuple[str | None, str]:
    """Convert one upload. Returns ``(preview_png_or_None, mesh_path)``.

    Paths are unique. The preview is omitted, rather than failing the
    conversion, when PyVista cannot render.
    """

    if image is None:
        raise ValueError("upload an image first")
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise ValueError("expected a Pillow image or an array")

    suffix = output_format if str(output_format).startswith(".") else f".{output_format}"
    suffix = suffix.lower()
    mesh_path = _temp_path(suffix)
    convert(
        image,
        mesh_path,
        height_scale=float(height),
        resolution=int(round(float(resolution))),
        smooth=bool(smooth),
        smooth_radius=float(smooth_radius),
        invert=bool(invert),
        fill_holes=bool(fill_holes),
        center=bool(center),
        square=bool(square),
        base_thickness=float(base),
    )
    return _try_preview(mesh_path), mesh_path


def build_demo():
    """Construct the Gradio Blocks app. Imports Gradio only when called."""

    import gradio as gr

    def predict(
        image,
        output_format,
        height,
        resolution,
        base,
        smooth,
        smooth_radius,
        invert,
        center,
        square,
        fill_holes,
    ):
        if image is None:
            raise gr.Error("Upload an image first.")
        try:
            preview, mesh_path = convert_upload(
                image,
                output_format=output_format,
                height=height,
                resolution=resolution,
                base=base,
                smooth=smooth,
                smooth_radius=smooth_radius,
                invert=invert,
                center=center,
                square=square,
                fill_holes=fill_holes,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc
        return preview, mesh_path

    with gr.Blocks(title="Image to 3D mesh") as demo:
        gr.Markdown(
            """
            # Image to 3D mesh

            Brightness becomes a height map, then a triangle mesh.
            This is a relief, not a reconstruction of the object in the photo.
            Set **base thickness** above 0 for a watertight solid you can print.
            A base of 0 is an open sheet.
            """
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Input image")
                output_format = gr.Radio(
                    ["stl", "obj", "ply", "glb", "gltf", "3mf"],
                    value="stl",
                    label="Format",
                )
                height = gr.Slider(0, 5, value=1.0, step=0.05, label="Height scale")
                resolution = gr.Slider(
                    16,
                    512,
                    value=128,
                    step=16,
                    label="Resolution (samples on the long side)",
                )
                base = gr.Slider(
                    0,
                    1,
                    value=0.0,
                    step=0.01,
                    label="Base thickness (0 = open sheet, not a solid)",
                )
                smooth = gr.Checkbox(value=True, label="Smooth")
                smooth_radius = gr.Slider(0, 8, value=1.0, step=0.1, label="Smooth radius")
                invert = gr.Checkbox(value=False, label="Invert (dark = high)")
                center = gr.Checkbox(value=True, label="Center footprint, base on z = 0")
                square = gr.Checkbox(value=False, label="Square resize (stretches the image)")
                fill_holes = gr.Checkbox(value=True, label="Fill transparent pixels")
                button = gr.Button("Convert", variant="primary")
            with gr.Column():
                preview = gr.Image(label="Preview")
                download = gr.File(label="Download mesh")
        button.click(
            predict,
            inputs=[
                image,
                output_format,
                height,
                resolution,
                base,
                smooth,
                smooth_radius,
                invert,
                center,
                square,
                fill_holes,
            ],
            outputs=[preview, download],
        )
    return demo


def main() -> None:
    demo = build_demo()
    demo.launch()


def _temp_path(suffix: str) -> str:
    return str(Path(_WORKDIR.name) / f"{uuid.uuid4().hex}{suffix}")


def _try_preview(mesh_path: str) -> str | None:
    preview_path = _temp_path(".png")
    try:
        import trimesh

        from image2mesh.visualizer import save_screenshot

        mesh = trimesh.load(mesh_path, force="mesh")
        save_screenshot(mesh, preview_path)
    except Exception as exc:
        import warnings

        warnings.warn(f"preview render failed: {exc}", RuntimeWarning, stacklevel=2)
        return None
    return preview_path


if __name__ == "__main__":
    main()

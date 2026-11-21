"""PyVista preview. Imported only when a caller asks for a window or a PNG.

PyVista's ``PolyData`` wants a VTK cell array, ``[3, a, b, c, 3, d, e, f, ...]``.
Passing a raw ``(n, 3)`` face array from trimesh raises ``CellSizeError``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PREVIEW_SIZE = (800, 600)
_MESH_COLOR = "steelblue"


def trimesh_to_polydata(mesh):
    """Build a PyVista mesh whose cell count matches ``mesh.faces``."""

    pv = _import_pyvista()
    points = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        return pv.PolyData(points)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("expected a triangle face array of shape (n, 3)")
    counts = np.full((faces.shape[0], 1), 3, dtype=np.int64)
    cells = np.hstack((counts, faces)).ravel()
    return pv.PolyData(points, cells)


def show_mesh(mesh, title: str = "3D mesh preview") -> None:
    """Open an interactive window. Needs a display; use ``save_screenshot`` headless."""

    pv = _import_pyvista()
    plotter = pv.Plotter(off_screen=False, window_size=(1024, 768))
    try:
        plotter.add_mesh(
            trimesh_to_polydata(mesh),
            color=_MESH_COLOR,
            smooth_shading=True,
            show_edges=False,
        )
        plotter.add_axes()
        plotter.add_title(title)
        plotter.set_background("white")
        plotter.show()
    finally:
        plotter.close()


def save_screenshot(mesh, path: str | Path = "preview.png", window_size: tuple[int, int] = PREVIEW_SIZE) -> Path:
    """Render ``mesh`` to a PNG without opening a window."""

    pv = _import_pyvista()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # off_screen on this plotter only. Setting the global flag would also
    # force a later --show in the same process to render off-screen.
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    try:
        plotter.add_mesh(
            trimesh_to_polydata(mesh),
            color=_MESH_COLOR,
            smooth_shading=True,
            show_edges=False,
        )
        plotter.set_background("white")
        plotter.reset_camera()
        plotter.view_isometric()
        plotter.screenshot(str(out))
    finally:
        plotter.close()
    return out


def _import_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "PyVista is required for preview. Install it with: "
            "python -m pip install -r requirements-preview.txt"
        ) from exc
    return pv

"""Convert a 2D image into a 3D height-map mesh."""

from image2mesh.converter import (
    EXPORT_SUFFIXES,
    MAX_RESOLUTION,
    ConversionParams,
    build_mesh,
    convert,
    export_mesh,
    load_height_map,
    target_size,
    triangulate_grid,
)

__version__ = "1.0.0"

__all__ = [
    "EXPORT_SUFFIXES",
    "MAX_RESOLUTION",
    "ConversionParams",
    "build_mesh",
    "convert",
    "export_mesh",
    "load_height_map",
    "target_size",
    "triangulate_grid",
    "__version__",
]

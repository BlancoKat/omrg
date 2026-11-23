# image2mesh

Turn one 2D image into a triangle mesh by using brightness as height. The result is a relief, or a lithophane-style plate: a single surface over a flat base, not a model of the object in the photograph.

Developed using GrokBot. The design originated in a Qwen3.8 27B session; the transcript is in the repository root as `Qwen 21 Nov 2026 - SB Wiki.pdf`. This project is that design, rebuilt so the pieces that did not work actually run.

The long side of the mesh is 1 unit. `--height` is how far the brightest (or, with `--invert`, the darkest) pixel rises above the lowest pixel, in those same units. `--base` adds a solid slab under the relief and closes the surface so the mesh is watertight.

## How it works

1. The image is read as grayscale luminance (ITU-R 601-2, the same weights Pillow uses for mode `L`). EXIF orientation is applied first. Fully transparent pixels, alpha under 128 after resizing, are missing samples.
2. The picture is resized with a Lanczos filter. The longer side becomes `--resolution` samples and the other side follows the image aspect ratio, so a wide photo stays wide. `--square` stretches it to a square instead.
3. Missing samples are filled from the nearest opaque pixel when `--fill-holes` is on (the default). `--no-fill-holes` treats them as black.
4. An optional Gaussian blur (`--smooth`, sigma `--smooth-radius` in pixels of that resized grid) softens pixel steps.
5. `--invert` swaps high and low. The map is then contrast-stretched to 0..1, so `--height` is the full relief amplitude even when the photo is flat and gray.
6. The grid is triangulated with NumPy: two triangles per cell, normals pointing up. There is no per-cell Python loop.
7. With `--base THICKNESS` above 0, a second grid is placed at z = 0, the bottom faces are wound downward, and a wall is stitched around the boundary. The result is one closed solid.
8. The footprint is centered on X and Y and the lowest point stays on z = 0, so a solid sits on the build plate. `--no-center` leaves the minimum corner at the origin instead.

```
image  ->  grayscale height map  ->  triangle grid  ->  optional solid base
                                              |
                                              v
                                    STL OBJ PLY GLB GLTF 3MF
```

Brightness is only a stand-in for height. Nothing in the pipeline recovers hidden surfaces, the back of an object, or a true distance from the camera.

## Install

From this directory (`image2mesh/`), using the standard library `venv` and pip. That is the same on Omarchy (Arch Linux) and anywhere else with Python 3.10+. No system packages are required to convert and export.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Optional pieces, still with pip:

```bash
python -m pip install -r requirements-preview.txt   # PyVista, for --show and --screenshot
python -m pip install -r requirements-ui.txt        # Gradio UI, includes the preview extra
python -m pip install -r requirements-dev.txt       # pytest and the preview extra
```

`pip install -e ".[preview,ui,dev]"` installs the same set and adds an `image2mesh` command. Running `python -m image2mesh` from this directory does not need the editable install.

If Python itself is missing on Omarchy:

```bash
sudo pacman -S python python-pip
```

`--screenshot` renders off-screen through PyVista's bundled VTK and does not need a display. If that render fails because the machine has no OpenGL driver at all:

```bash
sudo pacman -S libglvnd mesa
```

`--show` opens a real window, so it needs a running desktop session. On a Wayland session that is the normal Omarchy setup; the off-screen path does not use that window.

## Usage

```bash
# Printable solid. Long side is 1 unit; scale it in the slicer.
python -m image2mesh samples/sample.png -o relief.stl --height 0.15 --base 0.04 --resolution 160

# Dark ink becomes the thick parts (a simple lithophane).
python -m image2mesh samples/sample.png -o plate.obj --invert --height 0.08 --base 0.03 --smooth-radius 1.5

# Keep a wide image wide. This is the default; --square is what stretches it.
python -m image2mesh samples/sample.png -o wide.ply --resolution 128

# Off-screen preview, no window. Writes preview.png when the path is omitted.
python -m image2mesh samples/sample.png -o relief.glb --height 0.2 --base 0.05 --screenshot preview.png

# Interactive view, on a machine with a display.
python -m image2mesh samples/sample.png -o relief.glb --show
```

`samples/sample.png` is a small generated 160×100 pattern you can use to try the tool without bringing your own image.

A base of 0 writes an open sheet. Slicers that insist on a closed volume will reject it. Pass `--base` with a thickness above 0 for a file meant to be printed.

For a 100 mm wide print with the example numbers above (`--height 0.15 --base 0.04`), scale the long side to 100 mm in the slicer. The relief is then 15 mm and the base is 4 mm, because every dimension is in the same units and the long side started at length 1.

### Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `image` | | Input image. Pillow formats: PNG, JPEG, BMP, TIFF, and so on. |
| `-o`, `--output` | `mesh.stl` | Output path. The suffix selects the format. |
| `--height` | `1.0` | Relief amplitude, in the same units as the mesh. The long side has length 1. |
| `--resolution` | `256` | Samples on the longer side, from 2 to 4096. The short side follows the aspect ratio and is at least 2. |
| `--square` | off | Stretch to `resolution` × `resolution`. |
| `--smooth` / `--no-smooth` | on | Gaussian blur before the contrast stretch. |
| `--smooth-radius` | `1.0` | Blur sigma, in pixels of the resized grid. |
| `--invert` | off | Dark pixels become high. |
| `--fill-holes` / `--no-fill-holes` | on | Transparent pixels are filled from the nearest opaque pixel, or treated as black. |
| `--base THICKNESS` | `0` | Slab under the relief. Above 0, the mesh is a closed solid. |
| `--no-center` | centering on | Leave the minimum corner at the origin. The default centers X and Y and keeps z = 0 on the base. |
| `--show` | off | Interactive PyVista window. Requires the preview extra and a display. |
| `--screenshot [PNG]` | off | Off-screen PNG. With no path, writes `preview.png`. |

A flat image has no contrast, so the stretch makes a flat sheet and only `--base` gives it thickness. A noisy image is stretched to the full `--height`, which can exaggerate grain; raise `--smooth-radius` when that happens.

### Output formats

| Suffix | Notes |
| --- | --- |
| `.stl` | Binary STL. The usual slicer input. |
| `.obj` | Wavefront OBJ, geometry only. |
| `.ply` | Polygon file. |
| `.glb`, `.gltf` | glTF 2.0, binary or JSON. |
| `.3mf` | 3D Manufacturing Format. trimesh writes it; `lxml` and `networkx` have to be installed, and they are in `requirements.txt`. |

### Gradio

```bash
python -m pip install -r requirements-ui.txt
python -m image2mesh.app
```

The page takes an image, the same options as the CLI, a preview render, and a file download. Each run writes a new temporary file and passes that path to the download control. The preview is skipped, and the mesh is still offered for download, if PyVista cannot render.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The tests are headless. They check aspect ratio, the solid base (`is_watertight` and positive volume), every exporter, the off-screen screenshot, and a CLI run against `samples/sample.png`. The Gradio layout test runs only when Gradio is installed.

## Limitations

- Brightness is not depth. A white shirt and a white wall land at the same height, and a dark crease is read as a valley (or a peak, with `--invert`).
- One image does not contain the back of an object, occluded surfaces, or a scale in millimeters. The mesh is a 2.5D height field: every point has a single z. Overhangs, caves, and separate objects are out of reach.
- The height map is contrast-stretched. `--height` is the gap between the darkest and lightest pixels after that stretch, not a multiplier on the raw 0–255 values.
- Without `--base`, the file is an open sheet of zero thickness. It is a surface, not a printable solid.
- With `--base`, bright regions (dark regions if inverted) are only as thick as the base, because the low end of the relief is z = base and the bottom of the solid is z = 0. That base is the minimum wall thickness.
- Transparent pixels are either inpainted or forced to black. Partial alpha is thresholded at 128. Animation frames other than the first are ignored.
- Very thin images are forced to at least 2 samples on the short side, so a 1-pixel-tall strip is not a faithful aspect ratio.
- Memory and file size grow with the square of `--resolution`. A 2048 grid is already millions of triangles.
- `--show` needs a display. `--screenshot` does not, but it does need the preview extra.

## Ideas for extending it

- Project the original image onto the relief as a UV texture (OBJ + MTL, or glTF).
- Replace luminance with a monocular depth model (MiDaS, DPT, or a later equivalent) and feed that map through the same mesher.
- Threshold a silhouette, trace it, and extrude the polygon for a cookie-cutter solid instead of a height field.
- Slice the height map into stacked layers and union them for a terraced relief.
- Give a lithophane a dedicated minimum and maximum thickness in millimeters, with a `--width` that sizes the long side directly.
- Batch a directory of images, or call the same functions from a Blender or FreeCAD script. `convert` and `build_mesh` are the API.

## Layout

```
image2mesh/
  README.md
  LICENSE
  pyproject.toml
  requirements.txt
  requirements-preview.txt
  requirements-ui.txt
  requirements-dev.txt
  image2mesh/
    __init__.py
    __main__.py
    cli.py
    converter.py
    visualizer.py
    app.py
  samples/sample.png
  tests/
```

## License

MIT, same as the rest of this repository. See `LICENSE`.

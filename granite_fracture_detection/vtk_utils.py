"""Utilities for exporting volumetric data to the VTK legacy format.

This module provides a simple function to write one or more volumetric
fields to a VTK file so that they can be visualised in 3D viewers such as
ParaView.  Only the ASCII legacy format with structured points is
implemented.  All fields must have the same shape.
"""

from __future__ import annotations

import numpy as np
from typing import Mapping, Sequence

__all__ = ["write_vtk"]


def write_vtk(
    filename: str,
    data: Mapping[str, np.ndarray],
    *,
    spacing: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
) -> None:
    """Write one or more 3‑D fields to a legacy VTK file.

    Parameters
    ----------
    filename : str
        Path of the file to create.  The suffix ``.vtk`` is not appended
        automatically.
    data : mapping
        Dictionary mapping field names to 3‑D NumPy arrays.  All arrays
        must have the same shape, interpreted as ``(z, y, x)``.
    spacing : sequence of 3 floats, optional
        Physical spacing between voxels along the x, y and z axes.
        Defaults to ``(1.0, 1.0, 1.0)``.
    origin : sequence of 3 floats, optional
        Physical coordinates of the origin.  Defaults to ``(0.0, 0.0, 0.0)``.

    Notes
    -----
    The VTK "structured points" dataset requires that data are stored
    point‑wise in x‑major order.  Internally, this function transposes
    arrays to match the required ordering (x, y, z) before flattening.
    """
    if not data:
        raise ValueError("No data provided for VTK output")
    # Verify all arrays have same shape
    shapes = {arr.shape for arr in data.values()}
    if len(shapes) != 1:
        raise ValueError("All fields must have the same shape")
    z_dim, y_dim, x_dim = next(iter(shapes))
    # Default spacing and origin
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)
    if origin is None:
        origin = (0.0, 0.0, 0.0)
    npts = x_dim * y_dim * z_dim
    # Prepare header lines
    lines: list[str] = []
    lines.append("# vtk DataFile Version 3.0")
    lines.append("Volume data written by granite_fracture_detection")
    lines.append("ASCII")
    lines.append("DATASET STRUCTURED_POINTS")
    lines.append(f"DIMENSIONS {x_dim} {y_dim} {z_dim}")
    lines.append(f"ORIGIN {origin[0]} {origin[1]} {origin[2]}")
    lines.append(f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}")
    lines.append(f"POINT_DATA {npts}")
    # Append each field
    for name, arr in data.items():
        # Flatten in x-major order: transpose to (x, y, z) then ravel
        arr_float = np.asarray(arr, dtype=float)
        flat = np.transpose(arr_float, (2, 1, 0)).ravel()
        lines.append(f"SCALARS {name} float 1")
        lines.append("LOOKUP_TABLE default")
        # Write values: break lines for readability (e.g., 9 per line)
        for i in range(0, len(flat), 9):
            chunk = flat[i : i + 9]
            lines.append(" ".join(f"{val:.6g}" for val in chunk))
    # Write file
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

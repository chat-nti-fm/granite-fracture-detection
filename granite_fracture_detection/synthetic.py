"""Synthetic data generator for fracture detection algorithms.

This module provides functions to build synthetic CT volumes that emulate
beam‑hardening artefacts, ring artefacts and thin planar fractures.  The
generated volumes are intended for algorithm development and testing rather
than photorealistic simulation.  Fractures are modelled as infinitely
extended planes with optional filling; their apertures are much thinner than
the voxel size.  Beam hardening is simulated as a quadratic radial shading
and ring artefacts as high‑frequency radial noise.

Example
-------

>>> vol, gt = generate_synthetic_volume((32, 128, 128), num_fractures=3)
>>> vol_corrected = correct_beam_hardening(vol)
>>> vol_denoised = remove_radial_artifacts(vol_corrected)
>>> fracture, fracture_std = detect_fractures(vol_denoised)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["generate_synthetic_volume"]


def _make_plane(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create two orthonormal vectors spanning a plane with normal ``n``.

    Parameters
    ----------
    n : ndarray of shape (3,)
        Unit normal vector.

    Returns
    -------
    u, v : ndarrays of shape (3,)
        Orthonormal basis vectors such that ``u × v = n`` and ``u·n = v·n = 0``.
    """
    # Choose an arbitrary vector not colinear with n
    if abs(n[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v


def generate_synthetic_volume(
    shape: tuple[int, int, int] = (32, 128, 128),
    *,
    num_fractures: int = 3,
    thickness: float = 1.0,
    noise_sigma: float = 0.02,
    ray_hardening_strength: float = 0.5,
    radial_fluctuation_strength: float = 0.1,
    fill_fraction: float = 0.3,
    rock_strength: float | None = None,
    rock_smooth: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic CT volume with optional rock structure, beam hardening, ring artefacts and fractures.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the volume ``(z, y, x)``.
    num_fractures : int, optional
        Number of planar fractures to embed in the volume.  Each fracture
        orientation and position is random.
    thickness : float, optional
        Half thickness of each fracture plane in voxel units.  The full
        aperture will therefore be ``2*thickness``.
    noise_sigma : float, optional
        Standard deviation of the additive Gaussian noise applied after all
        other artefacts.  Values are relative to the base intensity (1.0).
    ray_hardening_strength : float, optional
        Controls the magnitude of the quadratic radial shading.  A value of
        0 disables beam hardening; 0.5 reduces the intensity at the centre by
        50 % relative to the boundary.
    radial_fluctuation_strength : float, optional
        Amplitude of high‑frequency radial noise (ring artefacts).  This is
        added to the radial profile before shading.
    fill_fraction : float, optional
        Fraction of fractures that are considered filled (i.e. do not reduce
        intensity).  For such fractures the ground truth fracture fraction
        remains zero.
    rock_strength : float or None, optional
        Standard deviation of a synthetic rock texture to add to the base
        intensity.  If ``None`` (default), no rock texture is added.  A value
        greater than zero produces spatially correlated intensity fluctuations
        across the volume, emulating mineralogical heterogeneity of granite.
    rock_smooth : float, optional
        Standard deviation of the Gaussian kernel used to smooth the random
        field that generates the rock texture.  Larger values lead to more
        slowly varying rock structure.

    Returns
    -------
    volume : ndarray of shape ``shape``
        Simulated CT volume containing artefacts and fractures (and optional
        rock heterogeneity).
    gt_fracture : ndarray of shape ``shape``
        Ground‑truth fracture fraction per voxel (0–1).  This array can be
        used to assess the performance of detection algorithms.
    """
    z_dim, y_dim, x_dim = shape
    volume = np.ones(shape, dtype=np.float64)
    gt_frac = np.zeros(shape, dtype=np.float64)

    # Optionally add a synthetic rock texture to the base intensity.  Many
    # natural granites exhibit spatial variation in density due to different
    # mineralogical constituents.  We simulate this as a smoothed random
    # field whose standard deviation is ``rock_strength``.
    if rock_strength is not None and rock_strength > 0.0:
        rng = np.random.default_rng()
        rock_noise = rng.standard_normal(size=shape)
        from scipy.ndimage import gaussian_filter

        rock_noise = gaussian_filter(rock_noise, sigma=rock_smooth)
        # normalise to zero mean, unit variance
        rock_noise -= rock_noise.mean()
        if rock_noise.std() > 0:
            rock_noise /= rock_noise.std()
        # scale by requested strength and add to base intensity
        volume += rock_strength * rock_noise

    # Coordinates centred at origin
    zz, yy, xx = np.meshgrid(
        np.arange(z_dim) - (z_dim - 1) / 2.0,
        np.arange(y_dim) - (y_dim - 1) / 2.0,
        np.arange(x_dim) - (x_dim - 1) / 2.0,
        indexing="ij",
    )
    coords = np.stack((zz, yy, xx), axis=-1)  # shape (z, y, x, 3)

    # Insert planar fractures
    rng = np.random.default_rng()
    for i in range(num_fractures):
        # Random unit normal
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        # Random offset so that plane passes through the volume
        max_offset = 0.5 * np.sqrt(z_dim**2 + y_dim**2 + x_dim**2)
        offset = rng.uniform(-max_offset, max_offset)
        # Create plane basis
        u, v = _make_plane(n)
        # Generate fractal pattern along the plane using smoothed noise
        plane_extent = int(np.sqrt(y_dim**2 + x_dim**2))
        noise = rng.standard_normal((plane_extent, plane_extent))
        # Smooth noise to create large scale variation (fractures often ragged)
        from scipy.ndimage import gaussian_filter

        noise = gaussian_filter(noise, sigma=5)
        # Normalise noise to [0, 1]
        noise -= noise.min()
        if noise.max() > 0:
            noise /= noise.max()
        # Determine which points on the plane correspond to open fracture (not filled)
        filled = rng.random() < fill_fraction
        # For each voxel, compute signed distance and projection
        dist = np.tensordot(coords, n, axes=([3], [0])) - offset  # shape (z,y,x)
        mask = np.abs(dist) <= thickness
        if not filled:
            # Compute projection coordinates onto plane basis
            # Parameterise plane coordinates by integer grid for noise lookup
            proj_u = np.tensordot(coords, u, axes=([3], [0]))  # shape (z,y,x)
            proj_v = np.tensordot(coords, v, axes=([3], [0]))
            # Normalise projection to the range [0, plane_extent)
            proj_u_norm = (proj_u - proj_u.min()) / (proj_u.max() - proj_u.min()) * (plane_extent - 1)
            proj_v_norm = (proj_v - proj_v.min()) / (proj_v.max() - proj_v.min()) * (plane_extent - 1)
            # Sample noise pattern using nearest neighbour (indices must be ints)
            iu = np.clip(proj_u_norm.astype(int), 0, plane_extent - 1)
            iv = np.clip(proj_v_norm.astype(int), 0, plane_extent - 1)
            fracture_pattern = noise[iu, iv]
            # Add fracture thickness proportional to pattern
            frac_val = (1.0 - fracture_pattern)  # darker where noise low
            gt_frac[mask] = np.maximum(gt_frac[mask], frac_val[mask])
        # If the fracture is filled, do not modify gt_frac

    # Reduce intensity where fractures occur
    volume -= gt_frac

    # Apply radial beam hardening: quadratic shading from centre
    yy_grid, xx_grid = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing="ij")
    y_c = (y_dim - 1) / 2.0
    x_c = (x_dim - 1) / 2.0
    r = np.sqrt((yy_grid - y_c) ** 2 + (xx_grid - x_c) ** 2)
    r_max = r.max() if r.max() > 0 else 1.0
    shading = 1.0 - ray_hardening_strength * (r / r_max) ** 2
    volume *= shading[np.newaxis, :, :]

    # Add radial high‑frequency fluctuations (ring artefacts)
    r_int = np.floor(r).astype(int)
    r_max_int = int(r_int.max())
    # Generate 1D high‑frequency noise and remove its low‑frequency component
    radial_noise = rng.standard_normal(r_max_int + 1)
    from scipy.ndimage import gaussian_filter1d

    smooth = gaussian_filter1d(radial_noise, sigma=5.0, mode="nearest")
    high_freq = radial_noise - smooth
    # Scale by strength
    high_freq *= radial_fluctuation_strength
    # Add to each slice
    correction = high_freq[r_int]
    volume += correction[np.newaxis, :, :]

    # Add Gaussian noise
    volume += rng.normal(scale=noise_sigma, size=shape)

    # Clip to non‑negative intensities
    volume = np.clip(volume, 0.0, None)

    return volume, gt_frac

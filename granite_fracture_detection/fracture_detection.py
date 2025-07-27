"""Multi‑scale Hessian‑based fracture detection.

This module implements a simple Hessian‐matrix based filter to detect
dark, sheet‑like structures in three‑dimensional CT data.  Fractures
appearing darker than the surrounding material produce distinctive second
derivative signatures: along the plane of the fracture the intensity changes
slowly, whereas across the plane it drops sharply.  The Hessian of the
intensity field captures these behaviours in its eigenvalues.  By examining
the sign and relative magnitude of the eigenvalues across multiple scales,
we derive an estimate of the fraction of each voxel occupied by a fracture
and a measure of uncertainty.

The implementation here is inspired by vesselness and sheetness filters but
is simplified for robustness and computational efficiency【656784612022342†L28-L43】.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from numpy.typing import ArrayLike

__all__ = ["detect_fractures"]


def _hessian_eigvals(volume: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute eigenvalues of the Hessian matrix at a given scale.

    Parameters
    ----------
    volume : ndarray of shape (z, y, x)
        Input image data.
    sigma : float
        Standard deviation of the Gaussian kernel used to smooth the image
        before differentiation.  Larger values probe broader structures.

    Returns
    -------
    l1, l2, l3 : ndarrays of shape ``volume.shape``
        Sorted eigenvalues per voxel, sorted by increasing absolute value
        (|l1| <= |l2| <= |l3|).  Signs of the original eigenvalues are
        preserved.
    """
    # Compute second derivatives using Gaussian filters
    # The order parameter specifies the derivative order along each axis
    Ixx = gaussian_filter(volume, sigma=sigma, order=(2, 0, 0), mode="nearest")
    Iyy = gaussian_filter(volume, sigma=sigma, order=(0, 2, 0), mode="nearest")
    Izz = gaussian_filter(volume, sigma=sigma, order=(0, 0, 2), mode="nearest")
    Ixy = gaussian_filter(volume, sigma=sigma, order=(1, 1, 0), mode="nearest")
    Ixz = gaussian_filter(volume, sigma=sigma, order=(1, 0, 1), mode="nearest")
    Iyz = gaussian_filter(volume, sigma=sigma, order=(0, 1, 1), mode="nearest")

    # Flatten arrays for vectorised eigenvalue computation
    shape = volume.shape
    N = volume.size
    # Create Hessian matrices for each voxel, shape (N, 3, 3)
    H = np.empty((N, 3, 3), dtype=np.float64)
    H[:, 0, 0] = Ixx.ravel()
    H[:, 1, 1] = Iyy.ravel()
    H[:, 2, 2] = Izz.ravel()
    H[:, 0, 1] = H[:, 1, 0] = Ixy.ravel()
    H[:, 0, 2] = H[:, 2, 0] = Ixz.ravel()
    H[:, 1, 2] = H[:, 2, 1] = Iyz.ravel()

    eigs = np.linalg.eigvalsh(H).reshape(shape + (3,))
    # Sort eigenvalues by increasing absolute value while preserving sign
    abs_eigs = np.abs(eigs)
    # argsort returns indices that would sort along the last axis
    idx = np.argsort(abs_eigs, axis=-1)
    l1 = np.take_along_axis(eigs, idx[..., 0:1], axis=-1)[..., 0]
    l2 = np.take_along_axis(eigs, idx[..., 1:2], axis=-1)[..., 0]
    l3 = np.take_along_axis(eigs, idx[..., 2:3], axis=-1)[..., 0]
    return l1, l2, l3


def _sheetness(l1: np.ndarray, l2: np.ndarray, l3: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Compute a simple sheetness measure from sorted eigenvalues.

    For dark planes (fractures) we expect the largest magnitude eigenvalue
    ``l3`` to be negative (intensity decreases sharply across the plane),
    while ``l1`` and ``l2`` are comparatively small because the intensity
    varies slowly along the plane.  We therefore define the sheetness at
    each voxel as::

        S = max(0, -l3) * max(0, 1 - |l2|/|l3|) ** alpha

    where ``alpha`` controls how strongly the ratio penalises points
    that are not plate‑like.  A second factor penalises voxels where the
    average magnitude of eigenvalues is large relative to the strongest one
    using parameter ``beta``.  The final sheetness value is normalised
    between 0 and 1.

    Parameters
    ----------
    l1, l2, l3 : ndarrays
        Sorted eigenvalues from `_hessian_eigvals`.
    alpha : float
        Exponent controlling the weight of the eigenvalue ratio term.
    beta : float
        Exponent controlling the weight of the magnitude penalty.

    Returns
    -------
    sheetness : ndarray
        Estimated fracture fraction per voxel, normalised to [0, 1].
    """
    # Suppress voxels where the main eigenvalue is positive (bright plane)
    mask = l3 < 0
    # Avoid division by zero
    ratio = np.zeros_like(l3)
    abs_l3 = np.abs(l3)
    nonzero = abs_l3 > 0
    ratio[nonzero] = np.abs(l2[nonzero]) / abs_l3[nonzero]
    # Raw sheetness before normalisation
    S = np.zeros_like(l3)
    S[mask] = (-l3[mask]) * (np.maximum(0.0, 1.0 - ratio[mask]) ** alpha)
    # Penalise strong overall curvature (noise) via magnitude of all eigenvalues
    magnitude = np.sqrt(l1**2 + l2**2 + l3**2)
    mag_max = magnitude.max() if magnitude.max() > 0 else 1.0
    penalty = 1.0 - (magnitude / mag_max) ** beta
    S = S * penalty
    # Normalise S to [0, 1]
    max_S = S.max() if S.max() > 0 else 1.0
    return S / max_S


def detect_fractures(volume: ArrayLike, *, sigmas: list[float] | tuple[float, ...] = (1.0, 2.0, 3.0), alpha: float = 1.0, beta: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Identify fracture voxels using a multi‑scale Hessian filter.

    Parameters
    ----------
    volume : array‑like of shape (z, y, x)
        The input volume after artefact correction.  Intensities should be
        higher in the matrix and lower in fractures.
    sigmas : sequence of floats, optional
        Gaussian standard deviations defining the scales at which fractures
        are probed.  Larger sigmas detect wider fractures.
    alpha : float, optional
        Exponent controlling sensitivity of the eigenvalue ratio term in
        sheetness calculation.  Larger values favour cleaner planar
        structures.
    beta : float, optional
        Exponent controlling the penalty for overall curvature (noise).

    Returns
    -------
    fracture_mean : ndarray of shape ``volume.shape``
        Estimated fraction of each voxel occupied by a fracture, averaged
        across scales and normalised to [0, 1].
    fracture_std : ndarray of shape ``volume.shape``
        Standard deviation of the fracture estimate across scales, useful
        to gauge uncertainty.
    """
    img = np.asarray(volume, dtype=np.float64)
    if img.ndim != 3:
        raise ValueError("volume must be a 3‑D array (z, y, x)")
    sheets = []
    for sigma in sigmas:
        l1, l2, l3 = _hessian_eigvals(img, sigma)
        sheet = _sheetness(l1, l2, l3, alpha=alpha, beta=beta)
        sheets.append(sheet)
    sheets = np.stack(sheets, axis=0)  # shape (n_scales, z, y, x)
    fracture_mean = sheets.mean(axis=0)
    fracture_std = sheets.std(axis=0)
    return fracture_mean, fracture_std
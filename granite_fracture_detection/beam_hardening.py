"""Beam hardening correction for cylindrical CT data.

This module implements a simple beam‑hardening (also called cupping) correction
for reconstructed micro‑CT volumes of cylindrical samples.  The artefact
manifests itself as a gradual decrease of intensity from the outer boundary
towards the centre of the cylinder because low energy photons are absorbed
preferentially【284965845346994†L168-L178】.  The code below estimates a radial
intensity profile by averaging across all slices and fits a low‑order
polynomial to this profile.  The fitted profile is then used to remove the
radial bias from each slice.

The correction assumes that the sample has a roughly circular cross‑section and
that the core is centred in the image.  For real data where the centre is
offset, you should first recenter the volume or supply a custom centre.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["correct_beam_hardening"]


def _radial_profile(mean_slice: np.ndarray, r_int: np.ndarray, r_max: int) -> np.ndarray:
    """Compute the average intensity for each integer radial bin.

    Parameters
    ----------
    mean_slice : ndarray
        2‑D array of mean intensities in the (y, x) plane.
    r_int : ndarray
        2‑D array of integer radial distances for each pixel.
    r_max : int
        Maximum integer radial distance present in ``r_int``.

    Returns
    -------
    profile : ndarray of shape (r_max+1,)
        The mean intensity at each radial distance.  Values for bins with no
        pixels are interpolated from neighbouring bins.
    """
    profile = np.zeros(r_max + 1, dtype=np.float64)
    counts = np.zeros(r_max + 1, dtype=np.int64)
    # Accumulate sums and counts
    for r in range(r_max + 1):
        mask = r_int == r
        if mask.any():
            profile[r] = mean_slice[mask].mean()
            counts[r] = mask.sum()
        else:
            profile[r] = np.nan
    # Interpolate NaNs if present
    if np.isnan(profile).any():
        valid = ~np.isnan(profile)
        x = np.nonzero(valid)[0]
        y = profile[valid]
        profile = np.interp(np.arange(r_max + 1), x, y)
    return profile


def correct_beam_hardening(volume: ArrayLike, *, order: int = 2, centre: tuple[float, float] | None = None) -> np.ndarray:
    """Estimate and remove radial bias caused by beam hardening.

    This function operates on a 3‑D volume with shape ``(z, y, x)``.  It
    computes the average slice across the z‑axis and determines a radial
    intensity profile.  A low‑order polynomial (by default quadratic) is
    fitted to the profile to approximate the underlying bias.  The fitted
    function is then evaluated at each pixel's radius and subtracted from
    every slice so that the corrected volume has a uniform background.

    Parameters
    ----------
    volume : array‑like of shape (z, y, x)
        Input CT volume with attenuation values.  The data should be
        reconstructed intensities; raw projection data cannot be corrected
        using this method.
    order : int, optional
        Polynomial order used to fit the radial profile.  A higher order
        may better fit complex cupping behaviour but risks overfitting.
    centre : tuple of two floats or None
        Coordinates ``(y_c, x_c)`` of the cylinder centre in pixel units.
        If ``None``, the centre is assumed to be at the geometric centre of
        the volume.

    Returns
    -------
    corrected : ndarray of same shape as ``volume``
        Volume with the estimated radial bias removed.

    Notes
    -----
    The correction operates slice by slice but uses a single radial profile
    estimated from the mean of all slices.  For samples with significant
    variation along the z‑axis (e.g. heterogeneous materials), it may be
    beneficial to estimate a profile per slice or per small stack of slices.
    """
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError("volume must be a 3‑D array (z, y, x)")
    z_dim, y_dim, x_dim = vol.shape

    # Determine centre coordinates
    if centre is None:
        y_c = (y_dim - 1) / 2.0
        x_c = (x_dim - 1) / 2.0
    else:
        y_c, x_c = centre

    # Compute radial distances for the (y, x) grid
    yy, xx = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing="ij")
    r = np.sqrt((yy - y_c) ** 2 + (xx - x_c) ** 2)
    r_int = np.floor(r).astype(int)
    r_max = int(r_int.max())

    # Average slice across z
    mean_slice = vol.mean(axis=0)

    # Compute radial profile and fit polynomial
    profile = _radial_profile(mean_slice, r_int, r_max)
    radii = np.arange(r_max + 1)
    coeffs = np.polyfit(radii, profile, order)
    fitted = np.polyval(coeffs, r)

    # Remove bias: subtract fitted profile minus its minimum to keep dynamic range
    bias = fitted - fitted.min()
    corrected = vol - bias[np.newaxis, :, :]
    return corrected
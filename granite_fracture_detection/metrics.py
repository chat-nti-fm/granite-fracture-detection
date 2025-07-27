"""Utility metrics for evaluating artefact removal and fracture detection."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["snr", "f1_score", "mse"]


def snr(reference: ArrayLike, estimate: ArrayLike) -> float:
    """Compute the signal‑to‑noise ratio between a reference and an estimate.

    The SNR is defined as :math:`10 \log_{10}(\sigma_x^2 / \sigma_e^2)`, where
    ``σ_x^2`` is the variance of the reference signal and ``σ_e^2`` is the
    variance of the error ``estimate - reference``.  A larger SNR indicates a
    more accurate estimate.

    Parameters
    ----------
    reference : array‑like
        Ground‑truth data.
    estimate : array‑like
        Estimated data.

    Returns
    -------
    snr_db : float
        The SNR in decibels.  Returns ``inf`` if the estimate perfectly
        matches the reference.
    """
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    if ref.shape != est.shape:
        raise ValueError("reference and estimate must have the same shape")
    err = est - ref
    var_signal = np.var(ref)
    var_noise = np.var(err)
    if var_noise == 0:
        return float("inf")
    if var_signal == 0:
        return 0.0
    snr_val = 10.0 * np.log10(var_signal / var_noise)
    return snr_val


def f1_score(pred: ArrayLike, gt: ArrayLike, *, threshold: float = 0.5) -> float:
    """Compute the F1 score between a predicted fracture map and the ground truth.

    The F1 score is the harmonic mean of precision and recall.  Both the
    prediction and ground truth are thresholded at the given level to obtain
    binary masks.

    Parameters
    ----------
    pred : array‑like
        Predicted fracture fraction (float values between 0 and 1).
    gt : array‑like
        Ground‑truth fracture fraction (binary or continuous values in [0, 1]).
    threshold : float, optional
        Threshold applied to both ``pred`` and ``gt`` to obtain binary
        segmentation.

    Returns
    -------
    f1 : float
        F1 score between 0 and 1.  If no positive voxels are present in both
        ``pred`` and ``gt``, returns 1.0 by convention.
    """
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError("pred and gt must have the same shape")
    p_bin = p >= threshold
    g_bin = g >= threshold
    tp = np.logical_and(p_bin, g_bin).sum()
    fp = np.logical_and(p_bin, ~g_bin).sum()
    fn = np.logical_and(~p_bin, g_bin).sum()
    if tp + fp + fn == 0:
        return 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return f1


def mse(pred: ArrayLike, gt: ArrayLike) -> float:
    """Compute the mean squared error between a prediction and the ground truth."""
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError("pred and gt must have the same shape")
    return float(((p - g) ** 2).mean())
from __future__ import annotations

import numpy as np
import pandas as pd
from pybaselines.polynomial import modpoly
from scipy.interpolate import interp1d


def baseline_correction(df: pd.DataFrame, poly_order: int = 5) -> pd.DataFrame:
    """Apply ModPoly baseline correction row-wise."""
    corrected = []
    for spectrum in df.to_numpy(dtype=float):
        baseline, _ = modpoly(spectrum, poly_order=poly_order)
        corrected.append(spectrum - baseline)
    return pd.DataFrame(corrected, columns=df.columns, index=df.index)


def snv_normalization(spectra: np.ndarray) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=float)
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)
    return (spectra - mean) / np.maximum(std, 1e-8)


def vector_normalization(spectra: np.ndarray) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=float)
    norms = np.linalg.norm(spectra, axis=1, keepdims=True)
    return spectra / np.maximum(norms, 1e-12)


def interpolate_to_reference(df: pd.DataFrame, reference_axis: np.ndarray) -> pd.DataFrame:
    original_axis = df.columns.to_numpy(dtype=float)
    order = np.argsort(original_axis)
    original_axis = original_axis[order]
    values = df.to_numpy(dtype=float)[:, order]

    interpolated = np.empty((values.shape[0], len(reference_axis)), dtype=float)
    for i, row in enumerate(values):
        fn = interp1d(
            original_axis,
            row,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        interpolated[i] = fn(reference_axis)
    return pd.DataFrame(interpolated, columns=reference_axis, index=df.index)


def cosine_similarity_matrix(query: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Absolute cosine similarity, vectorized for serverless execution."""
    query = np.asarray(query, dtype=float)
    references = np.asarray(references, dtype=float)
    q = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    r = references / np.maximum(np.linalg.norm(references, axis=1, keepdims=True), 1e-12)
    return np.abs(q @ r.T)


def hqi_similarity_matrix(query: np.ndarray, references: np.ndarray) -> np.ndarray:
    """HQI equals squared cosine similarity for row vectors."""
    cos = cosine_similarity_matrix(query, references)
    return cos * cos

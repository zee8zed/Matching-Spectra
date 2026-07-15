from __future__ import annotations

import base64
import html
import io
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from preprocessing_utils import (
    baseline_correction,
    cosine_similarity_matrix,
    hqi_similarity_matrix,
    interpolate_to_reference,
    snv_normalization,
    vector_normalization,
)

ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "public" / "index.html"
REFERENCE_FILE = ROOT / "data" / "reference_data.pkl"
MAX_UPLOAD_BYTES = 4_000_000  # stays below Vercel's 4.5 MB request limit
MAX_SPECTRA = 1200
NC_THRESHOLD = 0.8
HQI_THRESHOLD = 0.8

app = FastAPI(title="Spectra Match API", version="1.0.0")

ORGANISM_MAP = [
    ("ATCC25922", "E. coli ATCC 25922 (non-pathogenic)"),
    ("O103", "E. coli O103:H2 (STEC)"),
    ("O121", "E. coli O121:H19 (STEC)"),
    ("O157", "E. coli O157:H7 (STEC)"),
    ("ATCC700408", "Salmonella (ATCC 700408)"),
    ("Typhimurium", "Salmonella Typhimurium"),
    (" LM", "Listeria monocytogenes"),
    (" EC", "E. coli"),
    (" SE ", "Salmonella enterica"),
]


@lru_cache(maxsize=1)
def load_reference() -> dict:
    data = joblib.load(REFERENCE_FILE)
    required = {"reference_axis", "combined_data", "combined_data_hqi", "dataset_labels"}
    missing = required.difference(data)
    if missing:
        raise RuntimeError(f"Reference file is missing keys: {sorted(missing)}")
    return data


def resolve_organism(label: str) -> str:
    label = str(label)
    for token, organism in ORGANISM_MAP:
        if token in label:
            return organism
    return label.replace(".csv", "")[:60]


def clean_name(name: str) -> str:
    value = Path(name).stem
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "spectrum"


def parse_spectrum_file(content: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        text = content.decode("latin-1")
    except Exception as exc:
        raise ValueError("The uploaded file could not be decoded.") from exc

    lines = [line for line in text.splitlines() if not line.strip().startswith("#")]
    if not lines:
        raise ValueError("The file contains no tabular data after comment lines are removed.")

    raw_text = "\n".join(lines)
    raw = pd.read_csv(io.StringIO(raw_text), sep="\t", header=None, dtype=str)

    header_row = None
    for row_index, row in raw.iterrows():
        numeric_tail = pd.to_numeric(row.iloc[2:], errors="coerce")
        if len(numeric_tail) > 0 and numeric_tail.notna().all():
            header_row = row_index
            break
    if header_row is None:
        raise ValueError("Could not detect the wavenumber header row.")

    df = pd.read_csv(io.StringIO(raw_text), sep="\t", skiprows=header_row, header=0)
    start_idx = None
    for i, column in enumerate(df.columns):
        try:
            if float(column) > 0:
                start_idx = i
                break
        except (TypeError, ValueError):
            continue
    if start_idx is None or start_idx < 2:
        raise ValueError("Could not find two coordinate columns before the spectral columns.")

    df = df.iloc[:, start_idx - 2 :].copy()
    df.columns = ["X", "Y", *df.columns[2:]]
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")

    spectra = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
    valid = spectra.notna().all(axis=1) & spectra.ne(0).any(axis=1) & df[["X", "Y"]].notna().all(axis=1)
    spectra = spectra.loc[valid]
    coords = df.loc[valid, ["X", "Y"]]

    if spectra.empty:
        raise ValueError("No valid non-zero spectra were found.")
    if len(spectra) > MAX_SPECTRA:
        raise ValueError(f"This deployment accepts at most {MAX_SPECTRA} spectra per request.")

    try:
        spectra.columns = spectra.columns.astype(float)
    except ValueError as exc:
        raise ValueError("Spectral column names must be numeric wavenumbers.") from exc
    return coords.reset_index(drop=True), spectra.reset_index(drop=True)


def figure_data_uri(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def make_figures(reference_axis, unknown_nc, ref_nc, x, y, max_nc, max_hqi, best_idx):
    figures: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    mask = max_nc > 0.6
    ax.scatter(x, y, s=36, marker="s", alpha=0.35)
    ax.scatter(x[mask], y[mask], s=58, marker="s")
    ax.set(title="Spatial map: Max NC > 0.6", xlabel="X", ylabel="Y")
    figures["spatial"] = figure_data_uri(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(max_nc, bins=min(40, max(10, len(max_nc) // 5)), alpha=0.75, label="Max NC")
    ax.hist(max_hqi, bins=min(40, max(10, len(max_hqi) // 5)), alpha=0.55, label="Max HQI")
    ax.axvline(NC_THRESHOLD, linestyle="--", linewidth=1)
    ax.set(title="Score distributions", xlabel="Score", ylabel="Spectra")
    ax.legend()
    figures["distribution"] = figure_data_uri(fig)

    best_flat = int(np.argmax(max_nc))
    ref_i = int(best_idx[best_flat])
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    ax.plot(reference_axis, unknown_nc[best_flat], label=f"Unknown point {best_flat}")
    ax.plot(reference_axis, ref_nc[ref_i], linestyle="--", label="Best reference")
    ax.set(title=f"Best overall NC match: {max_nc[best_flat]:.4f}", xlabel="Raman shift", ylabel="Intensity")
    ax.legend(fontsize=8)
    figures["overlay"] = figure_data_uri(fig)
    return figures


@app.get("/", response_class=HTMLResponse)
def home() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/api/health")
def health() -> dict:
    ref = load_reference()
    return {
        "ok": True,
        "reference_spectra": int(np.asarray(ref["combined_data"]).shape[0]),
        "reference_points": int(np.asarray(ref["reference_axis"]).shape[0]),
    }


@app.post("/api/match")
async def match(file: Annotated[UploadFile, File(description="Tab-separated Raman spectrum text file")]):
    filename = file.filename or "spectrum.txt"
    if not filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Please upload a .txt file.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is too large for this Vercel deployment (maximum 4 MB).")

    try:
        coords, spectra = parse_spectrum_file(content)
        ref = load_reference()
        axis = np.asarray(ref["reference_axis"], dtype=float)
        ref_nc = np.asarray(ref["combined_data"], dtype=float)
        ref_hqi = np.asarray(ref["combined_data_hqi"], dtype=float)
        labels = np.asarray(ref["dataset_labels"], dtype=str)

        corrected = baseline_correction(spectra)
        unknown_nc = interpolate_to_reference(
            pd.DataFrame(snv_normalization(corrected.to_numpy()), columns=corrected.columns), axis
        ).to_numpy()
        unknown_hqi = interpolate_to_reference(
            pd.DataFrame(vector_normalization(corrected.to_numpy()), columns=corrected.columns), axis
        ).to_numpy()

        nc_scores = cosine_similarity_matrix(unknown_nc, ref_nc)
        hqi_scores = hqi_similarity_matrix(unknown_hqi, ref_hqi)
        max_nc = nc_scores.max(axis=1)
        mean_nc = nc_scores.mean(axis=1)
        max_hqi = hqi_scores.max(axis=1)
        best_nc_idx = nc_scores.argmax(axis=1)
        best_hqi_idx = hqi_scores.argmax(axis=1)
        best_nc_labels = labels[best_nc_idx]
        best_hqi_labels = labels[best_hqi_idx]
        organisms_nc = np.array([resolve_organism(x) for x in best_nc_labels])
        organisms_hqi = np.array([resolve_organism(x) for x in best_hqi_labels])
        qualifying = (max_nc > NC_THRESHOLD) & (max_hqi > HQI_THRESHOLD)

        result_df = pd.DataFrame({
            "X": coords["X"], "Y": coords["Y"], "Max_NC": max_nc,
            "Mean_NC": mean_nc, "Max_HQI": max_hqi,
            "Best_NC_Match": best_nc_labels, "Best_HQI_Match": best_hqi_labels,
            "Organism_NC": organisms_nc, "Organism_HQI": organisms_hqi,
            "Qualifying": qualifying,
        })

        q_orgs = organisms_nc[qualifying]
        dominant = None
        dominant_fraction = 0.0
        agreement = 0.0
        confidence = "NO MATCH"
        if qualifying.any():
            values, counts = np.unique(q_orgs, return_counts=True)
            order = np.argsort(counts)[::-1]
            dominant = str(values[order][0])
            dominant_fraction = float(counts[order][0] / qualifying.sum())
            agreement = float(np.mean(organisms_nc[qualifying] == organisms_hqi[qualifying]))
            median_hqi = float(np.median(max_hqi[qualifying]))
            confidence = (
                "HIGH" if dominant_fraction >= 0.70 and median_hqi >= 0.90 and agreement >= 0.70
                else "MEDIUM" if dominant_fraction >= 0.50 and median_hqi >= 0.80
                else "LOW"
            )

        figures = make_figures(
            axis, unknown_nc, ref_nc, coords["X"].to_numpy(), coords["Y"].to_numpy(),
            max_nc, max_hqi, best_nc_idx,
        )

        top_rows = result_df.sort_values(["Max_NC", "Max_HQI"], ascending=False).head(100)
        payload = {
            "sample": clean_name(filename),
            "summary": {
                "total_spectra": int(len(result_df)),
                "qualifying_spectra": int(qualifying.sum()),
                "rejected_spectra": int((~qualifying).sum()),
                "dominant_organism": dominant,
                "dominant_fraction": dominant_fraction,
                "nc_hqi_organism_agreement": agreement,
                "confidence": confidence,
                "median_qualifying_nc": float(np.median(max_nc[qualifying])) if qualifying.any() else None,
                "median_qualifying_hqi": float(np.median(max_hqi[qualifying])) if qualifying.any() else None,
                "nc_threshold": NC_THRESHOLD,
                "hqi_threshold": HQI_THRESHOLD,
            },
            "figures": figures,
            "rows": json.loads(result_df.to_json(orient="records")),
            "top_matches": json.loads(top_rows.to_json(orient="records")),
        }
        return JSONResponse(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

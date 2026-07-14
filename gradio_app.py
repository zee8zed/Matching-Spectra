#!/usr/bin/env python
# coding: utf-8
#
# GRADIO WEB APP for spectra_match.py
#
# Wraps the original spectra_match.py matching logic in a simple Gradio
# interface:
#   - Upload the unknown .txt spectrum file with a normal click-to-upload
#     widget (instead of typing a file path)
#   - Click "Choose output folder..." to pop up a native OS folder-picker
#     (same tkinter dialog used by the original script) and pick where the
#     HTML report + CSV/PNG/TXT outputs are saved
#   - Click "Run Matching" to process the file
#
# Run with:   python gradio_app.py
# Requires preprocessing_utils.py and reference_data.pkl in this same
# folder, and the gradio package installed (pip install gradio).

#!/usr/bin/env python
# coding: utf-8
#
# TERMINAL SCRIPT.  Run from a command prompt in the same folder as
# preprocessing_utils.py and reference_data.pkl:
#
#     python spectra_match.py
#
# It prompts for the unknown .txt file, runs NC + HQI matching, saves
# ALL outputs (CSV / PNG / TXT + the HTML report) into a per-sample
# subfolder under DEFAULT_OUTPUT_DIR, then opens ONLY the HTML report
# in the default browser.  No other files pop open.

import os
import re
import numpy as np
import pandas as pd
import joblib
from io import StringIO
import matplotlib
matplotlib.use("Agg")            # headless backend: no figure windows pop up
import matplotlib.pyplot as plt
import gradio as gr

# ---------------------------------------------------------
#                     OUTPUT LOCATION
# ---------------------------------------------------------
# Everything is written here, inside a subfolder named after the sample.
# Edit this ONE line to send outputs somewhere else, e.g.
#   DEFAULT_OUTPUT_DIR = r"D:\Spectra_Results"
DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Spectra_Match_Outputs")


# ---------------------------------------------------------
#              ASK USER WHERE TO SAVE (folder picker)
# ---------------------------------------------------------
def choose_output_dir(default_dir):
    """Pop up a native folder-picker so the user chooses the save location.
    Falls back to default_dir if the dialog is cancelled or unavailable
    (e.g. no GUI available)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()                     # hide the empty tk root window
        root.attributes("-topmost", True)   # bring the dialog in front of the console
        chosen = filedialog.askdirectory(
            initialdir=default_dir if os.path.isdir(default_dir) else os.path.expanduser("~"),
            title="Choose a folder to save the spectra-match outputs",
        )
        root.destroy()
        return chosen if chosen else default_dir
    except Exception as e:
        print(f"(Folder picker unavailable: {e}) -- using default location.")
        return default_dir

# ----- Load preprocessing utilities -----
from preprocessing_utils import (
    baseline_correction,
    snv_normalization,
    interpolate_to_reference,
    normalized_correlation,
    vector_normalization,
    hqi,
)


def clean_filename(name):
    """Remove/replace illegal Windows characters so files and folders save safely."""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.replace(" ", "_").replace("%", "")
    name = name.replace("(", "").replace(")", "")
    return name


# ---------------------------------------------------------
#                LOAD REFERENCE DATA
# ---------------------------------------------------------
ref = joblib.load("reference_data.pkl")

reference_axis = ref["reference_axis"]
combined_nc_array = ref["combined_data"]
combined_hqi_array = ref["combined_data_hqi"]
dataset_labels = ref["dataset_labels"]

print("\nLoaded reference data:")
print("  NC spectra:", combined_nc_array.shape)
print("  HQI spectra:", combined_hqi_array.shape)
print("  Labels:", len(dataset_labels))


def run_spectra_match(txt_file_path, output_dir):


    # ---------------------------------------------------------
    #                INPUT UNKNOWN TXT FILE
    # ---------------------------------------------------------
    # txt_file_path is provided as a function argument (uploaded file path)

    # Ask the user where to save everything (file-explorer popup).
    base_dir = output_dir  # output_dir is provided as a function argument
    os.makedirs(base_dir, exist_ok=True)
    print("Outputs will be saved under:", base_dir)

    with open(txt_file_path, 'r', encoding='latin-1') as f:
        lines = f.readlines()

    filtered_lines = [ln for ln in lines if not ln.strip().startswith("#")]
    data_str = "".join(filtered_lines)

    df_raw = pd.read_csv(StringIO(data_str), sep="\t", header=None)

    # detect header row
    for i, row in df_raw.iterrows():
        if all(str(x).replace(".", "", 1).isdigit() for x in row[2:]):
            header_row = i
            break

    df = pd.read_csv(StringIO(data_str), sep="\t", skiprows=header_row, header=0)

    # get X,Y + spectra columns
    for i, col in enumerate(df.columns):
        try:
            if float(col) > 0:
                wavenumber_start_idx = i
                break
        except:
            continue

    x_col_idx = wavenumber_start_idx - 2
    df = df.iloc[:, x_col_idx:]
    df.columns = ["X", "Y"] + list(df.columns[2:])

    # ---------------------------------------------------------
    #          PREPROCESS UNKNOWN (NC + HQI)
    # ---------------------------------------------------------
    unknown_df = df.iloc[:, 2:]
    unknown_df = unknown_df[(unknown_df != 0).any(axis=1)]

    unknown_corr = baseline_correction(unknown_df)

    # NC preprocessing (SNV path)
    unknown_snv = snv_normalization(unknown_corr.values)
    unknown_interp_nc = interpolate_to_reference(
        pd.DataFrame(unknown_snv, columns=unknown_corr.columns),
        reference_axis
    )

    # HQI preprocessing (vector-norm path)
    unknown_vnorm = vector_normalization(unknown_corr.values)
    unknown_interp_hqi = interpolate_to_reference(
        pd.DataFrame(unknown_vnorm, columns=unknown_corr.columns),
        reference_axis
    )

    # coords (aligned to unknown_df's surviving rows, matches interp/corr row order)
    x_coords = df.loc[unknown_df.index, "X"].values
    y_coords = df.loc[unknown_df.index, "Y"].values

    # ---------------------------------------------------------
    #                NORMALIZED CORRELATION
    # ---------------------------------------------------------
    print("\nComputing NC scores...")

    nc_scores_matrix = np.array([
        [normalized_correlation(u, r) for r in combined_nc_array]
        for u in unknown_interp_nc.values
    ])

    max_nc = nc_scores_matrix.max(axis=1)
    mean_nc = nc_scores_matrix.mean(axis=1)
    best_nc_idx = nc_scores_matrix.argmax(axis=1)
    best_nc_label = [dataset_labels[i] for i in best_nc_idx]

    nc_df = pd.DataFrame({
        "Max_NC": max_nc,
        "Mean_NC": mean_nc
    })

    # ---------------------------------------------------------
    #                      HQI
    # ---------------------------------------------------------
    print("\nComputing HQI scores...")

    hqi_scores_matrix = np.array([
        [hqi(u, r) for r in combined_hqi_array]
        for u in unknown_interp_hqi.values
    ])

    max_hqi = hqi_scores_matrix.max(axis=1)
    best_hqi_idx = hqi_scores_matrix.argmax(axis=1)
    best_hqi_label = [dataset_labels[i] for i in best_hqi_idx]

    # ---------------------------------------------------------
    #          NC DISTRIBUTIONS (TEXT SUMMARY)
    # ---------------------------------------------------------
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels_bin = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]

    nc_df["Max_NC_Range"] = pd.cut(nc_df["Max_NC"], bins=bins, labels=labels_bin, include_lowest=True)
    nc_df["Mean_NC_Range"] = pd.cut(nc_df["Mean_NC"], bins=bins, labels=labels_bin, include_lowest=True)

    max_range_counts = nc_df["Max_NC_Range"].value_counts().sort_index()
    mean_range_counts = nc_df["Mean_NC_Range"].value_counts().sort_index()

    # ---------------------------------------------------------
    #                 HEATMAP PLOT
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    mask = max_nc > 0.6
    plt.scatter(x_coords, y_coords, color="lightgray", s=150, marker="s")
    plt.scatter(x_coords[mask], y_coords[mask], color="red", s=250, marker="s")
    plt.title("Spectra with Max NC > 0.6")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.tight_layout()

    # ---------------------------------------------------------
    #                 TOP 5 MATCHES
    # ---------------------------------------------------------
    flat_sorted = np.argsort(nc_scores_matrix.ravel())[::-1]
    top5 = np.unravel_index(flat_sorted[:5], nc_scores_matrix.shape)
    top5_pairs = list(zip(top5[0], top5[1]))

    # base_dir was chosen via the folder picker near the top of the script.

    raw_name = os.path.splitext(os.path.basename(txt_file_path))[0]
    file_base_name = clean_filename(raw_name)

    save_dir = os.path.join(base_dir, file_base_name)
    os.makedirs(save_dir, exist_ok=True)

    overlay_paths = []

    print("\n===== TOP 5 MATCHES =====")
    for rank, (u_idx, r_idx) in enumerate(top5_pairs, 1):
        unknown_spec = unknown_interp_nc.iloc[u_idx].values
        ref_spec = combined_nc_array[r_idx]
        label = dataset_labels[r_idx]
        score = nc_scores_matrix[u_idx][r_idx]

        x = x_coords[u_idx]; y = y_coords[u_idx]

        print(f"Top {rank}: Unknown#{u_idx} -> Reference#{r_idx} ({label}), NC={score:.4f}")

        plt.figure(figsize=(10, 4))
        plt.plot(reference_axis, unknown_spec, color="black",
                 label=f"Unknown#{u_idx} (X={x:.1f},Y={y:.1f})")
        plt.plot(reference_axis, ref_spec, color="blue",
                 label=f"Reference: {label}")
        plt.title(f"Top {rank} Match (NC={score:.4f})")
        plt.xlabel("Wavenumber"); plt.ylabel("Intensity")
        plt.legend()

        clean_out = clean_filename(f"Top{rank}_Match_{file_base_name}.png")
        out = os.path.join(save_dir, clean_out)

        overlay_paths.append(out)
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()

    # ---------------------------------------------------------
    #      HIGH-PROBABILITY BACTERIA EXPORT
    #      (Max_NC > 0.8 AND Max_HQI > 0.8)
    #      Exports BOTH sides of the split (qualifying + rejected),
    #      each in raw and baseline-corrected form, plus the two
    #      interpolated (500-pt) forms for the qualifying set.
    # ---------------------------------------------------------
    NC_THRESHOLD = 0.8
    HQI_THRESHOLD = 0.8

    qualifying_mask = (max_nc > NC_THRESHOLD) & (max_hqi > HQI_THRESHOLD)
    rejected_mask = ~qualifying_mask
    n_qualifying = int(qualifying_mask.sum())
    n_rejected = int(rejected_mask.sum())
    print(f"\n{n_qualifying} / {len(qualifying_mask)} spectra pass Max_NC > {NC_THRESHOLD} "
          f"AND Max_HQI > {HQI_THRESHOLD}  ({n_rejected} rejected)")


    def _build_export(source_df, mask, extra_cols):
        """Slice source_df by mask, prepend X/Y, then insert extra named columns."""
        idx = mask.nonzero()[0]
        out = source_df.iloc[idx].copy()
        out.insert(0, "Y", y_coords[mask])
        out.insert(0, "X", x_coords[mask])
        col_pos = 2
        for col_name, values in extra_cols.items():
            out.insert(col_pos, col_name, values[mask])
            col_pos += 1
        return out


    export_paths = {}

    if n_qualifying > 0:
        # --- 1) RAW, qualifying (non-baseline-corrected) ---
        raw_qualifying = _build_export(
            unknown_df, qualifying_mask, {"Max_NC": max_nc, "Max_HQI": max_hqi}
        )
        export_paths["raw_qualifying"] = os.path.join(
            save_dir, clean_filename(f"{file_base_name}_Raw_Qualifying_Spectra.csv")
        )
        raw_qualifying.to_csv(export_paths["raw_qualifying"], index=False)

        # --- 2) BASELINE-CORRECTED, qualifying (original axis, not interpolated) ---
        baseline_qualifying = _build_export(
            unknown_corr, qualifying_mask, {"Max_NC": max_nc, "Max_HQI": max_hqi}
        )
        export_paths["baseline_qualifying"] = os.path.join(
            save_dir, clean_filename(f"{file_base_name}_BaselineCorrected_Qualifying_Spectra.csv")
        )
        baseline_qualifying.to_csv(export_paths["baseline_qualifying"], index=False)

        # --- 3) Interpolated 500-pt qualifying spectra: NC path (SNV) ---
        interp_nc_qualifying = _build_export(
            unknown_interp_nc, qualifying_mask,
            {"Max_NC": max_nc, "Best_NC_Match": np.array(best_nc_label)}
        )
        export_paths["interp_nc_qualifying"] = os.path.join(
            save_dir, clean_filename(f"{file_base_name}_Interpolated_NC_Qualifying_Spectra.csv")
        )
        interp_nc_qualifying.to_csv(export_paths["interp_nc_qualifying"], index=False)

        # --- 4) Interpolated 500-pt qualifying spectra: HQI path (vector-norm) ---
        interp_hqi_qualifying = _build_export(
            unknown_interp_hqi, qualifying_mask,
            {"Max_HQI": max_hqi, "Best_HQI_Match": np.array(best_hqi_label)}
        )
        export_paths["interp_hqi_qualifying"] = os.path.join(
            save_dir, clean_filename(f"{file_base_name}_Interpolated_HQI_Qualifying_Spectra.csv")
        )
        interp_hqi_qualifying.to_csv(export_paths["interp_hqi_qualifying"], index=False)

        print("Saved qualifying-spectra exports:")
        for p in export_paths.values():
            print(" ", p)
    else:
        print("No spectra passed both thresholds — no qualifying export files created.")

    if n_rejected > 0:
        # --- 5) RAW, rejected (non-baseline-corrected) ---
        raw_rejected = _build_export(
            unknown_df, rejected_mask, {"Max_NC": max_nc, "Max_HQI": max_hqi}
        )
        export_paths["raw_rejected"] = os.path.join(
            save_dir, clean_filename(f"{file_base_name}_Raw_Rejected_Spectra.csv")
        )
        raw_rejected.to_csv(export_paths["raw_rejected"], index=False)

        # --- 6) BASELINE-CORRECTED, rejected (original axis, not interpolated) ---
        baseline_rejected = _build_export(
            unknown_corr, rejected_mask, {"Max_NC": max_nc, "Max_HQI": max_hqi}
        )
        export_paths["baseline_rejected"] = os.path.join(
            save_dir, clean_filename(f"{file_base_name}_BaselineCorrected_Rejected_Spectra.csv")
        )
        baseline_rejected.to_csv(export_paths["baseline_rejected"], index=False)

        # --- 7) Interpolated 500-pt rejected spectra: NC path (SNV) ---
        interp_nc_rejected = _build_export(
            unknown_interp_nc, rejected_mask,
            {"Max_NC": max_nc, "Best_NC_Match": np.array(best_nc_label)}
        )
        export_paths["interp_nc_rejected"] = os.path.join(
            save_dir, clean_filename(f"{file_base_name}_Interpolated_NC_Rejected_Spectra.csv")
        )
        interp_nc_rejected.to_csv(export_paths["interp_nc_rejected"], index=False)

        # --- 8) Interpolated 500-pt rejected spectra: HQI path (vector-norm) ---
        interp_hqi_rejected = _build_export(
            unknown_interp_hqi, rejected_mask,
            {"Max_HQI": max_hqi, "Best_HQI_Match": np.array(best_hqi_label)}
        )
        export_paths["interp_hqi_rejected"] = os.path.join(
            save_dir, clean_filename(f"{file_base_name}_Interpolated_HQI_Rejected_Spectra.csv")
        )
        interp_hqi_rejected.to_csv(export_paths["interp_hqi_rejected"], index=False)

        print("Saved rejected-spectra exports:")
        print(" ", export_paths["raw_rejected"])
        print(" ", export_paths["baseline_rejected"])
        print(" ", export_paths["interp_nc_rejected"])
        print(" ", export_paths["interp_hqi_rejected"])
    else:
        print("No spectra were rejected — no rejected export files created.")

    # ---------------------------------------------------------
    #               FULL SUMMARY FILE OUTPUT
    # ---------------------------------------------------------
    summary_path = os.path.join(save_dir, f"{file_base_name}_NC_HQI_Results.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"File: {file_base_name}.txt\n\n")

        f.write("===== Normalized Correlation Score Distribution (Max NC) =====\n")
        for L, C in max_range_counts.items():
            f.write(f"Range {L}: {C} spectra\n")

        f.write("\n===== Normalized Correlation Score Distribution (Mean NC) =====\n")
        for L, C in mean_range_counts.items():
            f.write(f"Range {L}: {C} spectra\n")

        f.write("\n===== HQI Results Summary =====\n")
        f.write(f"Max HQI: {max_hqi.max():.4f}\n")
        f.write(f"Mean HQI: {max_hqi.mean():.4f}\n")
        f.write(f"Spectra with HQI >= 0.9: {np.sum(max_hqi >= 0.9)}\n")

        f.write("\n===== High-Probability Bacteria Export =====\n")
        f.write(f"Threshold: Max_NC > {NC_THRESHOLD} AND Max_HQI > {HQI_THRESHOLD}\n")
        f.write(f"Qualifying spectra: {n_qualifying} / {len(qualifying_mask)}\n")
        f.write(f"Rejected spectra: {n_rejected} / {len(qualifying_mask)}\n")
        if n_qualifying > 0:
            f.write(f"Raw (qualifying) export: {os.path.basename(export_paths['raw_qualifying'])}\n")
            f.write(f"Baseline-corrected (qualifying) export: {os.path.basename(export_paths['baseline_qualifying'])}\n")
            f.write(f"Interpolated NC-path (qualifying) export: {os.path.basename(export_paths['interp_nc_qualifying'])}\n")
            f.write(f"Interpolated HQI-path (qualifying) export: {os.path.basename(export_paths['interp_hqi_qualifying'])}\n")
        if n_rejected > 0:
            f.write(f"Raw (rejected) export: {os.path.basename(export_paths['raw_rejected'])}\n")
            f.write(f"Baseline-corrected (rejected) export: {os.path.basename(export_paths['baseline_rejected'])}\n")
            f.write(f"Interpolated NC-path (rejected) export: {os.path.basename(export_paths['interp_nc_rejected'])}\n")
            f.write(f"Interpolated HQI-path (rejected) export: {os.path.basename(export_paths['interp_hqi_rejected'])}\n")

        f.write("\n===== Overlay Plot Info =====\n")
        best_u, best_r = top5_pairs[0]
        best_nc = nc_scores_matrix[best_u][best_r]
        f.write(f"Best NC match -> Unknown #{best_u}, Reference #{best_r} ({dataset_labels[best_r]}), NC={best_nc:.4f}\n")

    # ---------------------------------------------------------
    #                   SAVE ALL PLOTS
    # ---------------------------------------------------------
    nc_hist_path = os.path.join(save_dir, f"NC_Hist_{file_base_name}.png")
    hqi_hist_path = os.path.join(save_dir, f"HQI_Hist_{file_base_name}.png")
    heatmap_path = os.path.join(save_dir, f"Heatmap_{file_base_name}.png")

    # NC histogram
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(max_nc, bins=50, color="dodgerblue", edgecolor="black")
    plt.axvline(0.8, color="red", linestyle="--")
    plt.title("Max Normalized Correlation")

    plt.subplot(1, 2, 2)
    plt.hist(mean_nc, bins=50, color="seagreen", edgecolor="black")
    plt.axvline(0.6, color="red", linestyle="--")
    plt.title("Mean Normalized Correlation")
    plt.tight_layout()
    plt.savefig(nc_hist_path, dpi=120)
    plt.close()

    # HQI histogram
    plt.figure(figsize=(8, 5))
    plt.hist(max_hqi, bins=50, color="steelblue", edgecolor="black")
    plt.axvline(0.8, color="red", linestyle="--")
    plt.title("HQI Distribution")
    plt.savefig(hqi_hist_path, dpi=120)
    plt.close()

    # Heatmap save
    plt.figure(figsize=(8, 5))
    plt.scatter(x_coords, y_coords, color="lightgray", s=150, marker="s")
    plt.scatter(x_coords[mask], y_coords[mask], color="red", s=250, marker="s")
    plt.title("Spectra with Max NC > 0.6")
    plt.savefig(heatmap_path, dpi=120)
    plt.close()

    # ---------------------------------------------------------
    #            (individual auto-open removed)
    #   Per request: the CSV / PNG / TXT files are saved silently.
    #   Only the HTML report is opened, at the very end of the script.
    # ---------------------------------------------------------
    print("\nAll outputs saved in:", save_dir)


    ###-----------ADDITION------------------###

    # ============================================================
    #  QUALIFYING-SPECTRA HTML REPORT
    #  Written by Claude
    #  ------------------------------------------------------------
    #  Builds the self-contained <sample>_Report.html from the values
    #  computed above and opens it in the browser.
    # ============================================================

    import base64, html
    from io import BytesIO
    from datetime import datetime

    # ---------------------------------------------------------------
    #  ORGANISM NAME MAP  ->  edit / verify the three flagged lines.
    #  First matching substring wins, so keep specific keys above
    #  generic ones. Unmatched labels fall back to a cleaned filename.
    # ---------------------------------------------------------------
    ORGANISM_MAP = [
        ("ATCC25922",   "E. coli ATCC 25922 (non-pathogenic)"),
        ("O103",        "E. coli O103:H2 (STEC)"),
        ("O121",        "E. coli O121:H19 (STEC)"),
        ("O157",        "E. coli O157:H7 (STEC)"),
        ("ATCC700408",  "Salmonella (ATCC 700408)"),
        ("Typhimurium", "Salmonella Typhimurium"),
        (" LM",         "Listeria monocytogenes"),        # <-- verify (LM token)
        (" EC",         "E. coli (generic EC)"),          # <-- verify (EC token)
        (" SE ",        "Sample 'SE' - VERIFY organism"), # <-- unsure what SE is
    ]

    def resolve_organism(lbl):
        for key, name in ORGANISM_MAP:
            if key in lbl:
                return name
        return lbl.replace(".csv", "")[:40]

    # ---------------------------------------------------------------
    #  Small helpers
    # ---------------------------------------------------------------
    def _png_b64(path):
        if not path or not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            return "data:image/png;base64," + base64.b64encode(fh.read()).decode()

    def _fig_b64(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig); buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    # ---------------------------------------------------------------
    #  Pull what we need out of the in-memory run
    # ---------------------------------------------------------------
    qmask   = np.asarray(qualifying_mask)
    n_qual  = int(qmask.sum())
    n_total = int(qmask.size)

    nc_all  = np.asarray(max_nc)
    hqi_all = np.asarray(max_hqi)

    # per-qualifying-spectrum organism (NC path is the script's primary path)
    org_nc_all  = np.array([resolve_organism(l) for l in best_nc_label])
    org_hqi_all = np.array([resolve_organism(l) for l in best_hqi_label])

    sections = []  # list of (title, html_chunk)

    # ---------------------------------------------------------------
    #  VERDICT (plain language, colleague-facing)
    # ---------------------------------------------------------------
    if n_qual > 0:
        q_org_nc  = org_nc_all[qmask]
        q_org_hqi = org_hqi_all[qmask]
        q_nc      = nc_all[qmask]
        q_hqi     = hqi_all[qmask]

        orgs, counts = np.unique(q_org_nc, return_counts=True)
        order = np.argsort(counts)[::-1]
        orgs, counts = orgs[order], counts[order]
        dominant  = orgs[0]
        dom_frac  = counts[0] / n_qual
        agree_frac = float(np.mean(q_org_nc == q_org_hqi))  # NC vs HQI identity agreement

        conf = ("HIGH"   if dom_frac >= 0.70 and np.median(q_hqi) >= 0.90 and agree_frac >= 0.70 else
                "MEDIUM" if dom_frac >= 0.50 and np.median(q_hqi) >= 0.80 else
                "LOW")
        conf_color = {"HIGH": "#1a7f37", "MEDIUM": "#bf8700", "LOW": "#b42318"}[conf]

        verdict = (
            f'<div class="verdict" style="border-left-color:{conf_color}">'
            f'<div class="conf" style="color:{conf_color}">Confidence: {conf}</div>'
            f'<p><b>{n_qual}</b> of <b>{n_total}</b> measured spectra passed both quality '
            f'thresholds (NC&gt;{NC_THRESHOLD} and HQI&gt;{HQI_THRESHOLD}).</p>'
            f'<p>Most likely identity: <b>{html.escape(str(dominant))}</b> '
            f'&mdash; <b>{dom_frac*100:.0f}%</b> of the passing spectra point to it.</p>'
            f'<p>The two independent scores (NC and HQI) agree on the organism for '
            f'<b>{agree_frac*100:.0f}%</b> of passing spectra.</p>'
            f'<p>Median NC among passing spectra: <b>{np.median(q_nc):.3f}</b> &nbsp;|&nbsp; '
            f'Median HQI: <b>{np.median(q_hqi):.3f}</b></p>'
            f'</div>'
        )
    else:
        verdict = ('<div class="verdict" style="border-left-color:#b42318">'
                   '<div class="conf" style="color:#b42318">No confident matches</div>'
                   f'<p>0 of {n_total} spectra passed both thresholds '
                   f'(NC&gt;{NC_THRESHOLD} and HQI&gt;{HQI_THRESHOLD}). '
                   'Nothing in this sample matched the reference library strongly enough '
                   'to call an identity.</p></div>')
    sections.append(("Summary", verdict))

    # ---------------------------------------------------------------
    #  Figures that need the in-memory arrays (only if we have hits)
    # ---------------------------------------------------------------
    if n_qual > 0:
        # palette shared across the organism figures
        present = list(orgs)
        cmap = plt.get_cmap("tab10")
        color_of = {o: cmap(i % 10) for i, o in enumerate(present)}

        # --- (A) organism breakdown bar chart -----------------------
        fig, ax = plt.subplots(figsize=(8, max(2.4, 0.5 * len(orgs))))
        ypos = np.arange(len(orgs))[::-1]
        ax.barh(ypos, counts, color=[color_of[o] for o in orgs])
        ax.set_yticks(ypos); ax.set_yticklabels(orgs)
        ax.set_xlabel("Number of passing spectra")
        ax.set_title("Which organism the passing spectra matched")
        for y, c in zip(ypos, counts):
            ax.text(c, y, f" {c}", va="center")
        sections.append(("Identity breakdown", f'<img src="{_fig_b64(fig)}">'))

        # --- (B) mean passing spectrum vs dominant reference --------
        try:
            S = unknown_interp_nc.values[qmask]            # (n_qual, 500) SNV/interp
            m, sd = S.mean(0), S.std(0)
            dom_ref_mask = np.array([resolve_organism(l) == dominant for l in dataset_labels])
            ref_mean = combined_nc_array[dom_ref_mask].mean(0)   # same SNV/interp space

            fig, ax = plt.subplots(figsize=(9, 3.6))
            ax.fill_between(reference_axis, m - sd, m + sd, alpha=0.22,
                            color="#4c78a8", label="sample ±1 std")
            ax.plot(reference_axis, m, color="#1f3b57", lw=1.4, label="sample mean")
            ax.plot(reference_axis, ref_mean, color="#c44e52", lw=1.2, ls="--",
                    label=f"reference: {dominant}")
            ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_ylabel("SNV intensity (a.u.)")
            ax.set_title(f"Mean of {n_qual} passing spectra vs. matched reference")
            ax.legend(fontsize=8)
            sections.append(("Sample vs. reference", f'<img src="{_fig_b64(fig)}">'))
        except Exception as e:
            sections.append(("Sample vs. reference",
                             f'<p class="warn">Skipped: {html.escape(str(e))}</p>'))

        # --- (C) NC vs HQI scatter, colored by organism ------------
        fig, ax = plt.subplots(figsize=(6.2, 5))
        for o in present:
            sel = q_org_nc == o
            ax.scatter(q_nc[sel], q_hqi[sel], s=28, alpha=0.75,
                       color=color_of[o], label=o, edgecolor="white", linewidth=0.4)
        ax.axvline(NC_THRESHOLD, color="gray", ls=":", lw=1)
        ax.axhline(HQI_THRESHOLD, color="gray", ls=":", lw=1)
        ax.set_xlabel("Max NC"); ax.set_ylabel("Max HQI")
        ax.set_title("Score cluster of passing spectra")
        ax.legend(fontsize=7, loc="lower left")
        sections.append(("Score cluster", f'<img src="{_fig_b64(fig)}">'))

    # ---------------------------------------------------------------
    #  Embed the PNGs the main script already saved
    # ---------------------------------------------------------------
    gallery = ""
    for path, cap in [(heatmap_path, "Spatial map (NC &gt; 0.6)"),
                      (nc_hist_path, "NC distribution"),
                      (hqi_hist_path, "HQI distribution")]:
        b = _png_b64(path)
        if b:
            gallery += f'<figure><img src="{b}"><figcaption>{cap}</figcaption></figure>'
    if gallery:
        sections.append(("Distributions &amp; spatial map", f'<div class="gallery">{gallery}</div>'))

    overlays = ""
    for p in overlay_paths:
        b = _png_b64(p)
        if b:
            tag = os.path.basename(p).split("_Match_")[0]
            overlays += f'<figure><img src="{b}"><figcaption>{tag}</figcaption></figure>'
    if overlays:
        sections.append(("Top-5 match overlays", f'<div class="gallery">{overlays}</div>'))

    # ---------------------------------------------------------------
    #  Per-spectrum table (passing spectra only)
    # ---------------------------------------------------------------
    if n_qual > 0:
        idx = np.where(qmask)[0]
        rows = sorted(zip(x_coords[qmask], y_coords[qmask], q_nc, q_hqi, q_org_nc),
                      key=lambda r: r[3], reverse=True)
        trs = "".join(
            f"<tr><td>{x:.1f}</td><td>{y:.1f}</td><td>{nc:.4f}</td>"
            f"<td>{hq:.4f}</td><td style='text-align:left'>{html.escape(str(o))}</td></tr>"
            for x, y, nc, hq, o in rows)
        table = ('<table class="tbl"><thead><tr><th>X</th><th>Y</th>'
                 '<th>Max NC</th><th>Max HQI</th><th style="text-align:left">Match</th>'
                 f'</tr></thead><tbody>{trs}</tbody></table>')
        sections.append(("Passing spectra (per point)", table))

    # ---------------------------------------------------------------
    #  Assemble the self-contained HTML
    # ---------------------------------------------------------------
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:960px;
         margin:24px auto;padding:0 18px;color:#1a1a1a;line-height:1.5}
    h1{font-size:21px;margin-bottom:2px} .sub{color:#777;font-size:13px;margin-top:0}
    h3{font-size:14px;color:#444;border-bottom:1px solid #eee;padding-bottom:5px;margin-top:34px}
    .verdict{background:#f7f9fb;border-left:6px solid #888;padding:14px 20px;border-radius:6px}
    .conf{font-size:17px;font-weight:700;margin-bottom:6px}
    .gallery{display:flex;flex-wrap:wrap;gap:14px}
    .gallery figure{margin:0;flex:1 1 300px;text-align:center}
    figcaption{font-size:12px;color:#666;margin-top:3px}
    img{max-width:100%;border:1px solid #eee;border-radius:5px}
    .tbl{border-collapse:collapse;font-size:12px;width:100%}
    .tbl th,.tbl td{border:1px solid #e6e6e6;padding:4px 8px;text-align:right}
    .tbl th{background:#f2f4f7} .warn{color:#b42318}
    .foot{color:#aaa;font-size:11px;margin-top:44px}
    """
    body = (f"<h1>Spectral Match Report</h1>"
            f'<p class="sub">Sample: {html.escape(file_base_name)} &nbsp;·&nbsp; '
            f"Generated {datetime.now():%Y-%m-%d %H:%M}</p>")
    for title, chunk in sections:
        body += f"<h3>{title}</h3>{chunk}"
    body += ('<p class="foot">Reference library: 3056 spectra, 22 entries. '
             'Thresholds NC&gt;%.2f and HQI&gt;%.2f. Identity taken from the NC best match; '
             'NC/HQI agreement reported as a cross-check.</p>' % (NC_THRESHOLD, HQI_THRESHOLD))

    doc = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
           f"<title>{html.escape(file_base_name)} report</title>"
           f"<style>{css}</style></head><body>{body}</body></html>")

    report_path = os.path.join(save_dir, f"{file_base_name}_Report.html")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(doc)

    print("HTML report saved ->", report_path)
    return report_path


# ============================================================
#  GRADIO WEB INTERFACE
#  - Upload the .txt file with a normal click-to-upload widget
#  - Click "Choose output folder..." to get a native OS folder-picker popup
#  - Click "Run Matching" to process; the HTML report is saved to that
#    folder and also previewed / downloadable in the browser
# ============================================================

def browse_folder(current_value):
    """Opens the native OS folder-picker (same tkinter dialog as the
    original terminal script) and returns the chosen path."""
    start_dir = current_value if current_value and os.path.isdir(current_value) else DEFAULT_OUTPUT_DIR
    chosen = choose_output_dir(start_dir)
    return chosen


def process(uploaded_file, output_dir):
    if uploaded_file is None:
        raise gr.Error("Please upload a .txt spectrum file first.")
    if not output_dir or not os.path.isdir(output_dir):
        raise gr.Error("Please choose a valid output folder first (click 'Choose output folder...').")

    upload_path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
    report_path = run_spectra_match(upload_path, output_dir)

    with open(report_path, "r", encoding="utf-8") as fh:
        html_content = fh.read()

    return html_content, report_path


with gr.Blocks(title="Spectra Match") as demo:
    gr.Markdown(
        "## Spectra Match\n"
        "Upload an unknown spectrum `.txt` file, choose an output folder, "
        "then run the match. The HTML report will be saved there and shown below."
    )

    file_input = gr.File(label="Unknown spectrum (.txt)", file_types=[".txt"], file_count="single")

    with gr.Row():
        output_dir_box = gr.Textbox(
            label="Output folder",
            placeholder="Click 'Choose output folder...' to pick a save location",
            interactive=False,
            scale=4,
        )
        browse_btn = gr.Button("Choose output folder...", scale=1)

    run_btn = gr.Button("Run Matching", variant="primary")

    report_file = gr.File(label="Download HTML report")
    report_view = gr.HTML(label="Report preview")

    browse_btn.click(fn=browse_folder, inputs=output_dir_box, outputs=output_dir_box)
    run_btn.click(fn=process, inputs=[file_input, output_dir_box], outputs=[report_view, report_file])


if __name__ == "__main__":
    demo.launch()

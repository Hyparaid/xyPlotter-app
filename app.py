
# app_optimized.py
# ------------------------------------------------------------
# Battery Cell Data — Visualizer (Optimized)
# Key improvements vs. original:
# - Upload + parse gated behind a form submit button (prevents N*(N+1)/2 re-parses)
# - Stable caching keyed on (filename + bytes md5) instead of Streamlit UploadedFile object
# - Temp files are deleted after parse (no disk creep)
# - Avoids giant "raw" concatenations unless needed; plots iterate per-file
# - Avoids executing all tabs on every rerun by using a sidebar "View" selector (lazy execution)
# - Fixes selected_sources overwrite bug
# ------------------------------------------------------------

import re
import io
import os
import math
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import warnings

# ----------------------------
# NDAX imports
# ----------------------------
try:
    import NewareNDA as nda
except Exception:
    nda = None

try:
    from ndax_min_builder import build_ndax_df
except Exception:
    build_ndax_df = None


# ----------------------------
# App config
# ----------------------------
CAMERA_CFG = {
    "toImageButtonOptions": {
        "format": "png",
        "filename": "plot",
        "scale": 4
    }
}

st.set_page_config(
    page_title="Battery Cell Data — Visualizer",
    page_icon="🔋",
    layout="wide",
)

HERE = Path(__file__).parent
LOGO_PATH = HERE / "logo.png"

# ----------------------------
# Styling / colors
# ----------------------------
NV_COLORDICT = {
    "white": "#ffffff",
    "black": "#000000",

    "nv_green1": "#9ef3c0",
    "nv_green2": "#81f1c4",
    "nv_green3": "#09f392",
    "nv_green4": "#03a567",
    "nv_green5": "#028133",
    "nv_green6": "#009632",
    "nv_green7": "#1e644b",
    "nv_green8": "#174335",
    "nv_green9": "#0e2922",

    "nv_blue0": "#bcf0fc",
    "nv_blue1": "#79c4c2",
    "nv_blue2": "#2ea2a8",
    "nv_blue3": "#058e91",
    "nv_blue4": "#006663",
    "nv_blue5": "#0f5280",
    "nv_blue6": "#2891CE",
    "nv_blue7": "#094246",
    "nv_blue8": "#034e7b",
    "nv_blue9": "#013b5a",

    "nv_gray1": "#f3f3f3",
    "nv_gray2": "#e2e2e2",
    "nv_gray3": "#c0c0c0",
    "nv_gray4": "#9b9b9b",
    "nv_gray5": "#6e6e6d",
    "nv_gray6": "#b3b3b3",
    "nv_gray7": "#8c8c8c",
    "nv_gray8": "#666666",
    "nv_gray9": "#4d4d4d",
    "nv_gray10": "#333333",

    "nv_power_green": "#44a27a",
    "nv_silver": "#8A8D8F",
    "nv_red": "#d84437",
    "nv_orange": "#f78618",
    "nv_amber": "#f7a600",
}

PALETTES = {
    "Preli long": [
        NV_COLORDICT["nv_power_green"], NV_COLORDICT["nv_blue5"], NV_COLORDICT["nv_blue1"],
        "#e7298a", "#0383a3", "#013824", "#c7e9b4", "#7fcdbb",
        "#d7301f", "#990000", NV_COLORDICT["nv_gray4"], NV_COLORDICT["nv_gray3"],
        NV_COLORDICT["nv_blue8"], "#6C88AD", "#5C9BBA",
        NV_COLORDICT["nv_green5"], NV_COLORDICT["nv_blue9"], NV_COLORDICT["nv_gray8"],
        NV_COLORDICT["nv_amber"], NV_COLORDICT["nv_orange"]
    ],
    "Preli blues": [
        NV_COLORDICT["nv_blue0"], NV_COLORDICT["nv_blue1"], NV_COLORDICT["nv_blue2"],
        NV_COLORDICT["nv_blue3"], NV_COLORDICT["nv_blue4"], NV_COLORDICT["nv_blue5"],
        NV_COLORDICT["nv_blue6"], NV_COLORDICT["nv_blue7"], NV_COLORDICT["nv_blue8"],
        NV_COLORDICT["nv_blue9"],
    ],
    "Preli greens": [
        NV_COLORDICT["nv_green1"], NV_COLORDICT["nv_green2"], NV_COLORDICT["nv_green3"],
        NV_COLORDICT["nv_green4"], NV_COLORDICT["nv_green5"], NV_COLORDICT["nv_green6"],
        NV_COLORDICT["nv_green7"], NV_COLORDICT["nv_green8"], NV_COLORDICT["nv_green9"],
    ],
    "Preli greys": [
        NV_COLORDICT["nv_gray1"], NV_COLORDICT["nv_gray2"], NV_COLORDICT["nv_gray6"],
        NV_COLORDICT["nv_gray3"], NV_COLORDICT["nv_gray7"], NV_COLORDICT["nv_gray4"],
        NV_COLORDICT["nv_gray8"], NV_COLORDICT["nv_gray5"], NV_COLORDICT["nv_gray9"],
        NV_COLORDICT["nv_gray10"],
    ],
    "Preli Warm": [
        "#FFE8A3", "#FFD166", "#F7B733",
        "#F49E4C", "#F07C28", "#F25C05",
        "#E63B2E", "#CC2F27", "#A72222", "#7F1D1D"
    ],
}

warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
    category=FutureWarning,
)

# ----------------------------
# Helpers
# ----------------------------
def style_for_ppt(fig):
    fig.update_layout(
        font=dict(family="Arial", size=16, color="black"),
        margin=dict(l=80, r=40, t=60, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bordercolor="black",
            borderwidth=0.5,
            bgcolor="rgba(255,255,255,0.9)",
        ),
    )
    fig.update_xaxes(
        showline=True, linecolor="black", linewidth=1, mirror=True,
        ticks="outside", tickwidth=1, ticklen=6,
        title_font=dict(family="Arial", size=14, color="black"),
        tickfont=dict(family="Arial", size=14, color="black"),
    )
    fig.update_yaxes(
        showline=True, linecolor="black", linewidth=1, mirror=True,
        ticks="outside", tickwidth=1, ticklen=6,
        title_font=dict(family="Arial", size=14, color="black"),
        tickfont=dict(family="Arial", size=14, color="black"),
    )

def add_ppt_download(fig, filename_base: str):
    buf = io.BytesIO()
    try:
        fig.write_image(buf, format="png", width=1600, height=900, scale=2)
    except Exception:
        st.info("Static image export not available. Install `kaleido` to enable PNG downloads.")
        return
    buf.seek(0)
    st.download_button(
        label="⬇️ Download PNG for PPT",
        data=buf,
        file_name=f"{filename_base}.png",
        mime="image/png",
    )

def pretty_src(src: str) -> str:
    return Path(src).stem

def family_from_filename(name: str) -> str:
    stem = Path(str(name)).stem.lower()
    for sep in ["_", "-", " "]:
        if sep in stem:
            stem = stem.split(sep)[0]
            break
    return re.sub(r"\d+$", "", stem) or stem

def _concat_nonempty(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    keep = []
    cols = pd.Index([])
    for f in frames:
        if isinstance(f, pd.DataFrame):
            cols = cols.union(f.columns)
            if not f.empty and (not f.isna().all().all()):
                keep.append(f)
    if not keep:
        return pd.DataFrame(columns=cols)
    return pd.concat(keep, ignore_index=True, copy=False)

def normalize_neware_headers(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    lower_map = {c.lower(): c for c in d.columns}

    def ensure(target: str, *srcs: str):
        if target in d.columns:
            return
        for s in srcs:
            key = s.lower()
            if key in lower_map:
                d.rename(columns={lower_map[key]: target}, inplace=True)
                lower_map[target.lower()] = target
                break

    ensure("Total Time", "totaltime", "total time")
    ensure("Time", "time(s)", "time")
    ensure("Voltage(V)", "voltage (v)", "voltage")
    ensure("Current(mA)", "current (ma)", "current")

    ensure("Spec. Cap.(mAh/g)", "specific capacity (mAh/g)", "specific capacity (mah/g)", "spec cap (mah/g)")
    ensure("Capacity(mAh)", "capacity (mah)", "capacity")
    ensure("Chg. Spec. Cap.(mAh/g)", "chg. specific capacity (mAh/g)", "chg. spec. cap.(mAh/g)")
    ensure("DChg. Spec. Cap.(mAh/g)", "dchg. specific capacity (mAh/g)", "dchg. spec. cap.(mAh/g)")
    ensure("Chg. Cap.(mAh)", "chg. capacity (mah)")
    ensure("DChg. Cap.(mAh)", "dchg. capacity (mah)")

    ensure("Cycle Index", "cycle", "cycle number", "cycle_index")
    ensure("Step Type", "status", "state", "mode", "step")
    return d

def infer_rest_step(
    df: pd.DataFrame,
    step_col: str = "Step Type",
    current_col: str = "Current(mA)",
    abs_threshold: float = 0.5,
    win: int = 5,
) -> pd.DataFrame:
    d = df.copy()
    if current_col not in d.columns:
        return d
    cur = pd.to_numeric(d[current_col], errors="coerce")
    med = cur.rolling(win, min_periods=1, center=True).median().abs()
    is_rest = med <= abs_threshold
    if step_col in d.columns:
        stype = d[step_col].astype(str)
        mask = (stype == "") | stype.isna()
        d.loc[mask & is_rest, step_col] = "Rest"
    else:
        d[step_col] = np.where(is_rest, "Rest", "")
    return d

TIMESTAMP_CANDIDATES = [
    "Timestamp", "TimeStamp", "Record Time", "RecordTime", "Date Time", "DateTime",
    "datetime", "Measured Time", "MeasuredTime", "Test Time", "TestTime", "Start Time", "StartTime",
    "System Time", "Local Time"
]

def pick_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    for c in TIMESTAMP_CANDIDATES:
        if c in df.columns:
            return c
    lower = {col.lower(): col for col in df.columns}
    for c in TIMESTAMP_CANDIDATES:
        if c.lower() in lower:
            return lower[c.lower()]
    # heuristic: any column with many parseable datetimes
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce")
        if s.notna().mean() > 0.8:
            return c
    return None

def build_global_time_seconds(
    df: pd.DataFrame,
    time_col: Optional[str],
    cycle_col: str = "Cycle Index",
    step_col: str  = "Step Type",
) -> pd.Series:
    d = df

    # 1) Absolute timestamp
    ts_col = pick_timestamp_column(d)
    if ts_col:
        t = pd.to_datetime(d[ts_col], errors="coerce")
        if t.notna().any():
            return (t - t.iloc[0]).dt.total_seconds()

    # 2) Total Time (HH:MM:SS-like)
    if time_col and time_col in d.columns and time_col.lower() == "total time":
        td = pd.to_timedelta(d[time_col].astype(str), errors="coerce")
        return (td - td.iloc[0]).dt.total_seconds()

    # 3) Stitch step-local 'Time'
    raw = d[time_col] if (time_col and time_col in d.columns) else None
    if raw is None:
        return pd.Series(np.zeros(len(d)), index=d.index, dtype="float64")

    if cycle_col in d.columns and step_col in d.columns:
        gkey = d[cycle_col].astype("Int64").astype(str) + "|" + d[step_col].astype(str)
    elif cycle_col in d.columns:
        gkey = d[cycle_col].astype("Int64").astype(str)
    elif step_col in d.columns:
        gkey = d[step_col].astype(str)
    else:
        gkey = pd.Series(range(len(d)), index=d.index)

    sec = pd.to_numeric(raw, errors="coerce")
    if sec.isna().all():
        sec = pd.to_timedelta(raw.astype(str), errors="coerce").dt.total_seconds()

    within = sec.groupby(gkey).transform(lambda s: s - s.iloc[0])

    # stable order of groups (by first appearance)
    _, first_idx = np.unique(gkey.to_numpy(), return_index=True)
    ordered_groups = gkey.iloc[np.sort(first_idx)]
    offsets: Dict[str, float] = {}
    total = 0.0
    for grp in ordered_groups:
        offsets[grp] = total
        last = within[gkey == grp].iloc[-1]
        total += float(last if pd.notna(last) else 0.0)

    stitched = within + gkey.map(offsets).astype(float)
    return stitched - stitched.iloc[0]

def insert_line_breaks_vq(df: pd.DataFrame, cap_col: str, v_col: str) -> pd.DataFrame:
    if df.empty or cap_col not in df.columns or v_col not in df.columns:
        return df
    d = df.reset_index(drop=True).copy()
    cap = pd.to_numeric(d[cap_col], errors="coerce").fillna(0)
    idxs = cap[(cap.shift(-1) == 0) & (cap > 0)].index.tolist()
    if not idxs:
        return d
    pieces, last = [], 0
    for i in idxs:
        pieces.append(d.iloc[last:i + 1])
        nan_row = {c: (np.nan if c in [cap_col, v_col] else d.iloc[i].get(c)) for c in d.columns}
        pieces.append(pd.DataFrame([nan_row]))
        last = i + 1
    pieces.append(d.iloc[last:])
    return _concat_nonempty(pieces)

def compute_ce(df: pd.DataFrame, cell_type: str = "cathode") -> pd.DataFrame:
    if "Cycle Index" not in df.columns:
        return pd.DataFrame()
    sc_chg = "Chg. Spec. Cap.(mAh/g)" if "Chg. Spec. Cap.(mAh/g)" in df.columns else None
    sc_dch = "DChg. Spec. Cap.(mAh/g)" if "DChg. Spec. Cap.(mAh/g)" in df.columns else None
    sc_any = "Spec. Cap.(mAh/g)" if "Spec. Cap.(mAh/g)" in df.columns else None
    cap_mAh = "Capacity(mAh)" if "Capacity(mAh)" in df.columns else None
    has_step = "Step Type" in df.columns

    out = []
    for c in pd.unique(df["Cycle Index"].dropna()):
        sub = df[df["Cycle Index"] == c]
        q_chg = np.nan
        q_dch = np.nan
        if sc_chg and sc_dch:
            q_chg = pd.to_numeric(sub[sc_chg], errors="coerce").max()
            q_dch = pd.to_numeric(sub[sc_dch], errors="coerce").max()
        elif sc_any and has_step:
            chg = sub[sub["Step Type"].str.contains("Chg", na=False)].get(sc_any)
            dch = sub[sub["Step Type"].str.contains("DChg", na=False)].get(sc_any)
            q_chg = pd.to_numeric(chg, errors="coerce").max() if chg is not None else np.nan
            q_dch = pd.to_numeric(dch, errors="coerce").max() if dch is not None else np.nan
        elif cap_mAh and has_step:
            chg = sub[sub["Step Type"].str.contains("Chg", na=False)].get(cap_mAh)
            dch = sub[sub["Step Type"].str.contains("DChg", na=False)].get(cap_mAh)
            q_chg = pd.to_numeric(chg, errors="coerce").max() if chg is not None else np.nan
            q_dch = pd.to_numeric(dch, errors="coerce").max() if dch is not None else np.nan
        elif cap_mAh:
            q_chg = pd.to_numeric(sub.get(cap_mAh), errors="coerce").max()
            q_dch = q_chg

        if pd.isna(q_chg) or pd.isna(q_dch) or q_dch == 0 or q_chg == 0:
            ce = np.nan
        else:
            ce = (q_dch / q_chg * 100.0) if cell_type == "cathode" else (q_chg / q_dch * 100.0)
        out.append({"cycle": c, "ce": ce, "q_chg": q_chg, "q_dch": q_dch})
    return pd.DataFrame(out).sort_values("cycle")

def detect_columns(columns: List[str]) -> Dict[str, Optional[str]]:
    cols = {c.lower(): c for c in columns}
    def pick(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None
    return {
        "time": pick("Total Time", "Time"),
        "voltage": pick("Voltage(V)"),
        "capacity": pick("Spec. Cap.(mAh/g)", "Capacity(mAh)"),
        "cycle": pick("Cycle Index"),
        "step": pick("Step Type"),
    }

# ----------------------------
# DCIR function (kept compatible)
# ----------------------------
def compute_dcir_for_ndax(
    df: pd.DataFrame,
    cell_id: str,
    pulse_length_s: float,
    soc_mode: str = "design",
    soc_levels=None,
    pulses_per_soc: int = 2,
    pre_rest_window_s: float = 60.0,
    inst_window_s=(0.5, 1.5),
    end_window_s: float = 5.0,
    pulse_tol_s: float = 2.0,
    current_threshold_A: float = 0.001,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    cols = d.columns

    if "Step_Index" not in cols:
        raise ValueError("Expected 'Step_Index' column but did not find it.")

    step_index_col = "Step_Index"
    status_col = "Step Type" if "Step Type" in cols else "Status"
    time_col = "Time"
    volt_col = "Voltage(V)" if "Voltage(V)" in cols else "Voltage"
    cur_col = "Current(mA)"

    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d["Current_A"] = pd.to_numeric(d[cur_col], errors="coerce") / 1000.0

    step_summary = (
        d.groupby(step_index_col)
         .agg(
             Status=(status_col, "first"),
             tmin=(time_col, "min"),
             tmax=(time_col, "max"),
             I_mean=("Current_A", "mean"),
         )
         .sort_index()
    )
    step_summary["duration_s"] = step_summary["tmax"] - step_summary["tmin"]

    status_str = step_summary["Status"].astype(str).str.lower()
    is_rest = status_str.str.contains("rest")

    is_pulse_dur = step_summary["duration_s"].between(
        pulse_length_s - pulse_tol_s,
        pulse_length_s + pulse_tol_s,
    )
    enough_I = step_summary["I_mean"].abs() >= current_threshold_A

    pulse_mask = (~is_rest) & is_pulse_dur & enough_I
    pulse_steps = step_summary.index[pulse_mask].tolist()

    dcir_end_col = f"DCIR_{int(round(pulse_length_s))}s_Ohm"

    if not pulse_steps:
        return pd.DataFrame(columns=[
            "Cell_ID", "Pulse_Direction", "SoC_label", "pulse_duration_s",
            dcir_end_col, "DCIR_inst_Ohm",
        ])

    # Design-based SOC
    soc_labels_by_step = {}
    if soc_mode == "design" and soc_levels:
        expanded = []
        for lvl in soc_levels:
            expanded.extend([lvl] * pulses_per_soc)
        for i, step in enumerate(sorted(pulse_steps)):
            soc_labels_by_step[step] = expanded[i] if i < len(expanded) else expanded[-1]

    # Data-based SOC capacity ref
    q_ref = None
    q_ref_source = None
    if soc_mode == "data":
        if "Discharge_Capacity(mAh)" in cols:
            q_ref = pd.to_numeric(d["Discharge_Capacity(mAh)"], errors="coerce").max()
            if math.isfinite(q_ref) and q_ref > 0:
                q_ref_source = "discharge"
        if (q_ref is None or not math.isfinite(q_ref) or q_ref <= 0) and "Charge_Capacity(mAh)" in cols:
            q_ref = pd.to_numeric(d["Charge_Capacity(mAh)"], errors="coerce").max()
            if math.isfinite(q_ref) and q_ref > 0:
                q_ref_source = "charge"
        if q_ref is not None and (not math.isfinite(q_ref) or q_ref <= 0):
            q_ref = None
            q_ref_source = None

    rest_steps = step_summary.index[is_rest]
    inst_start, inst_end = inst_window_s

    records = []
    for step in sorted(pulse_steps):
        step_info = step_summary.loc[step]

        prior_rests = rest_steps[rest_steps < step]
        if len(prior_rests) == 0:
            continue
        rest_step = prior_rests[-1]

        df_rest = d[d[step_index_col] == rest_step]
        df_pulse = d[d[step_index_col] == step]
        if df_rest.empty or df_pulse.empty:
            continue

        # pre-pulse rest window
        rt_max = df_rest[time_col].max()
        rt_min = df_rest[time_col].min()
        rest_start = max(rt_min, rt_max - pre_rest_window_s)
        rest_win = df_rest[df_rest[time_col] >= rest_start]
        if rest_win.empty:
            rest_win = df_rest

        v_pre = pd.to_numeric(rest_win[volt_col], errors="coerce").mean()
        i_pre_mA = pd.to_numeric(rest_win[cur_col], errors="coerce").mean()

        # instant window
        inst = df_pulse[(df_pulse[time_col] >= inst_start) & (df_pulse[time_col] <= inst_end)]
        if inst.empty:
            inst = df_pulse.head(min(3, len(df_pulse)))

        v_inst = pd.to_numeric(inst[volt_col], errors="coerce").mean()
        i_inst_mA = pd.to_numeric(inst[cur_col], errors="coerce").mean()

        # end window
        pt_max = df_pulse[time_col].max()
        pt_min = df_pulse[time_col].min()
        end_start = max(pt_min, pt_max - end_window_s)
        end_win = df_pulse[df_pulse[time_col] >= end_start]
        if end_win.empty:
            end_win = df_pulse.iloc[[-1]]

        v_end = pd.to_numeric(end_win[volt_col], errors="coerce").mean()
        i_end_mA = pd.to_numeric(end_win[cur_col], errors="coerce").mean()

        # ΔV/ΔI
        delta_V_inst = v_pre - v_inst
        delta_I_inst_A = (i_inst_mA - i_pre_mA) / 1000.0
        delta_V_end = v_pre - v_end
        delta_I_end_A = (i_end_mA - i_pre_mA) / 1000.0

        dcir_inst = math.nan
        dcir_end = math.nan
        if abs(delta_I_inst_A) > 0:
            dcir_inst = abs(delta_V_inst / delta_I_inst_A)
        if abs(delta_I_end_A) > 0:
            dcir_end = abs(delta_V_end / delta_I_end_A)

        # direction
        pulse_i_mean = df_pulse["Current_A"].mean()
        if pulse_i_mean > 0:
            direction = "charge"
        elif pulse_i_mean < 0:
            direction = "discharge"
        else:
            direction = "unknown"

        # SoC label
        soc_label = None
        if soc_mode == "design" and soc_levels:
            soc_label = soc_labels_by_step.get(step, None)
        elif soc_mode == "data" and q_ref and q_ref_source:
            if q_ref_source == "discharge" and "Discharge_Capacity(mAh)" in cols:
                q_rest = pd.to_numeric(df_rest["Discharge_Capacity(mAh)"], errors="coerce").max()
                if math.isfinite(q_rest):
                    soc_label = 100.0 * (1.0 - q_rest / q_ref)
            elif q_ref_source == "charge" and "Charge_Capacity(mAh)" in cols:
                q_rest = pd.to_numeric(df_rest["Charge_Capacity(mAh)"], errors="coerce").max()
                if math.isfinite(q_rest):
                    soc_label = 100.0 * (q_rest / q_ref)
            if soc_label is not None:
                soc_label = round(float(soc_label), 1)

        records.append({
            "Cell_ID": cell_id,
            "Pulse_Direction": direction,
            "SoC_label": soc_label,
            "pulse_duration_s": round(float(step_info["duration_s"]), 3),
            dcir_end_col: dcir_end,
            "DCIR_inst_Ohm": dcir_inst,
        })

    return pd.DataFrame.from_records(records)


# ----------------------------
# NDAX loader (bytes-based cache)
# ----------------------------

# Keep only the columns you use across the app (big memory win).
KEEP_COLS = [
    "Total Time", "Time",
    "Voltage(V)", "Current(mA)",
    "Spec. Cap.(mAh/g)", "Capacity(mAh)",
    "Chg. Spec. Cap.(mAh/g)", "DChg. Spec. Cap.(mAh/g)",
    "Chg. Cap.(mAh)", "DChg. Cap.(mAh)",
    "Cycle Index", "Step Type",
    "Step_Index",
    "Charge_Capacity(mAh)", "Discharge_Capacity(mAh)",
    "Timestamp", "Record Time", "DateTime", "Date Time", "System Time", "Local Time",
]

def _md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def load_ndax_bytes(filename: str, file_bytes: bytes, file_md5: str) -> pd.DataFrame:
    # file_md5 is used to make cache key explicit & stable
    if nda is None:
        raise RuntimeError("NewareNDA is not installed. Install: pip install NewareNDA")

    if Path(filename).suffix.lower() != ".ndax":
        raise RuntimeError("Only .ndax files are supported.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ndax") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        if build_ndax_df is not None:
            df = build_ndax_df(tmp_path, cycle_mode="auto")
        else:
            df = nda.read(tmp_path)

        df = normalize_neware_headers(df)
        df = infer_rest_step(df)

        keep = [c for c in KEEP_COLS if c in df.columns]
        df = df.loc[:, keep].copy()

        return df

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ----------------------------
# Header / logo
# ----------------------------
c1, c2, c3 = st.columns([1, 1, 1])
with c2:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH))
    else:
        st.caption(f"Logo missing: {LOGO_PATH.name}")

st.title("🔋 BATTERY CELL DATA — VISUALIZER 📈")
st.caption("Built by the Preli team")

# ----------------------------
# Sidebar: upload + parse (GATED)
# ----------------------------
st.sidebar.header("Upload NDAX files only")



with st.sidebar.form("upload_form", clear_on_submit=False):
    uploaded_files = st.file_uploader(
        "Drop Neware .ndax files (multiple allowed)",
        type=["ndax"],
        accept_multiple_files=True,
    )
    parse_now = st.form_submit_button("🚀 launch files")

# Provide an easy way to clear state
top_l, top_r = st.columns([6, 1])
with top_r:
    if "parsed_by_file" in st.session_state:
        if st.button("🧹 Reset", key="clear_parsed_main"):
            for k in ["parsed_by_file", "file_checks", "uploaded_names_cache", "selected_files"]:
                st.session_state.pop(k, None)
            st.rerun()
            
if not uploaded_files and "parsed_by_file" not in st.session_state:
    st.info("Upload one or more NDAX files and press 🚀 launch.")
    st.stop()

# Parse only when user clicks, or use existing parsed data
if parse_now:
    parsed: Dict[str, pd.DataFrame] = {}
    prog = st.sidebar.progress(0)
    for i, uf in enumerate(uploaded_files):
        b = uf.getvalue()
        h = _md5_bytes(b)
        df = load_ndax_bytes(uf.name, b, h)
        df["__file"] = uf.name
        df["__family"] = family_from_filename(uf.name)
        parsed[uf.name] = df
        prog.progress((i + 1) / max(1, len(uploaded_files)))
    st.sidebar.success(f"Parsed {len(parsed)} file(s).")
    st.session_state["parsed_by_file"] = parsed
    st.session_state["uploaded_names_cache"] = sorted(list(parsed.keys()))

if "parsed_by_file" not in st.session_state:
    st.info("Upload your files, then click **launch files**.")
    st.stop()

parsed_by_file: Dict[str, pd.DataFrame] = st.session_state["parsed_by_file"]

# If user uploads new set but didn't click parse, keep old parsed; show hint
current_upload_names = sorted([uf.name for uf in uploaded_files]) if uploaded_files else []
cached_names = st.session_state.get("uploaded_names_cache", sorted(list(parsed_by_file.keys())))
if current_upload_names and set(current_upload_names) != set(cached_names) and not parse_now:
    st.warning("You changed the uploaded files. Click **launch files** to apply the new upload set.")

# ----------------------------
# Sidebar: options
# ----------------------------
with st.sidebar.expander("CE options", expanded=True):
    cell_type_sel = st.radio(
        "Cell type (for CE direction)",
        ["anode", "cathode", "full"],
        index=1,
        horizontal=True,
    )

palette_name = st.sidebar.selectbox("Palette 🎨", list(PALETTES.keys()), index=0)
palette = PALETTES[palette_name]

with st.sidebar.expander("Formatting", expanded=False):
    show_markers = st.checkbox("Show markers", False)
    marker_size = st.number_input("Marker size", 1, 20, 4)
    line_width = st.number_input("Line width", 1.0, 10.0, 2.5, 0.5)
    show_grid_global = st.checkbox("Show grid", True)

# ----------------------------
# File selection checkboxes
# ----------------------------
with st.sidebar.expander("Files to plot", expanded=False):

    files_all = sorted(list(parsed_by_file.keys()))
    if "file_checks" not in st.session_state or set(st.session_state["file_checks"].keys()) != set(files_all):
        st.session_state["file_checks"] = {f: True for f in files_all}

    colA, colB = st.columns(2)
    with colA:
        if st.button("Select all", key="files_select_all"):
            st.session_state["file_checks"] = {f: True for f in files_all}
    with colB:
        if st.button("Select none", key="files_select_none"):
            st.session_state["file_checks"] = {f: False for f in files_all}

    for f in files_all:
        st.session_state["file_checks"][f] = st.checkbox(
            f,
            value=st.session_state["file_checks"].get(f, True),
            key=f"file_cb__{f}",
        )

selected_files = [f for f, on in st.session_state["file_checks"].items() if on]
if not selected_files:
    st.info("Select at least one file in the sidebar to plot.")
    st.stop()
# ----------------------------
# Color mode (global)
# ----------------------------
families_in_data = sorted({family_from_filename(f) for f in files_all})
with st.sidebar.expander("Color mode", expanded=True):
    color_mode_global = st.radio(
        "Color by",
        ["Per file", "Filename family"],
        index=0,
        horizontal=True
    )

color_map_file = {src: palette[i % len(palette)] for i, src in enumerate(files_all)}
color_map_fam  = {fam: palette[i % len(palette)] for i, fam in enumerate(families_in_data)}

def color_for_src(src: str) -> str:
    if color_mode_global == "Per file":
        return color_map_file.get(src, palette[0])
    return color_map_fam.get(family_from_filename(src), palette[0])

# ----------------------------
# Utility: get selected frames
# ----------------------------
def get_selected_frames() -> List[pd.DataFrame]:
    return [parsed_by_file[f] for f in selected_files if f in parsed_by_file]

def get_union_columns(frames: List[pd.DataFrame]) -> List[str]:
    cols = []
    seen = set()
    for df in frames:
        for c in df.columns:
            if c not in seen:
                seen.add(c)
                cols.append(c)
    return cols

frames_selected = get_selected_frames()
union_cols = get_union_columns(frames_selected)
G = detect_columns(union_cols)

# ----------------------------
# View selector (LAZY EXECUTION)
# ----------------------------
PAGES = [
    "XY Builder", "Voltage–Time", "Voltage–Capacity",
    "Capacity vs Cycle", "Capacity & CE", "DCIR",
    "ICE Boxplot", "Raw File Preview",
]


view = st.segmented_control(
    "View selector",
    options=PAGES,
    default=PAGES[3],
    key="view_selector_main",
    label_visibility="collapsed",
)
# ----------------------------
# Raw File Preview
# ----------------------------
if view == "Raw File Preview":
    st.subheader("Preview parsed data (first rows per file)")
    max_rows = st.number_input("Rows per file", min_value=5, max_value=1000, value=900, step=50)
    for f in selected_files:
        df = parsed_by_file[f]
        am = None
        try:
            am = df.attrs.get("active_mass_g", None)
        except Exception:
            am = None
        if isinstance(am, (int, float)) and am > 0:
            am_str = f"{am:.6f} g"
        else:
            am_str = "N/A"
        st.markdown(f"**{f}**  —  shape: {df.shape}  —  active_mass_g: {am_str}")
        st.dataframe(df.head(int(max_rows)), width="stretch")

    st.stop()

# ----------------------------
# XY Builder
# ----------------------------
if view == "XY Builder":
    st.subheader("Free-form XY plot builder")

    all_cols = [c for c in union_cols if c not in ["__file", "__family"]]
    # numeric columns = numeric in at least one selected frame
    numeric_cols = []
    for c in all_cols:
        for df in frames_selected:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                numeric_cols.append(c)
                break

    if not all_cols:
        st.warning("No columns found in selected files.")
        st.stop()

    if "xy_x" not in st.session_state:
        st.session_state.xy_x = G.get("time") if G.get("time") in all_cols else all_cols[0]
    if "xy_y" not in st.session_state or not st.session_state.xy_y:
        defaults = [c for c in [G.get("voltage"), G.get("capacity")] if c in numeric_cols]
        st.session_state.xy_y = defaults[:1] if defaults else (numeric_cols[:1] if numeric_cols else [])

    row1 = st.columns([1.1, 1.2, 0.9])
    with row1[0]:
        x_col = st.selectbox(
            "X axis",
            all_cols,
            index=(all_cols.index(st.session_state.xy_x) if st.session_state.xy_x in all_cols else 0),
        )
    is_time_x = (x_col == G.get("time"))
    align_t0 = is_time_x
    if is_time_x:
        st.caption("Time axis is aligned to t₀")

    with row1[1]:
        y_cols = st.multiselect(
            "Y axis (one or more)",
            numeric_cols,
            default=[y for y in st.session_state.xy_y if y in numeric_cols] or (numeric_cols[:1] if numeric_cols else []),
        )
    with row1[2]:
        rolling = st.number_input("Rolling mean window (pts)", 1, 9999, 1, 1)
        use_global_colors_xy = st.checkbox(
            "Use global colors here",
            value=True,
            help="If off, Plotly default color cycle is used.",
        )

    row2 = st.columns([1, 1, 0.6])
    with row2[0]:
        y_min = st.text_input("Y min (blank=auto)", "")
    with row2[1]:
        y_max = st.text_input("Y max (blank=auto)", "")

    row3 = st.columns([1, 1, 0.6])
    with row3[0]:
        x_min = st.text_input("X min (blank=auto)", "")
    with row3[1]:
        x_max = st.text_input("X max (blank=auto)", "")

    st.session_state.xy_x = x_col
    st.session_state.xy_y = y_cols

    if not y_cols:
        st.warning("Pick at least one Y column to plot.")
        st.stop()

    fig = go.Figure()
    added = False

    for src in selected_files:
        df = parsed_by_file[src]
        if x_col not in df.columns:
            continue

        # build x used
        if align_t0:
            df_local = df.dropna(subset=[x_col]).copy()
            df_local["_x"] = build_global_time_seconds(df_local, time_col=x_col, cycle_col="Cycle Index", step_col="Step Type")
            x_used = "_x"
        else:
            df_local = df

        # optional smoothing (per file)
        if rolling > 1:
            df_local = df_local.sort_values(x_used)
            for y in y_cols:
                if y in df_local.columns:
                    df_local[y] = pd.to_numeric(df_local[y], errors="coerce")
                    df_local[y] = df_local[y].rolling(rolling, min_periods=1).mean()

        for y in y_cols:
            if y not in df_local.columns:
                continue
            s = df_local.dropna(subset=[x_used, y])
            if s.empty:
                continue

            if use_global_colors_xy:
                c = color_for_src(src)
                fig.add_trace(go.Scatter(
                    x=s[x_used], y=s[y],
                    mode="lines",
                    name=f"{pretty_src(src)} — {y}",
                    line=dict(color=c, width=line_width),
                    marker=dict(size=marker_size)
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=s[x_used], y=s[y],
                    mode="lines",
                    name=f"{pretty_src(src)} — {y}",
                    line=dict(width=line_width),
                    marker=dict(size=marker_size)
                ))
            added = True

    if not added:
        st.warning("No data drawn — check your column choices.")
    else:
        try:
            if x_min != "":
                fig.update_xaxes(range=[float(x_min), float(x_max) if x_max != "" else None])
        except Exception:
            pass
        try:
            if y_min != "":
                fig.update_yaxes(range=[float(y_min), float(y_max) if y_max != "" else None])
        except Exception:
            pass

        fig.update_layout(template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        if show_grid_global:
            fig.update_xaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)
            fig.update_yaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)

        st.plotly_chart(fig, width="stretch", config=CAMERA_CFG)

    st.stop()

# ----------------------------
# Voltage–Time
# ----------------------------
if view == "Voltage–Time":
    st.subheader("Voltage–Time")
    tcol, vcol = G.get("time"), G.get("voltage")
    if not tcol or not vcol:
        st.warning("Couldn’t detect time/voltage. Check NDAX headers.")
        st.stop()

    unit = st.selectbox("Time units", ["seconds", "minutes", "hours"], 0)
    DIV = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}
    ABBR = {"seconds": "s", "minutes": "min", "hours": "h"}

    fig_vt = go.Figure()
    for src in selected_files:
        df = parsed_by_file[src]
        if tcol not in df.columns or vcol not in df.columns:
            continue

        s = df.dropna(subset=[tcol, vcol]).copy()
        if s.empty:
            continue

        s["_t"] = build_global_time_seconds(s, time_col=tcol, cycle_col="Cycle Index", step_col="Step Type")
        s = s.dropna(subset=["_t", vcol]).sort_values("_t")
        if s.empty:
            continue

        fig_vt.add_trace(go.Scatter(
            x=s["_t"] / DIV[unit],
            y=s[vcol],
            name=pretty_src(src),
            mode=("lines+markers" if show_markers else "lines"),
            line=dict(color=color_for_src(src), width=line_width),
            marker=dict(size=marker_size),
        ))

    fig_vt.update_layout(
        template="plotly_white",
        xaxis_title=f"Time ({ABBR[unit]})",
        yaxis_title=vcol,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    if show_grid_global:
        fig_vt.update_xaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)
        fig_vt.update_yaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)

    style_for_ppt(fig_vt)
    st.plotly_chart(fig_vt, width="stretch", config=CAMERA_CFG)
    add_ppt_download(fig_vt, filename_base="voltage_time")
    st.stop()

# ----------------------------
# Voltage–Capacity
# ----------------------------
if view == "Voltage–Capacity":
    st.subheader("Voltage–Capacity")
    ccol, vcol = G.get("capacity"), G.get("voltage")
    if not ccol or not vcol:
        st.warning("Couldn’t detect capacity/voltage. Check NDAX headers.")
        st.stop()

    # cycle filter (works if cycle exists)
    cyc_col = G.get("cycle") if G.get("cycle") in union_cols else None
    cycles_available = None
    if cyc_col:
        cyc_values = []
        for df in frames_selected:
            if cyc_col in df.columns:
                cyc_values.append(df[cyc_col].dropna())
        if cyc_values:
            cycles_available = sorted(pd.unique(pd.concat(cyc_values, ignore_index=True)))

    sel_cycle = None
    rng = None
    if cycles_available:
        sel_cycle = st.selectbox("Cycle", ["All"] + cycles_available, 0)
        cmin, cmax = int(min(cycles_available)), int(max(cycles_available))
        rng = st.slider("Cycle range (optional)", cmin, cmax, (cmin, cmax), 1)

    fig_vq = go.Figure()
    for src in selected_files:
        df = parsed_by_file[src]
        if ccol not in df.columns or vcol not in df.columns:
            continue
        s = df.dropna(subset=[ccol, vcol]).copy()
        if s.empty:
            continue
        if cyc_col and cyc_col in s.columns and cycles_available:
            if sel_cycle != "All":
                s = s[s[cyc_col] == sel_cycle]
            else:
                lo, hi = rng
                if (lo, hi) != (int(min(cycles_available)), int(max(cycles_available))):
                    s = s[s[cyc_col].between(lo, hi)]
        if s.empty:
            continue
        cols = [ccol, vcol, "__file"] + ([cyc_col] if cyc_col and cyc_col in s.columns else []) + (["Step Type"] if "Step Type" in s.columns else [])
        plot_df = insert_line_breaks_vq(s[cols], cap_col=ccol, v_col=vcol)

        fig_vq.add_trace(go.Scatter(
            x=plot_df[ccol], y=plot_df[vcol],
            name=pretty_src(src),
            mode=("lines+markers" if show_markers else "lines"),
            line=dict(color=color_for_src(src), width=line_width),
            marker=dict(size=marker_size),
        ))

    x_label = f"{ccol}" if "mAh/g" in ccol else ccol
    fig_vq.update_layout(
        template="plotly_white",
        xaxis_title=x_label,
        yaxis_title=vcol,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    if show_grid_global:
        fig_vq.update_xaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)
        fig_vq.update_yaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)

    style_for_ppt(fig_vq)
    st.plotly_chart(fig_vq, width="stretch", config=CAMERA_CFG)
    add_ppt_download(fig_vq, filename_base="voltage_capacity")
    st.stop()

# ----------------------------
# Capacity vs Cycle
# ----------------------------
if view == "Capacity vs Cycle":
    st.subheader("Capacity vs Cycle")

    cyc = G.get("cycle")
    if not cyc:
        st.info("Need Cycle Index to plot.")
        st.stop()

    # choose a capacity column that exists in at least one file
    cap_col = None
    for cand in ["Spec. Cap.(mAh/g)", "DChg. Spec. Cap.(mAh/g)", "Chg. Spec. Cap.(mAh/g)", "Capacity(mAh)"]:
        if cand in union_cols:
            cap_col = cand
            break
    if cap_col is None:
        st.info("No capacity column detected.")
        st.stop()

    fig_cap = go.Figure()
    for src in selected_files:
        df = parsed_by_file[src]
        if cyc not in df.columns or cap_col not in df.columns:
            continue
        s = df[[cyc, cap_col]].dropna()
        if s.empty:
            continue
        cap_cycle = s.groupby(cyc)[cap_col].max().reset_index()
        if cap_cycle.empty:
            continue
        c = color_for_src(src)
        fig_cap.add_trace(go.Scatter(
            x=cap_cycle[cyc], y=cap_cycle[cap_col],
            name=pretty_src(src),
            mode=("lines+markers" if show_markers else "lines"),
            line=dict(color=c, width=line_width),
            marker=dict(size=marker_size, symbol="circle-open", color=c,
                        line=dict(color=c, width=max(1, int(line_width - 1)))),
        ))

    y_label = "Specific capacity (mAh/g)" if "mAh/g" in cap_col else "Capacity (mAh)"
    fig_cap.update_layout(
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig_cap.update_xaxes(title_text="Cycle", showgrid=show_grid_global, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)
    fig_cap.update_yaxes(title_text=y_label, showgrid=show_grid_global, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)

    style_for_ppt(fig_cap)
    st.plotly_chart(fig_cap, width="stretch", config=CAMERA_CFG)
    add_ppt_download(fig_cap, filename_base="capacity_vs_cycle")
    st.stop()

# ----------------------------
# Capacity & CE
# ----------------------------
if view == "Capacity & CE":
    st.subheader("Capacity & Coulombic Efficiency vs Cycle")

    cyc = G.get("cycle")
    if not cyc:
        st.info("Need Cycle Index for CE plot.")
        st.stop()

    cap_col = None
    for cand in ["Spec. Cap.(mAh/g)", "DChg. Spec. Cap.(mAh/g)", "Chg. Spec. Cap.(mAh/g)", "Capacity(mAh)"]:
        if cand in union_cols:
            cap_col = cand
            break
    if cap_col is None:
        st.info("Need a capacity column for CE plot.")
        st.stop()

    fig_ce = make_subplots(specs=[[{"secondary_y": True}]])
    ce_cell_type = "cathode" if cell_type_sel == "full" else cell_type_sel

    for src in selected_files:
        df = parsed_by_file[src]
        if df.empty or cyc not in df.columns:
            continue
        c = color_for_src(src)

        ce_df = compute_ce(df, cell_type=ce_cell_type)
        cap_cycle = None
        if cap_col in df.columns:
            cap_cycle = df[[cyc, cap_col]].dropna().groupby(cyc)[cap_col].max().reset_index()

        if cap_cycle is not None and not cap_cycle.empty:
            fig_ce.add_trace(go.Scatter(
                x=cap_cycle[cyc], y=cap_cycle[cap_col],
                name=f"Cap — {pretty_src(src)}",
                mode=("lines+markers" if show_markers else "lines"),
                line=dict(color=c, width=line_width),
                marker=dict(size=marker_size, symbol="circle-open", color=c,
                            line=dict(color=c, width=max(1, int(line_width - 1)))),
            ), secondary_y=False)

        if ce_df is not None and not ce_df.empty:
            fig_ce.add_trace(go.Scatter(
                x=ce_df["cycle"], y=ce_df["ce"],
                name=f"CE — {pretty_src(src)}",
                mode=("lines+markers" if show_markers else "lines"),
                line=dict(color=c, dash="dash", width=line_width),
                marker=dict(size=marker_size, symbol="diamond", color=c),
            ), secondary_y=True)

    y_left = "Specific capacity (mAh/g)" if "mAh/g" in (cap_col or "") else "Capacity (mAh)"
    show_grid = st.checkbox("Show grid", value=True, key="ce_show_grid")
    grid_side = st.radio("Y-grid on", ["left", "right", "both", "none"], index=0, horizontal=True)

    fig_ce.update_yaxes(title_text=y_left, secondary_y=False)
    fig_ce.update_yaxes(title_text="CE (%)", range=[90, 105], secondary_y=True)
    fig_ce.update_xaxes(title_text="Cycle")
    fig_ce.update_layout(template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02))

    if show_grid and grid_side != "none":
        fig_ce.update_xaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)
        fig_ce.update_yaxes(showgrid=(grid_side in ["left", "both"]), gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5, secondary_y=False)
        fig_ce.update_yaxes(showgrid=(grid_side in ["right", "both"]), gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5, secondary_y=True)
    else:
        fig_ce.update_xaxes(showgrid=False)
        fig_ce.update_yaxes(showgrid=False, secondary_y=False)
        fig_ce.update_yaxes(showgrid=False, secondary_y=True)

    style_for_ppt(fig_ce)
    st.plotly_chart(fig_ce, width="stretch", config=CAMERA_CFG)
    add_ppt_download(fig_ce, filename_base="ce_and_capacity")
    st.stop()

# ----------------------------
# DCIR
# ----------------------------
if view == "DCIR":
    st.subheader("DCIR calculator")

    left, right = st.columns([1.2, 1])
    with left:
        pulse_length_s = st.number_input("DCIR pulse length [s]", min_value=1.0, value=18.0, step=1.0)
        soc_mode_label = st.radio(
            "SOC labelling mode",
            ["By test design (e.g. 80/50/20/5%)", "Estimate from data (capacity-based)"],
            index=0,
        )
        design_mode = soc_mode_label.startswith("By test")
        soc_levels_text = st.text_input(
            "SOC levels (%) in pulse order",
            value="80, 50, 20, 5",
            help="Example: 80, 50, 20, 5 with 2 pulses per SOC → 8 pulses total.",
            disabled=(not design_mode),
        )
        pulses_per_soc = st.number_input(
            "Pulses per SOC level",
            min_value=1, max_value=10, value=2, step=1,
            disabled=(not design_mode),
        )
        if design_mode:
            soc_mode = "design"
            soc_levels = [float(x.strip()) for x in soc_levels_text.split(",") if x.strip()]
        else:
            soc_mode = "data"
            soc_levels = None

        run_btn = st.button("Compute DCIR")

    with right:
        st.markdown(
            """
**How this works :**

- Pre-rest OCV = mean Voltage over last **60 s** of the Rest step.
- **Instantaneous DCIR** = ΔV/ΔI using early window (0.5–1.5 s into pulse).
- **End-of-pulse DCIR** = ΔV/ΔI using last **5 s** of pulse.
- Pulses detected as non-Rest steps with duration ≈ your pulse length and non-zero current.
            """
        )

    if not run_btn:
        st.info("Set your options, then click **Compute DCIR**.")
        st.stop()

    all_results = []
    for src in selected_files:
        df = parsed_by_file[src]
        if df.empty:
            continue
        try:
            res = compute_dcir_for_ndax(
                df=df, cell_id=src,
                pulse_length_s=pulse_length_s,
                soc_mode=soc_mode,
                soc_levels=soc_levels,
                pulses_per_soc=int(pulses_per_soc),
            )
        except Exception as e:
            st.warning(f"Skipping {src} due to error: {e}")
            continue
        if res is not None and not res.empty:
            all_results.append(res)

    if not all_results:
        st.info("No DCIR pulses detected. Check pulse length and inputs.")
        st.stop()

    dcir_results = pd.concat(all_results, ignore_index=True)

    st.subheader("DCIR results (raw, one row per pulse)")
    st.dataframe(dcir_results, width="stretch",)

    # summary wide table
    dcir_end_col = f"DCIR_{int(round(pulse_length_s))}s_Ohm"
    wide_summary = None

    if "SoC_label" in dcir_results.columns and dcir_end_col in dcir_results.columns and "DCIR_inst_Ohm" in dcir_results.columns:
        tmp = dcir_results.copy()

        def soc_fmt(x):
            s = str(x)
            try:
                v = float(s.replace("%", ""))
                return f"{v:g}%"
            except Exception:
                return s

        tmp["SoC_fmt"] = tmp["SoC_label"].apply(soc_fmt)
        metric_map = {dcir_end_col: "18s", "DCIR_inst_Ohm": "inst"}

        long = tmp.melt(
            id_vars=["Cell_ID", "Pulse_Direction", "SoC_fmt"],
            value_vars=list(metric_map.keys()),
            var_name="metric",
            value_name="DCIR",
        )
        long["metric_suffix"] = long["metric"].map(metric_map)
        long["column_name"] = long["SoC_fmt"] + "_" + long["metric_suffix"]

        wide = long.pivot_table(
            index=["Cell_ID", "Pulse_Direction"],
            columns="column_name",
            values="DCIR",
            aggfunc="mean",
        ).reset_index()

        value_cols = [c for c in wide.columns if c not in ["Cell_ID", "Pulse_Direction"]]

        def sort_key(col):
            if "_" in col:
                soc_part, suffix = col.split("_", 1)
            else:
                soc_part, suffix = col, ""
            try:
                soc_num = float(soc_part.replace("%", ""))
            except Exception:
                soc_num = math.inf
            metric_order = {"inst": 0, "18s": 1}
            return (metric_order.get(suffix, 2), soc_num)

        value_cols_sorted = sorted(value_cols, key=sort_key)
        wide_summary = wide[["Cell_ID", "Pulse_Direction"] + value_cols_sorted]

        st.subheader(f"DCIR summary (instant + {int(round(pulse_length_s))}s, SoC as columns)")
        st.dataframe(wide_summary, width="stretch")
    else:
        st.info("Could not build summary table (missing SoC_label or DCIR columns).")

    # Excel export
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        dcir_results.to_excel(writer, index=False, sheet_name="DCIR_raw")
        if wide_summary is not None:
            wide_summary.to_excel(writer, index=False, sheet_name="DCIR_summary")
    output.seek(0)

    st.download_button(
        label="⬇️ Download DCIR tables (Excel)",
        data=output,
        file_name="dcir_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.stop()

# ----------------------------
# ICE Boxplot
# ----------------------------
if view == "ICE Boxplot":
    st.subheader("First-cycle ICE / capacity boxplot")

    ce_cell_type = "cathode" if cell_type_sel == "full" else cell_type_sel

    grouped: Dict[str, Dict[str, List[float]]] = {}
    for src in selected_files:
        df = parsed_by_file[src]
        if "Cycle Index" not in df.columns:
            continue
        fam = family_from_filename(src)
        ce_df = compute_ce(df, cell_type=ce_cell_type)
        if ce_df.empty:
            continue
        ce_df = ce_df.sort_values("cycle")
        ice_row = None
        for _, r in ce_df.iterrows():
            q_chg = r.get("q_chg", np.nan)
            q_dch = r.get("q_dch", np.nan)
            if pd.notna(q_chg) and pd.notna(q_dch) and q_chg > 0 and q_dch > 0:
                ice_row = r
                break
        if ice_row is None:
            continue

        max_c = float(ice_row["q_chg"])
        max_d = float(ice_row["q_dch"])
        ice = float(ice_row["ce"]) if not pd.isna(ice_row["ce"]) else np.nan

        if fam not in grouped:
            grouped[fam] = {"Charge": [], "Discharge": [], "ICE": []}
        grouped[fam]["Charge"].append(max_c)
        grouped[fam]["Discharge"].append(max_d)
        if np.isfinite(ice):
            grouped[fam]["ICE"].append(ice)

    if not grouped:
        st.info("Could not compute capacities / ICE for any selected file.")
        st.stop()

    rows = []
    ce_rows = []
    for fam, vals in grouped.items():
        for v in vals["Charge"]:
            rows.append({"Group": fam, "Type": "Charge capacity", "Capacity": v})
        for v in vals["Discharge"]:
            rows.append({"Group": fam, "Type": "Discharge capacity", "Capacity": v})
        if vals["ICE"]:
            ce_rows.append({
                "Group": fam,
                "ICE_mean(%)": float(np.mean(vals["ICE"])),
                "ICE_std(%)": float(np.std(vals["ICE"])),
                "n": len(vals["ICE"]),
            })

    plot_data = pd.DataFrame(rows)
    ce_summary = pd.DataFrame(ce_rows)

    if plot_data.empty:
        st.info("No capacity data available for boxplot.")
        st.stop()

    has_spec_cap = any(c in union_cols for c in ["Spec. Cap.(mAh/g)", "DChg. Spec. Cap.(mAh/g)", "Chg. Spec. Cap.(mAh/g)"])
    y_label = "Specific capacity (mAh/g)" if has_spec_cap else "Capacity"

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    groups = sorted(plot_data["Group"].unique().tolist())
    positions = np.arange(len(groups)) * 2.0

    y_min_auto = 0.0
    y_max_auto = float(plot_data["Capacity"].max()) * 1.15
    use_custom_ylim = st.checkbox("Set custom y-axis limits", value=False)
    if use_custom_ylim:
        y_min = st.number_input("Y min", value=y_min_auto, step=10.0)
        y_max = st.number_input("Y max", value=y_max_auto, step=10.0)
    else:
        y_min, y_max = y_min_auto, y_max_auto
    ax.set_ylim(y_min, y_max)

    for i, fam in enumerate(groups):
        gd = plot_data[plot_data["Group"] == fam]
        chg = gd[gd["Type"] == "Charge capacity"]["Capacity"]
        dchg = gd[gd["Type"] == "Discharge capacity"]["Capacity"]

        ax.boxplot(
            chg, positions=[positions[i] - 0.4], widths=0.6, patch_artist=True,
            boxprops=dict(facecolor=NV_COLORDICT["nv_blue2"], color="black"),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
        )
        ax.boxplot(
            dchg, positions=[positions[i] + 0.4], widths=0.6, patch_artist=True,
            boxprops=dict(facecolor=NV_COLORDICT["nv_power_green"], color="black"),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
        )

        x_chg = np.random.normal(positions[i] - 0.4, 0.05, size=len(chg))
        x_dchg = np.random.normal(positions[i] + 0.4, 0.05, size=len(dchg))

        ax.scatter(
            x_chg, chg, alpha=0.6, color=NV_COLORDICT["nv_blue2"],
            edgecolor="black", linewidth=0.5,
            label="Charge capacity" if i == 0 else "",
        )
        ax.scatter(
            x_dchg, dchg, alpha=0.6, color=NV_COLORDICT["nv_power_green"],
            edgecolor="black", linewidth=0.5,
            label="Discharge capacity" if i == 0 else "",
        )

        ice_vals = grouped[fam]["ICE"]
        if ice_vals:
            ice_mean = np.mean(ice_vals)
            ice_std = np.std(ice_vals)
            ax.text(
                positions[i], y_max * 0.9,
                f"ICE: {ice_mean:.2f} ± {ice_std:.2f}%",
                ha="center", va="center", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.9),
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(groups)
    ax.set_ylabel(y_label)
    ax.set_title(f"First-cycle charge / discharge capacity and ICE ({ce_cell_type} CE)")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color=NV_COLORDICT["nv_gray3"])

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower right")

    st.pyplot(fig, width="content")

    st.subheader("ICE summary per group")
    if not ce_summary.empty:
        st.dataframe(ce_summary, width="stretch")
    else:
        st.caption("_No ICE values could be computed (missing charge/discharge pair)._")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=400, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="⬇️ Download boxplot as PNG (400 dpi)",
        data=buf,
        file_name="ice_boxplot.png",
        mime="image/png",
    )
    plt.close(fig)
    st.stop()

# Fallback
st.success("Loaded. Use the tabs above to explore your NDAX data.")

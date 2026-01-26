
import re
import io
import os
import textwrap
import math
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from streamlit_theme import st_theme
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
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

_theme = st_theme(key="app_theme") or {}
BASE_THEME = _theme.get("base")

if BASE_THEME not in ("light", "dark"):
    BASE_THEME = st.get_option("theme.base") or "light"

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
def apply_preli_style(fig, base: str, show_grid: bool = True, for_export: bool = False):
    """
    - On screen: adapts to Streamlit light/dark.
    - On export: forces PPT-friendly styling + transparent background.
    """
    is_dark = (base == "dark") and (not for_export)

    # Colors
    if for_export or not is_dark:
        fg = "#000000"
        grid = NV_COLORDICT["nv_gray3"]
        paper_bg = "rgba(0,0,0,0)" if for_export else "#ffffff"
        plot_bg  = "rgba(0,0,0,0)" if for_export else "#ffffff"
        legend_bg = "rgba(255,255,255,0.90)"
        legend_border = "rgba(0,0,0,0.35)"
    else:
        # use Streamlit theme background when available
        bg = st.get_option("theme.backgroundColor") or "#0E1117"
        fg = "#ffffff"
        grid = "rgba(255,255,255,0.18)"
        paper_bg = bg
        plot_bg = bg
        legend_bg = "rgba(0,0,0,0.35)"
        legend_border = "rgba(255,255,255,0.25)"

    fig.update_layout(
        template=None,  # important: stop Plotly template overriding your styling
        font=dict(family="Arial", size=16, color=fg),
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        margin=dict(l=80, r=40, t=60, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bordercolor=legend_border,
            borderwidth=0.8,
            bgcolor=legend_bg,
        ),
    )

    fig.update_xaxes(
        showline=True, linecolor=fg, linewidth=1, mirror=True,
        ticks="outside", tickwidth=1, ticklen=6,
        title_font=dict(family="Arial", size=14, color=fg),
        tickfont=dict(family="Arial", size=14, color=fg),
        showgrid=bool(show_grid),
        gridcolor=grid,
        gridwidth=0.5,
        zeroline=False,
    )
    fig.update_yaxes(
        showline=True, linecolor=fg, linewidth=1, mirror=True,
        ticks="outside", tickwidth=1, ticklen=6,
        title_font=dict(family="Arial", size=14, color=fg),
        tickfont=dict(family="Arial", size=14, color=fg),
        showgrid=bool(show_grid),
        gridcolor=grid,
        gridwidth=0.5,
        zeroline=False,
    )

def add_ppt_download(fig, filename_base: str, *, show_grid: bool = True):
    buf = io.BytesIO()
    try:
        fig_export = go.Figure(fig)  # clone
        apply_preli_style(fig_export, base="light", show_grid=show_grid, for_export=True)  # transparent export
        fig_export.write_image(buf, format="png", width=1600, height=900, scale=2)
    except Exception:
        st.info("Static image export not available. Install `kaleido` to enable PNG downloads.")
        return

    buf.seek(0)
    st.download_button(
        label="⬇️ Download PNG for PPT (transparent)",
        data=buf,
        file_name=f"{filename_base}.png",
        mime="image/png",
        key=f"dl_{filename_base}",
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

    # Time columns come in many variants depending on export settings.
    ensure(
        "Total Time",
        "totaltime", "total time", "total_time",
        "total time(s)", "total time (s)", "total time[s]",
        "totaltime(s)", "totaltime (s)",
    )
    ensure("Time", "time(s)", "time (s)", "time", "time[s]")
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

    # Need a time column
    if not time_col or time_col not in d.columns:
        return pd.Series(np.zeros(len(d)), index=d.index, dtype="float64")
    raw = d[time_col]

    # Parse to seconds.
    # Neware exports can mix numeric seconds and timedelta-like strings (especially in demo packs).
    # Instead of switching wholesale to timedelta parsing, combine both representations.
    def _parse_seconds(x: pd.Series) -> pd.Series:
        sec_num = pd.to_numeric(x, errors="coerce")
        td = pd.to_timedelta(x.astype(str), errors="coerce")
        sec_td = td.dt.total_seconds()
        return sec_num.where(sec_num.notna(), sec_td)

    # Prefer the requested time column, but if it is sparsely populated (common in some demo exports),
    # fall back to the other time column when it provides better coverage.
    sec_primary = _parse_seconds(raw)
    sec = sec_primary
    if str(time_col).lower() == "total time" and "Time" in d.columns:
        sec_alt = _parse_seconds(d["Time"])
        if sec_primary.notna().mean() < 0.80 and sec_alt.notna().mean() > sec_primary.notna().mean():
            sec = sec_alt

    # Forward fill to handle occasional missing timestamps.
    sec = sec.ffill().fillna(0.0)

    # 2) If it's already absolute / monotonic (no resets), just return it.
    # Don't rely on the column name; some exports label step-local time as 'Total Time'.
    if (sec.diff().fillna(0) >= 0).all():
        return sec - float(sec.iloc[0])

    # 3) Stitch step-local time using a SAFE grouping key
    # Use run-length segments so repeating Step_Index values across cycles don't get merged.
    if "Step_Index" in d.columns:
        stp = pd.to_numeric(d["Step_Index"], errors="coerce").ffill().fillna(0)
        boundary = stp.ne(stp.shift())
        if cycle_col in d.columns:
            boundary |= d[cycle_col].ne(d[cycle_col].shift())
        boundary |= sec.diff().fillna(0) < 0
        boundary |= (sec == 0) & (sec.shift().fillna(0) > 0)
        gkey = boundary.cumsum().astype(int).astype(str)
    else:
        # fallback: run-length segments (cycle/step changes or time reset)
        boundary = pd.Series(False, index=d.index)
        if cycle_col in d.columns:
            boundary |= d[cycle_col].ne(d[cycle_col].shift())
        if step_col in d.columns:
            boundary |= d[step_col].astype(str).ne(d[step_col].astype(str).shift())
        boundary |= sec.diff().fillna(0) < 0
        boundary |= (sec == 0) & (sec.shift().fillna(0) > 0)
        gkey = boundary.cumsum().astype(int).astype(str)

    within = sec.groupby(gkey).transform(lambda s: s - s.iloc[0])

    # stable order of groups (by first appearance)
    _, first_idx = np.unique(gkey.to_numpy(), return_index=True)
    ordered_groups = gkey.iloc[np.sort(first_idx)]

    offsets: Dict[str, float] = {}
    total = 0.0
    for grp in ordered_groups:
        offsets[grp] = total
        w = within[gkey == grp].dropna()
        dur = float(w.max()) if not w.empty else 0.0

        # add one typical dt to avoid boundary duplicates
        sg = sec[gkey == grp].dropna()
        dt = float(sg.diff().median()) if len(sg) > 1 else 0.0
        if not np.isfinite(dt) or dt < 0:
            dt = 0.0

        total += dur + dt

    stitched = within + gkey.map(offsets).astype(float)
    return stitched - float(stitched.iloc[0])

def auto_rotate_xticks(fig, ax, rotation=45, ha="right", pad_px=2):
    """
    Keep x tick labels horizontal unless they overlap or are clipped.
    Returns True if it rotated.
    """
    # Ensure text positions/sizes are finalized
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    labels = [t for t in ax.get_xticklabels() if t.get_text()]
    if not labels:
        return False

    bboxes = [t.get_window_extent(renderer=renderer) for t in labels]
    bboxes = sorted(bboxes, key=lambda b: b.x0)

    # Overlap check (adjacent bboxes)
    overlap = any(bboxes[i].x1 + pad_px > bboxes[i + 1].x0 for i in range(len(bboxes) - 1))

    # Clipping check (label extends beyond axes)
    ax_bb = ax.get_window_extent(renderer=renderer)
    clipped = any(b.x0 < ax_bb.x0 or b.x1 > ax_bb.x1 for b in bboxes)

    if overlap or clipped:
        for t in ax.get_xticklabels():
            t.set_rotation(rotation)
            t.set_ha(ha)
        fig.tight_layout()
        return True

    return False

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
# dQ/dV helpers (minimal / defaults)
# ----------------------------
@dataclass
class DQDVOptions:
    dv_min: float = 0.005       # keep points only when |ΔV| >= dv_min
    dv_eps: float = 1e-6        # drop derivatives when |ΔV| < dv_eps
    derivative: str = "central" # forward|backward|central
    shift: int = -1             # -1,0,+1 voltage alignment for central difference
    smooth: int = 3             # moving-average window on dQ/dV (points)
    smooth_center: bool = True

def _reduce_by_dv(v: np.ndarray, dv_min: float) -> np.ndarray:
    """Keep first point, then keep a point when |V - V_last_kept| >= dv_min."""
    if len(v) == 0:
        return np.array([], dtype=int)
    if dv_min <= 0:
        return np.arange(len(v), dtype=int)
    keep = [0]
    last = float(v[0])
    for i in range(1, len(v)):
        if abs(float(v[i]) - last) >= dv_min:
            keep.append(i)
            last = float(v[i])
    return np.asarray(keep, dtype=int)

def compute_dqdv_segment_df(seg: pd.DataFrame, vcol: str, qcol: str, opts: DQDVOptions) -> pd.DataFrame:
    """Compute dQ/dV for one segment (charge OR discharge) given voltage + capacity columns."""
    if seg is None or seg.empty or (vcol not in seg.columns) or (qcol not in seg.columns):
        return pd.DataFrame(columns=["Voltage", "dQdV"])

    s = seg[[vcol, qcol]].dropna().copy()
    if s.empty:
        return pd.DataFrame(columns=["Voltage", "dQdV"])

    v = pd.to_numeric(s[vcol], errors="coerce").to_numpy(dtype=float)
    q = pd.to_numeric(s[qcol], errors="coerce").to_numpy(dtype=float)

    m = np.isfinite(v) & np.isfinite(q)
    v = v[m]
    q = q[m]
    if len(v) < 2:
        return pd.DataFrame(columns=["Voltage", "dQdV"])

    # point reduction to avoid CV-region spikes from tiny dV
    idx = _reduce_by_dv(v, float(opts.dv_min))
    v = v[idx]
    q = q[idx]
    if len(v) < 2:
        return pd.DataFrame(columns=["Voltage", "dQdV"])

    der = str(opts.derivative).lower()
    shift = int(opts.shift)
    dv_eps = float(opts.dv_eps)

    if der not in {"forward", "backward", "central"}:
        der = "central"
    if shift not in {-1, 0, 1}:
        shift = -1

    if der == "forward":
        dv = np.diff(v)
        dq = np.diff(q)
        ok = np.abs(dv) >= dv_eps
        dqdv = dq[ok] / dv[ok]
        v_out = v[:-1][ok]
    elif der == "backward":
        dv = v[1:] - v[:-1]
        dq = q[1:] - q[:-1]
        ok = np.abs(dv) >= dv_eps
        dqdv = dq[ok] / dv[ok]
        v_out = v[1:][ok]
    else:  # central
        if len(v) < 3:
            dv = np.diff(v)
            dq = np.diff(q)
            ok = np.abs(dv) >= dv_eps
            dqdv = dq[ok] / dv[ok]
            v_out = v[:-1][ok]
        else:
            dv = v[2:] - v[:-2]
            dq = q[2:] - q[:-2]
            ok = np.abs(dv) >= dv_eps
            dqdv = dq[ok] / dv[ok]
            if shift == -1:
                v_out = v[:-2][ok]
            elif shift == 0:
                v_out = v[1:-1][ok]
            else:
                v_out = v[2:][ok]

    out = pd.DataFrame({"Voltage": v_out, "dQdV": dqdv})

    # smoothing
    w = int(opts.smooth)
    if w > 1 and len(out) > 1:
        out["dQdV"] = (
            out["dQdV"]
            .rolling(window=w, center=bool(opts.smooth_center), min_periods=1)
            .mean()
        )

    return out


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

def insert_line_breaks_generic(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    seg_cycle: bool = False,
    seg_step: bool = False,
    seg_cap_reset: bool = False,
    seg_current_flip: bool = False,
    cap_col_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Insert NaN rows in x/y to force Plotly line breaks.
    """
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return df

    d = df.loc[:, ~pd.Index(df.columns).duplicated()].reset_index(drop=True).copy()
    break_pos: List[int] = []
    n = len(d)

    # 1) Cycle changes
    if seg_cycle and "Cycle Index" in d.columns:
        cyc = d["Cycle Index"].to_numpy()
        for i in range(n - 1):
            if pd.notna(cyc[i]) and pd.notna(cyc[i + 1]) and cyc[i + 1] != cyc[i]:
                break_pos.append(i)

    # 2) Step changes
    if seg_step:
        if "Step_Index" in d.columns:
            stp = pd.to_numeric(d["Step_Index"], errors="coerce").ffill().fillna(0).astype(int)
            for i in range(n - 1):
                if stp.iloc[i + 1] != stp.iloc[i]:
                    break_pos.append(i)
        elif "Step Type" in d.columns:
            stp = d["Step Type"].astype(str)
            for i in range(n - 1):
                if stp.iloc[i + 1] != stp.iloc[i]:
                    break_pos.append(i)

# 3) Capacity resets (only if we have a usable capacity column)
    if seg_cap_reset and isinstance(cap_col_name, str) and cap_col_name in d.columns:
        cap = pd.to_numeric(d[cap_col_name], errors="coerce").fillna(0.0)
        idxs = cap[(cap.shift(-1) == 0) & (cap > 0)].index.tolist()
        break_pos.extend(idxs)

    # 4) Current sign flips (very good for separating chg/dchg if Step Type is messy)
    if seg_current_flip and "Current(mA)" in d.columns:
        cur = pd.to_numeric(d["Current(mA)"], errors="coerce")
        sgn = np.sign(cur)
        for i in range(n - 1):
            a, b = sgn.iloc[i], sgn.iloc[i + 1]
            if pd.notna(a) and pd.notna(b) and a != 0 and b != 0 and a != b:
                break_pos.append(i)

    if not break_pos:
        return d

    pieces = []
    last = 0
    for idx in sorted(set(break_pos)):
        pieces.append(d.iloc[last:idx + 1])
        nan_row = {c: (np.nan if c in [x_col, y_col] else d.iloc[idx].get(c)) for c in d.columns}
        pieces.append(pd.DataFrame([nan_row]))
        last = idx + 1
    pieces.append(d.iloc[last:])

    return _concat_nonempty(pieces)


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
# Demo dataset utilities
# ----------------------------
def _safe_demo_name(src: str, idx: int) -> str:
    """Generate a stable, non-identifying display name from a source string."""
    s = Path(str(src)).stem
    s_low = s.lower()
    if "ref" in s_low:
        group = "Ref"
    elif "7" in s_low:
        group = "7pct"
    else:
        group = "Sample"
    return f"Demo_{group}_{idx+1}"

def _downsample_light(df: pd.DataFrame, max_rows: int = 60000) -> pd.DataFrame:
    """Downsample large frames while keeping step boundaries."""
    if df is None or df.empty or len(df) <= max_rows:
        return df
    d = df.reset_index(drop=True)
    if "Step_Index" in d.columns:
        pieces = []
        for _, g in d.groupby("Step_Index", sort=False):
            n = len(g)
            if n <= 2000:
                pieces.append(g)
            else:
                # keep endpoints + uniform sampling
                keep_idx = np.unique(np.concatenate([
                    np.array([0, n-1]),
                    np.linspace(0, n-1, 2000).astype(int)
                ]))
                pieces.append(g.iloc[keep_idx])
        d2 = _concat_nonempty(pieces).reset_index(drop=True)
        if len(d2) <= max_rows:
            return d2
        # if still too big, fall back to uniform sample
        keep_idx = np.linspace(0, len(d2)-1, max_rows).astype(int)
        return d2.iloc[keep_idx].reset_index(drop=True)
    # no step index: uniform
    keep_idx = np.linspace(0, len(d)-1, max_rows).astype(int)
    return d.iloc[keep_idx].reset_index(drop=True)

def _hybrid_anonymize_df(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """
    Hybrid anonymization:
    - drops timestamp-like columns
    - lightly rescales time/capacity to reduce identifiability
    - downsamples for lightweight demo
    """
    if df is None or df.empty:
        return df

    rng = np.random.default_rng(seed)
    d = df.copy()

    # Drop timestamp-like columns (keep relative 'Time' / 'Total Time')
    drop_cols = [c for c in d.columns if c in [
        "Timestamp", "Record Time", "DateTime", "Date Time", "System Time", "Local Time"
    ]]
    if drop_cols:
        d = d.drop(columns=drop_cols, errors="ignore")

    # Light rescale (keeps shapes but hides exact numbers)
    # Time
    for tc in ["Time"]:
        if tc in d.columns:
            s = pd.to_numeric(d[tc], errors="coerce")
            if s.notna().any():
                scale = float(rng.uniform(0.85, 1.15))
                d[tc] = s * scale

    # Capacity-like columns
    cap_cols = [
        "Spec. Cap.(mAh/g)", "Capacity(mAh)",
        "Chg. Spec. Cap.(mAh/g)", "DChg. Spec. Cap.(mAh/g)",
        "Chg. Cap.(mAh)", "DChg. Cap.(mAh)",
        "Charge_Capacity(mAh)", "Discharge_Capacity(mAh)",
    ]
    cap_scale = float(rng.uniform(0.85, 1.15))
    for cc in cap_cols:
        if cc in d.columns:
            s = pd.to_numeric(d[cc], errors="coerce")
            if s.notna().any():
                d[cc] = s * cap_scale

    # Voltage tiny shift (optional, very small)
    if "Voltage(V)" in d.columns:
        v = pd.to_numeric(d["Voltage(V)"], errors="coerce")
        if v.notna().any():
            d["Voltage(V)"] = v + float(rng.uniform(-0.01, 0.01))

    return _downsample_light(d)

DEMO_VERSION = 1069
def load_demo_frames(_demo_version: str = DEMO_VERSION) -> Dict[str, pd.DataFrame]:
    """
    Load demo frames.

    We ONLY use the built-in synthetic dataset (no manifest / external demo pack).
    This guarantees identical behavior locally and on Streamlit Cloud.
    """

    # ----------------------------
    # Synthetic demo (no external files needed)
    # ----------------------------
    rng = np.random.default_rng(42)
    demo: Dict[str, pd.DataFrame] = {}

    # Make Time look like Neware-like elapsed time strings so pd.to_timedelta(...) works as expected downstream.
    def _time_str_from_seconds(t_s: np.ndarray) -> np.ndarray:
        return pd.to_timedelta(t_s, unit="s").astype(str).to_numpy()

    def make_profile(n, kind="charge"):
        x = np.linspace(0, 1, n, endpoint=False)
        if kind == "charge":
            return 0.05 + 0.95 * (1 / (1 + np.exp(-8 * (x - 0.35))))
        else:
            return 0.05 + 0.95 * (1 - 1 / (1 + np.exp(-8 * (x - 0.35))))

    def step_df(step_idx, cycle, step_type, t0_s, duration_s, npts, current_mA, q_spec_end, am_g, v_kind):
        # endpoint=False avoids duplicate timestamps at step boundaries
        t_rel = np.linspace(0, duration_s, npts, endpoint=False)
        t_abs = t0_s + t_rel

        v = make_profile(npts, kind=v_kind)
        q_spec = np.linspace(0, q_spec_end, npts, endpoint=False)
        q_mAh = q_spec * am_g

        row = {
            "Step_Index": np.full(npts, step_idx, dtype=int),
            "Cycle Index": np.full(npts, cycle, dtype=int),
            "Step Type": np.full(npts, step_type),
            "Time": t_rel,
            "Total Time": t_abs,
            "Voltage(V)": v,
            "Current(mA)": np.full(npts, current_mA),
            "Spec. Cap.(mAh/g)": q_spec,
            "Capacity(mAh)": q_mAh,
            "Charge_Capacity(mAh)": np.where(current_mA > 0, q_mAh, 0.0),
            "Discharge_Capacity(mAh)": np.where(current_mA < 0, q_mAh, 0.0),
            "Chg. Spec. Cap.(mAh/g)": np.where(current_mA > 0, q_spec, np.nan),
            "DChg. Spec. Cap.(mAh/g)": np.where(current_mA < 0, q_spec, np.nan),
            "Chg. Cap.(mAh)": np.where(current_mA > 0, q_mAh, np.nan),
            "DChg. Cap.(mAh)": np.where(current_mA < 0, q_mAh, np.nan),
        }
        return pd.DataFrame(row), (t0_s + duration_s)

    def rest_df(step_idx, cycle, t0_s, duration_s, npts, v_level):
        t_rel = np.linspace(0, duration_s, npts, endpoint=False)
        t_abs = t0_s + t_rel
        v = np.full(npts, v_level) + rng.normal(0, 0.0015, size=npts)

        row = {
            "Step_Index": np.full(npts, step_idx, dtype=int),
            "Cycle Index": np.full(npts, cycle, dtype=int),
            "Step Type": np.full(npts, "Rest"),
            "Time": t_rel,
            "Total Time": t_abs,
            "Voltage(V)": v,
            "Current(mA)": np.zeros(npts),
            "Spec. Cap.(mAh/g)": np.zeros(npts),
            "Capacity(mAh)": np.zeros(npts),
            "Charge_Capacity(mAh)": np.zeros(npts),
            "Discharge_Capacity(mAh)": np.zeros(npts),
            "Chg. Spec. Cap.(mAh/g)": np.nan,
            "DChg. Spec. Cap.(mAh/g)": np.nan,
            "Chg. Cap.(mAh)": np.nan,
            "DChg. Cap.(mAh)": np.nan,
        }
        return pd.DataFrame(row), (t0_s + duration_s)

    def pulse_df(step_idx, cycle, t0_s, duration_s, npts, current_mA, v_pre, dv):
        t_rel = np.linspace(0, duration_s, npts, endpoint=False)
        t_abs = t0_s + t_rel
        v = v_pre + dv * (1 - np.exp(-t_rel / 2.5))
        v = v + rng.normal(0, 0.0015, size=npts)

        row = {
            "Step_Index": np.full(npts, step_idx, dtype=int),
            "Cycle Index": np.full(npts, cycle, dtype=int),
            "Step Type": np.full(npts, "Pulse"),
            "Time": _time_str_from_seconds(t_abs),
            "Voltage(V)": v,
            "Current(mA)": np.full(npts, current_mA),
            "Spec. Cap.(mAh/g)": np.zeros(npts),
            "Capacity(mAh)": np.zeros(npts),
            "Charge_Capacity(mAh)": np.zeros(npts),
            "Discharge_Capacity(mAh)": np.zeros(npts),
            "Chg. Spec. Cap.(mAh/g)": np.nan,
            "DChg. Spec. Cap.(mAh/g)": np.nan,
            "Chg. Cap.(mAh)": np.nan,
            "DChg. Cap.(mAh)": np.nan,
        }
        return pd.DataFrame(row), (t0_s + duration_s)

    # Build 4 demo files
    specs = [
        ("Demo-7pct_1", 980, 920),
        ("Demo-7pct_2", 960, 905),
        ("Demo-Ref_1", 780, 750),
        ("Demo-Ref_2", 790, 760),
    ]

    # Longer steps so elapsed time ends up ~50 h (nice-looking “real test” scale)
    CYCLES = 10
    CHG_S = 7200   # 2 h
    DCHG_S = 7200  # 2 h
    REST_S = 1800  # 0.5 h

    # Points per step (keep files not too huge, still smooth)
    NPTS_CHG = 600
    NPTS_DCHG = 600
    NPTS_REST = 120

    for name, qchg1, qdch1 in specs:
        am_g = float(rng.uniform(0.006, 0.015))  # fake active mass
        frames = []
        step_idx = 1
        t0_s = 0.0  # absolute elapsed time (seconds)

        for cyc in range(1, CYCLES + 1):
            fade = 1.0 - 0.015 * (cyc - 1)
            qchg = qchg1 * fade * float(rng.uniform(0.98, 1.02))
            qdch = qdch1 * fade * float(rng.uniform(0.98, 1.02))

            df, t0_s = step_df(step_idx, cyc, "CC Chg", t0_s, CHG_S, NPTS_CHG, +100.0, qchg, am_g, "charge")
            frames.append(df); step_idx += 1

            df, t0_s = rest_df(step_idx, cyc, t0_s, REST_S, NPTS_REST, 1.0)
            frames.append(df); step_idx += 1

            df, t0_s = step_df(step_idx, cyc, "CC DChg", t0_s, DCHG_S, NPTS_DCHG, -100.0, qdch, am_g, "discharge")
            frames.append(df); step_idx += 1

            df, t0_s = rest_df(step_idx, cyc, t0_s, REST_S, NPTS_REST, 0.05)
            frames.append(df); step_idx += 1

        # Optional DCIR pulse block at cycle CYCLES+1
        cyc = CYCLES + 1
        soc_levels = [80, 50, 20, 5]
        for soc in soc_levels:
            for _ in range(2):
                v_pre = 0.2 + 0.0075 * soc
                df, t0_s = rest_df(step_idx, cyc, t0_s, 120, 60, v_pre)
                frames.append(df); step_idx += 1

                df, t0_s = pulse_df(step_idx, cyc, t0_s, 18, 60, -500.0, v_pre, dv=-0.06)
                frames.append(df); step_idx += 1

        d = _concat_nonempty(frames)
        d = normalize_neware_headers(d)
        d = infer_rest_step(d)
        d["__file"] = name
        d["__family"] = family_from_filename(name)
        d.attrs["active_mass_g"] = am_g
        demo[name] = d.reset_index(drop=True)

    return demo


def _activate_demo():
    demo = load_demo_frames()
    st.session_state["parsed_by_file"] = demo
    st.session_state["uploaded_names_cache"] = sorted(list(demo.keys()))
    st.session_state["demo_loaded"] = True


# ----------------------------
# Header / logo
# ----------------------------
LOGO_LIGHT_PATH = HERE / "logo_light.png"
LOGO_DARK_PATH  = HERE / "logo_dark.png"
IS_DARK = (BASE_THEME == "dark")
logo_path = LOGO_DARK_PATH if IS_DARK else LOGO_LIGHT_PATH

c1, c2, c3 = st.columns([1, 1, 1])
with c2:
    if logo_path.exists():
        st.image(str(logo_path))
    else:
        st.caption(f"Logo missing: {logo_path.name}")

st.title("🔋 BATTERY CELL DATA — VISUALIZER 📈")
st.caption("::::::: Built by the Preli team ::::::::")


# ----------------------------
# Sidebar: upload + parse (GATED)
# ----------------------------
with st.sidebar.form("upload_form", clear_on_submit=False):
    uploaded_files = st.file_uploader(
        "Drop Neware .ndax files (multiple allowed)",
        type=["ndax"],
        accept_multiple_files=True,
    )
    parse_now = st.form_submit_button("🚀 **launch files**")

# Provide an easy way to clear state
top_l, top_r = st.columns([6, 1])
with top_r:
    if "parsed_by_file" in st.session_state:
        if st.button("🧹 Reset", key="clear_parsed_main"):
            for k in ["parsed_by_file", "file_checks", "uploaded_names_cache", "selected_files","demo_loaded"]:
                st.session_state.pop(k, None)
            st.rerun()

if st.session_state.get("demo_loaded", False):
        st.info("**Demo dataset loaded.** You can still upload your own `.ndax` files from the sidebar to analyze real data.")

if not uploaded_files and "parsed_by_file" not in st.session_state:
    row_l, row_r = st.columns([6, 2])

    with row_l:
        st.info("Upload one or more NDAX files and press 🚀 **launch** or simply explore the app with a demo dataset.")

    with row_r:
        st.markdown("""
            <style>
            div.stButton > button {
            background: #0f5280;
            color: white;
            border-radius: 999px;
            padding: 0.5rem 1rem;
            border: 0;
            font-weight: 700;
            }
            div.stButton > button:hover { opacity: 0.9; }
            </style>
            """, unsafe_allow_html=True)
        if st.button(" ▶️ Explore demo", type="primary", key="load_demo_main"):
            _activate_demo(); st.rerun()
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
    st.session_state["demo_loaded"] = False

if "parsed_by_file" not in st.session_state:
    st.info("Upload your files, then click 🚀 **launch files**.")
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
    "dQ/dV",
    "Capacity vs Cycle", "Capacity & CE", "DCIR",
    "ICE Boxplot", "Capacity Fade Boxplot", "Raw File Preview",
]


view = st.segmented_control(
    "View selector",
    options=PAGES,
    default="Capacity vs Cycle",
    key="view_selector_main",
    label_visibility="collapsed",
)
# ----------------------------
# Raw File Preview
# ----------------------------
if view == "Raw File Preview":
    st.subheader("Preview parsed data (first rows per file)")
    max_rows = st.number_input("Rows per file", min_value=5, max_value=10000, value=900, step=50)
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
            key="xy_x_select",
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
            key="xy_y_multi",
        )

    with row1[2]:
        rolling = st.number_input("Rolling mean window (pts)", 1, 9999, 1, 1, key="xy_roll")
        use_global_colors_xy = st.checkbox(
            "Use global colors here",
            value=True,
            help="If off, Plotly default color cycle is used.",
            key="xy_use_global_colors",
        )

    row2 = st.columns([1, 1, 0.6])
    with row2[0]:
        y_min = st.text_input("Y min (blank=auto)", "", key="xy_ymin")
    with row2[1]:
        y_max = st.text_input("Y max (blank=auto)", "", key="xy_ymax")

    row3 = st.columns([1, 1, 0.6])
    with row3[0]:
        x_min = st.text_input("X min (blank=auto)", "", key="xy_xmin")
    with row3[1]:
        x_max = st.text_input("X max (blank=auto)", "", key="xy_xmax")

    st.session_state.xy_x = x_col
    st.session_state.xy_y = y_cols

    if not y_cols:
        st.warning("Pick at least one Y column to plot.")
        st.stop()

    fig = go.Figure()
    added = False

    CAP_LIKE = {
    "Spec. Cap.(mAh/g)", "Capacity(mAh)",
    "Chg. Spec. Cap.(mAh/g)", "DChg. Spec. Cap.(mAh/g)",
    "Chg. Cap.(mAh)", "DChg. Cap.(mAh)",
    }

    for src in selected_files:
        df = parsed_by_file[src]
        if x_col not in df.columns:
            continue

        # always define x_used + df_local
        x_used = x_col
        df_local = df

        # If aligning time, create a stitched time axis
        if align_t0:
            df_local = df_local.dropna(subset=[x_col]).copy()
            df_local["_x"] = build_global_time_seconds(
                df_local,
                time_col=x_col,
                cycle_col="Cycle Index",
                step_col="Step Type",
            )
            x_used = "_x"

        # optional smoothing (per file)
        if rolling > 1:
            try:
                df_local = df_local.sort_values(x_used)
            except Exception:
                pass
            for y in y_cols:
                if y in df_local.columns:
                    df_local[y] = pd.to_numeric(df_local[y], errors="coerce")
                    df_local[y] = df_local[y].rolling(rolling, min_periods=1).mean()

        for y in y_cols:
            if y not in df_local.columns or x_used not in df_local.columns:
                continue

            # Build a small frame with the extra columns needed for segmentation
            cols_needed = [x_used, y]
            for extra in ["Cycle Index", "Step Type", "Current(mA)"]:
                if extra in df_local.columns and extra not in cols_needed:
                    cols_needed.append(extra)

            df_plot = df_local[cols_needed].copy()

            # Insert breaks for V–Q / specQ plots (and generally whenever Step/Cycle exists)
            use_breaks = (not align_t0) and (
                (x_col in CAP_LIKE) or
                ("Step Type" in df_plot.columns) or
                ("Cycle Index" in df_plot.columns)
            )

            if use_breaks:
                df_plot = insert_line_breaks_generic(
                    df_plot,
                    x_col=x_used,
                    y_col=y,
                    seg_cycle=("Cycle Index" in df_plot.columns),
                    seg_step=("Step Type" in df_plot.columns),
                    seg_cap_reset=(x_col in CAP_LIKE),          # only when X is capacity-like
                    cap_col_name=(x_used if x_col in CAP_LIKE else None),
                    seg_current_flip=("Current(mA)" in df_plot.columns),
                )

            # Keep NaNs (they are the line breaks), but skip truly empty series
            if df_plot.dropna(subset=[x_used, y]).empty:
                continue
            s = df_plot

            c = color_for_src(src) if use_global_colors_xy else None
            fig.add_trace(go.Scatter(
                x=s[x_used],
                y=s[y],
                mode=("lines+markers" if show_markers else "lines"),
                name=f"{pretty_src(src)} — {y}",
                line=dict(color=c, width=line_width) if c else dict(width=line_width),
                marker=dict(size=marker_size),
            ))
            added = True

    if not added:
        st.warning("No data drawn — check your column choices (some files may not contain those columns).")
        st.stop()

    # axis ranges (only if user typed something + it’s numeric)
    try:
        if x_min != "" or x_max != "":
            lo = float(x_min) if x_min != "" else None
            hi = float(x_max) if x_max != "" else None
            fig.update_xaxes(range=[lo, hi])
    except Exception:
        pass

    try:
        if y_min != "" or y_max != "":
            lo = float(y_min) if y_min != "" else None
            hi = float(y_max) if y_max != "" else None
            fig.update_yaxes(range=[lo, hi])
    except Exception:
        pass

    fig.update_layout(
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
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
        s = s.dropna(subset=["_t", vcol])
        # IMPORTANT: Voltage–Time must be ordered by absolute time.
        # Sorting by Step_Index can reorder time when Step_Index repeats (causes crossing lines).
        s = s.sort_values("_t")
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

    apply_preli_style(fig_vt,base=BASE_THEME,show_grid=show_grid_global)  
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
        # default to cycle 1 if present, else the smallest cycle
        default_cycle = 1 if 1 in cycles_available else cycles_available[0]
        default_index = 1 + cycles_available.index(default_cycle)  # +1 because "All" is index 0

        sel_cycle = st.selectbox(
            "Cycle",
            ["All"] + cycles_available,
            index=default_index,
            key="vq_cycle_select",
        )

        # only show the range slider when plotting "All" (keeps it fast + clean)
        if sel_cycle == "All":
            cmin, cmax = int(min(cycles_available)), int(max(cycles_available))
            rng = st.slider("Cycle range (optional)", cmin, cmax, (cmin, cmax), 1, key="vq_cycle_range")
        else:
            rng = None

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

    apply_preli_style(fig_vq, base=BASE_THEME, show_grid=show_grid_global)   # <-- on-screen adapts to Streamlit theme
    st.plotly_chart(fig_vq, width="stretch", config=CAMERA_CFG)
    add_ppt_download(fig_vq, filename_base="voltage_capacity")  # <-- exports light copy
    st.stop()


# ----------------------------
# dQ/dV
# ----------------------------
if view == "dQ/dV":
    st.subheader("dQ/dV")

    vcol = G.get("voltage")
    cyc_col = G.get("cycle") if (G.get("cycle") in union_cols) else None
    tcol = G.get("time") if (G.get("time") in union_cols) else None

    if not vcol:
        st.warning("Couldn’t detect voltage column.")
        st.stop()
    if not cyc_col:
        st.info("Need Cycle Index to plot dQ/dV.")
        st.stop()

    mode = st.radio(
        "Mode",
        ["Single cycle (compare files)", "Cycle overlay (one file)"],
        horizontal=True,
        key="dqdv_mode",
    )

    # Defaults (same spirit as your script; no extra tuning controls here)
    opts = DQDVOptions()

    def _get_current_mA(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]:
        if "Current(mA)" in df.columns:
            s = pd.to_numeric(df["Current(mA)"], errors="coerce")
            return s, "Current(mA)"
        if "Current(A)" in df.columns:
            s = pd.to_numeric(df["Current(A)"], errors="coerce") * 1000.0
            return s, "Current(A)"
        return None, None

    def _pick_qcols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        base_cap = G.get("capacity")
        ch_qcol = None
        for cand in ["Chg. Spec. Cap.(mAh/g)", "Charge_Capacity(mAh)", "Charge Capacity(mAh)", base_cap]:
            if cand and cand in df.columns:
                ch_qcol = cand
                break
        dch_qcol = None
        for cand in ["DChg. Spec. Cap.(mAh/g)", "Discharge_Capacity(mAh)", "Discharge Capacity(mAh)", base_cap]:
            if cand and cand in df.columns:
                dch_qcol = cand
                break
        return ch_qcol, dch_qcol

    def _cycles_from_df(df: pd.DataFrame) -> List[int]:
        if cyc_col not in df.columns:
            return []
        s = pd.to_numeric(df[cyc_col], errors="coerce").dropna()
        vals = sorted(pd.unique(s))
        out = []
        for c in vals:
            try:
                cf = float(c)
                if cf.is_integer():
                    out.append(int(cf))
            except Exception:
                pass
        return out

    fig = go.Figure()
    any_qcol = None

    if mode == "Single cycle (compare files)":
        # Collect available cycles across selected files
        cyc_values = []
        for df in frames_selected:
            if cyc_col in df.columns:
                cyc_values.append(pd.to_numeric(df[cyc_col], errors="coerce").dropna())
        cycles_available = sorted(pd.unique(pd.concat(cyc_values, ignore_index=True))) if cyc_values else []
        cycles_available = [int(c) for c in cycles_available if float(c).is_integer()]

        if not cycles_available:
            st.info("No cycle data found in the selected files.")
            st.stop()

        default_cycle = 1 if 1 in cycles_available else cycles_available[0]
        sel_cycle = st.selectbox(
            "Cycle",
            cycles_available,
            index=cycles_available.index(default_cycle),
            key="dqdv_cycle_select_compare",
        )

        for src_name in selected_files:
            df = parsed_by_file.get(src_name)
            if df is None or df.empty:
                continue
            if cyc_col not in df.columns or vcol not in df.columns:
                continue

            d = df[df[cyc_col] == sel_cycle].copy()
            if d.empty:
                continue

            # sort by time if possible (keeps segments clean)
            if tcol and tcol in d.columns:
                d[tcol] = pd.to_numeric(d[tcol], errors="coerce")
                d = d.sort_values(tcol)

            cur_mA, _ = _get_current_mA(d)
            if cur_mA is None:
                continue

            ch_qcol, dch_qcol = _pick_qcols(d)
            if not ch_qcol and not dch_qcol:
                continue
            any_qcol = any_qcol or (ch_qcol or dch_qcol)

            ch_seg = d[cur_mA > 0.0]
            dch_seg = d[cur_mA < 0.0]

            ch_curve = compute_dqdv_segment_df(ch_seg, vcol=vcol, qcol=ch_qcol, opts=opts) if ch_qcol else pd.DataFrame(columns=["Voltage","dQdV"])
            dch_curve = compute_dqdv_segment_df(dch_seg, vcol=vcol, qcol=dch_qcol, opts=opts) if dch_qcol else pd.DataFrame(columns=["Voltage","dQdV"])

            color = color_for_src(src_name)

            if not ch_curve.empty:
                fig.add_trace(go.Scatter(
                    x=ch_curve["Voltage"], y=ch_curve["dQdV"],
                    mode="lines",
                    name=pretty_src(src_name),
                    legendgroup=pretty_src(src_name),
                    line=dict(color=color, width=line_width, dash="solid"),
                ))
            if not dch_curve.empty:
                fig.add_trace(go.Scatter(
                    x=dch_curve["Voltage"], y=dch_curve["dQdV"],
                    mode="lines",
                    name=pretty_src(src_name),
                    legendgroup=pretty_src(src_name),
                    showlegend=False,
                    line=dict(color=color, width=line_width, dash="solid"),
                ))

        title = f"dQ/dV vs Voltage — Cycle {sel_cycle}"

    else:
        # Overlay a cycle range for a single file
        if not selected_files:
            st.info("Select at least one file to plot.")
            st.stop()

        src_name = st.selectbox(
            "File",
            options=selected_files,
            format_func=pretty_src,
            key="dqdv_file_select_overlay",
        )
        df = parsed_by_file.get(src_name)
        if df is None or df.empty:
            st.info("Selected file has no data.")
            st.stop()

        cycles_available = _cycles_from_df(df)
        if not cycles_available:
            st.info("No cycle data found in that file.")
            st.stop()

        cmin, cmax = min(cycles_available), max(cycles_available)
        # sensible default range
        default_end = min(cmin + 9, cmax)
        c_start, c_end = st.slider(
            "Cycle range",
            min_value=cmin,
            max_value=cmax,
            value=(cmin, default_end),
            step=1,
            key="dqdv_cycle_range_overlay",
        )

        # Precompute palette for cycles
        cycle_list = list(range(c_start, c_end + 1))
        for i, cyc in enumerate(cycle_list):
            d = df[df[cyc_col] == cyc].copy()
            if d.empty:
                continue

            if tcol and tcol in d.columns:
                d[tcol] = pd.to_numeric(d[tcol], errors="coerce")
                d = d.sort_values(tcol)

            cur_mA, _ = _get_current_mA(d)
            if cur_mA is None:
                continue

            ch_qcol, dch_qcol = _pick_qcols(d)
            if not ch_qcol and not dch_qcol:
                continue
            any_qcol = any_qcol or (ch_qcol or dch_qcol)

            ch_seg = d[cur_mA > 0.0]
            dch_seg = d[cur_mA < 0.0]

            ch_curve = compute_dqdv_segment_df(ch_seg, vcol=vcol, qcol=ch_qcol, opts=opts) if ch_qcol else pd.DataFrame(columns=["Voltage","dQdV"])
            dch_curve = compute_dqdv_segment_df(dch_seg, vcol=vcol, qcol=dch_qcol, opts=opts) if dch_qcol else pd.DataFrame(columns=["Voltage","dQdV"])

            color = palette[i % len(palette)] if len(palette) else None

            label = f"Cycle {cyc}"
            if not ch_curve.empty:
                fig.add_trace(go.Scatter(
                    x=ch_curve["Voltage"], y=ch_curve["dQdV"],
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    line=dict(color=color, width=line_width, dash="solid"),
                ))
            if not dch_curve.empty:
                fig.add_trace(go.Scatter(
                    x=dch_curve["Voltage"], y=dch_curve["dQdV"],
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=False,
                    line=dict(color=color, width=line_width, dash="solid"),
                ))

        title = f"dQ/dV vs Voltage — {pretty_src(src_name)} (Cycles {c_start}–{c_end})"

    if len(fig.data) == 0:
        st.info("No dQ/dV data found for the selection.")
        st.stop()

    y_label = "dQ/dV (mAh/V)"
    if any_qcol and ("mAh/g" in str(any_qcol)):
        y_label = "dQ/dV (mAh/g/V)"

    fig.update_layout(
        template="plotly_white",
        title="",
        xaxis_title=vcol,
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, groupclick="togglegroup"),
    )
    if show_grid_global:
        fig.update_xaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)
        fig.update_yaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)

    apply_preli_style(fig, base=BASE_THEME, show_grid=show_grid_global)
    st.plotly_chart(fig, width="stretch", config=CAMERA_CFG)
    add_ppt_download(fig, filename_base="dqdv")
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

    apply_preli_style(fig_cap, base=BASE_THEME, show_grid=show_grid_global)   # <-- on-screen adapts to Streamlit theme
    st.plotly_chart(fig_cap, width="stretch", config=CAMERA_CFG)
    add_ppt_download(fig_cap, filename_base="capacity_vs_cycle")  # <-- exports light copy
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

    apply_preli_style(fig_ce, base=BASE_THEME, show_grid=show_grid_global)   
    st.plotly_chart(fig_ce, width="stretch", config=CAMERA_CFG)
    add_ppt_download(fig_ce, filename_base="ce_and_capacity",)  
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
                ha="center", va="center", fontsize=5,
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.9),
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(groups)
    auto_rotate_xticks(fig, ax, rotation=45, ha="right")
    ax.set_ylabel(y_label)
    ax.set_title(f"First-cycle capacity and ICE ")
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
# ----------------------------
# Capacity Fade Boxplot (cycle window)
# ----------------------------
if view == "Capacity Fade Boxplot":
    st.subheader("Capacity fade boxplot ")

    if not selected_files:
        st.info("Select at least one NDAX/NDAX demo file on the left.")
        st.stop()

    # Detect cycle bounds across selected files
    mins, maxs = [], []
    for src in selected_files:
        df = parsed_by_file.get(src)
        if df is None or df.empty or "Cycle Index" not in df.columns:
            continue
        cyc = pd.to_numeric(df["Cycle Index"], errors="coerce")
        cyc = cyc.dropna()
        if cyc.empty:
            continue
        mins.append(int(cyc.min()))
        maxs.append(int(cyc.max()))

    if not mins:
        st.error("Could not detect 'Cycle Index' in the selected files.")
        st.stop()

    min_cycle = int(min(mins))
    max_cycle = int(max(maxs))

    max_window = 100
    default_start = min_cycle
    default_end = min(min_cycle + max_window, max_cycle)

    c0, c1, c2, c3 = st.columns([2, 2, 2, 2])
    with c0:
        y_metric = st.selectbox(
            "Y metric",
            ["Discharge capacity", "Charge capacity", "CE (%)"],
            index=0,
            key="capfade_y_metric",
        )
    with c1:
        x_mode = st.selectbox("X axis", ["Cycle", "Group"], index=0, key="capfade_x_axis")


    start, end = st.slider(
        "**Cycle range** (100 cycle window)",
        min_value=min_cycle,
        max_value=max_cycle,
        value=(default_start, default_end),
        step=1,
        key="capfade_cycle_range",
    )

    if (end - start) > max_window:
        end = min(start + max_window, max_cycle)
        st.info(f"Window capped at {max_window} cycles → showing {start}–{end}")
    if end <= start:
        st.warning("End cycle must be greater than start cycle.")
        st.stop()

    # Build plotting table: one value per file per cycle
    ce_cell_type = "cathode" if cell_type_sel == "full" else cell_type_sel

    rows = []
    for src in selected_files:
        df = parsed_by_file.get(src)
        if df is None or df.empty or "Cycle Index" not in df.columns:
            continue

        fam = family_from_filename(src)
        ce_df = compute_ce(df, cell_type=ce_cell_type)
        if ce_df is None or ce_df.empty:
            continue

        ce_df = ce_df[(ce_df["cycle"] >= start) & (ce_df["cycle"] <= end)].copy()
        if ce_df.empty:
            continue

        if y_metric == "Discharge capacity":
            ycol = "q_dch"
            yvals = ce_df[ycol]
        elif y_metric == "Charge capacity":
            ycol = "q_chg"
            yvals = ce_df[ycol]
        else:
            ycol = "ce"
            yvals = ce_df[ycol]

        for cyc, val in zip(ce_df["cycle"].tolist(), pd.to_numeric(yvals, errors="coerce").tolist()):
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                continue
            rows.append({"Cycle": int(cyc), "Value": float(val), "Group": fam, "File": src})

    if not rows:
        st.info("No data found in that cycle window for the selected files.")
        st.stop()

    df_plot = pd.DataFrame(rows)

    # Use the same color mapping as the rest of the app
    color_key = "File" if color_mode_global == "Per file" else "Group"
    color_map = color_map_file if color_key == "File" else color_map_fam

    # Y label
    if y_metric == "CE (%)":
        y_label = "Coulombic efficiency (%)"
    else:
        # heuristic: if any selected file has spec capacity columns, label as mAh/g
        has_spec = any(
            c in union_cols for c in [
                "Spec. Cap.(mAh/g)",
                "DChg. Spec. Cap.(mAh/g)",
                "Chg. Spec. Cap.(mAh/g)",
            ]
        )
        y_label = f"{y_metric} (mAh/g)" if has_spec else f"{y_metric} (mAh)"

    if x_mode == "Group":
        fig_box = px.box(
            df_plot,
            x="Group",
            y="Value",
            color=color_key,
            points="all",
            color_discrete_map=color_map,
            labels={"Group": "Group", "Value": y_label},
        )
        fig_box.update_layout(title=f"{y_metric} distribution by group (cycles {start}–{end})")
    else:
        # per-cycle boxplots, grouped by group color
        fig_box = px.box(
            df_plot,
            x="Cycle",
            y="Value",
            color=color_key,
            points="all",
            color_discrete_map=color_map,
            labels={"Cycle": "Cycle", "Value": y_label, "Group": "Group"},
        )
        fig_box.update_xaxes(type="category", tickangle=30)
        fig_box.update_layout(boxmode="group", title=f"{y_metric} — cycles {start}–{end}")
    fig_box.update_layout(legend_title_text="")
    fig_box.update_traces(
    selector=dict(type="box"),
    boxpoints="all",   # (already implied by points="all", but harmless)
    jitter=0.15,       # <-- reduce (try 0.05–0.25)
    pointpos=0,        # center the points over the box
    marker=dict(size=6, opacity=0.7),
)    

    apply_preli_style(fig_box, base=BASE_THEME, show_grid=show_grid_global)
    st.plotly_chart(fig_box, width="stretch", config=CAMERA_CFG)
    add_ppt_download(fig_box, filename_base=f"capfade_box_{int(start)}_{int(end)}")
    st.stop()


# Fallback
st.success("Loaded. Use the tabs above to explore your NDAX data.")

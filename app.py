import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import warnings
import io
import plotly.io as pio
import plotly.graph_objects as go
import math

# ----------------------------
# NDAX imports
# ----------------------------
try:
    import NewareNDA as nda
except Exception:
    nda = None

try:
    # your helper in the same folder as app.py
    from ndax_min_builder import build_ndax_df
except Exception:
    build_ndax_df = None

# ----------------------------
# App config
# ----------------------------

# Export via the modebar camera button
CAMERA_CFG = {
    "toImageButtonOptions": {
        "format": "png",      # or "svg"
        "filename": "plot",
        "scale": 4            # 2–4 is great for PPT
    }
}

st.set_page_config(
    page_title="Battery Cell Data — Visualizer",
    page_icon="🔋",
    layout="wide",
)

HERE = Path(__file__).parent
LOGO_PATH = HERE / "logo.png"   # exact filename
# Center the banner
c1, c2, c3 = st.columns([1, 1, 1])
with c2:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH))
    else:
        st.caption(f"Logo missing: {LOGO_PATH.name}")

st.title("🔋BATTERY CELL DATA —  VISUALIZER 📈")
st.caption("     Built by the Preli team    ")

# =============================
# Helpers
# =============================
NV_COLORDICT = {
    # Core
    "white": "#ffffff",
    "black": "#000000",

    # Greens (light → dark)
    "nv_green1": "#9ef3c0",
    "nv_green2": "#81f1c4",   
    "nv_green3": "#09f392",
    "nv_green4": "#03a567",   
    "nv_green5": "#028133",
    "nv_green6": "#009632",   
    "nv_green7": "#1e644b",   
    "nv_green8": "#174335",   
    "nv_green9": "#0e2922",

    # Blues/Teals (light → dark)
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

    # Greys (light → dark)
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

    # Accents
    "nv_power_green": "#44a27a",  # existing
    "nv_silver": "#8A8D8F",       # existing
    "nv_red": "#d84437",
    "nv_orange": "#f78618",
    "nv_amber": "#f7a600",
}

# ----------------------------
# Palettes (8–10 colors each)
# ----------------------------
PALETTES = {
    # Long, varied palette for multi-series plots (20+)
    "Preli long": [
        NV_COLORDICT["nv_power_green"], NV_COLORDICT["nv_blue5"], NV_COLORDICT["nv_blue1"],
        "#e7298a", "#0383a3", "#013824", "#c7e9b4", "#7fcdbb",
        "#d7301f", "#990000", NV_COLORDICT["nv_gray4"], NV_COLORDICT["nv_gray3"],
        NV_COLORDICT["nv_blue8"], "#6C88AD", "#5C9BBA",
        NV_COLORDICT["nv_green5"], NV_COLORDICT["nv_blue9"], NV_COLORDICT["nv_gray8"],
        NV_COLORDICT["nv_amber"], NV_COLORDICT["nv_orange"]
    ],

    # 10 blues/teals (light → dark)
    "Preli blues": [
        NV_COLORDICT["nv_blue0"], NV_COLORDICT["nv_blue1"], NV_COLORDICT["nv_blue2"],
        NV_COLORDICT["nv_blue3"], NV_COLORDICT["nv_blue4"], NV_COLORDICT["nv_blue5"],
        NV_COLORDICT["nv_blue6"], NV_COLORDICT["nv_blue7"], NV_COLORDICT["nv_blue8"],
        NV_COLORDICT["nv_blue9"],
    ],

    # 9 greens (light → dark)
    "Preli greens": [
        NV_COLORDICT["nv_green1"], NV_COLORDICT["nv_green2"], NV_COLORDICT["nv_green3"],
        NV_COLORDICT["nv_green4"], NV_COLORDICT["nv_green5"], NV_COLORDICT["nv_green6"],
        NV_COLORDICT["nv_green7"], NV_COLORDICT["nv_green8"], NV_COLORDICT["nv_green9"],
    ],

    # 10 greys (light → dark)
    "Preli greys": [
        NV_COLORDICT["nv_gray1"], NV_COLORDICT["nv_gray2"], NV_COLORDICT["nv_gray6"],
        NV_COLORDICT["nv_gray3"], NV_COLORDICT["nv_gray7"], NV_COLORDICT["nv_gray4"],
        NV_COLORDICT["nv_gray8"], NV_COLORDICT["nv_gray5"], NV_COLORDICT["nv_gray9"],
        NV_COLORDICT["nv_gray10"],
    ],

    # Warm accents
     "Preli Warm": [
        # yellows (light → deep)
        "#FFE8A3", "#FFD166", "#F7B733",
        # oranges (soft → vivid)
        "#F49E4C", "#F07C28", "#F25C05",
        # reds (warm → deep)
        "#E63B2E", "#CC2F27", "#A72222", "#7F1D1D"
    ]
}

# ----------------------------
# Figure styling + PPT export helpers
# ----------------------------
def style_for_ppt(fig):
    """
    Apply a consistent, PPT-friendly style:
    - Larger font
    - Horizontal legend with border
    - Matplotlib-like axis borders & ticks
    """
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
    # Axis borders + ticks (we *don’t* touch showgrid here)
    fig.update_xaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickwidth=1,
        ticklen=6,
        title_font=dict(family="Arial", size=14, color="black"),
        tickfont=dict(family="Arial", size=14, color="black"),
        
    )
    fig.update_yaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickwidth=1,
        ticklen=6,
        title_font=dict(family="Arial", size=14, color="black"),
        tickfont=dict(family="Arial", size=14, color="black"),
    )


def add_ppt_download(fig, filename_base: str):
    """
    High-res PNG download for PowerPoint.
    Requires `kaleido` in your env: pip install -U kaleido
    """
    import io
    buf = io.BytesIO()

    try:
        fig.write_image(
            buf,
            format="png",
            width=1600,   # pixels
            height=900,
            scale=2,      # supersampling → crisp text/lines
        )
    except Exception:
        st.info(
            "Static image export not available. "
            "Install `kaleido` (pip install -U kaleido) "
            "to enable high-quality PNG downloads."
        )
        return

    buf.seek(0)
    st.download_button(
        label="⬇️ Download PNG for PPT",
        data=buf,
        file_name=f"{filename_base}.png",
        mime="image/png",
    )

def pretty_src(src: str) -> str:
    """Shorten file name for legend labels."""
    return Path(src).stem  # "preli_2.ndax" -> "preli_2"

# ----------------------------
# Helpers
# ----------------------------
def normalize_neware_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Map common NDAX/Neware variants to canonical headers used downstream."""
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

    # time / timestamp variants
    ensure("Total Time", "totaltime", "total time")
    ensure("Time", "time(s)", "time")  # often numeric seconds (resets per step)
    # voltage/current
    ensure("Voltage(V)", "voltage (v)", "voltage")
    ensure("Current(mA)", "current (ma)", "current")
    # capacity (total + specific + per-step)
    ensure("Spec. Cap.(mAh/g)", "specific capacity (mAh/g)", "specific capacity (mah/g)", "spec cap (mah/g)")
    ensure("Capacity(mAh)", "capacity (mah)", "capacity")
    ensure("Chg. Spec. Cap.(mAh/g)", "chg. specific capacity (mAh/g)", "chg. spec. cap.(mAh/g)")
    ensure("DChg. Spec. Cap.(mAh/g)", "dchg. specific capacity (mAh/g)", "dchg. spec. cap.(mAh/g)")
    ensure("Chg. Cap.(mAh)", "chg. capacity (mah)")
    ensure("DChg. Cap.(mAh)", "dchg. capacity (mah)")
    # cycle & step
    ensure("Cycle Index", "cycle", "cycle number", "cycle_index")
    ensure("Step Type", "status", "state", "mode", "step")
    return d

warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
    category=FutureWarning,
)



def _concat_nonempty(frames):
    """Concat after dropping empty or all-NaN DataFrames."""
    if not frames:
        return pd.DataFrame()
    keep = []
    for f in frames:
        if f is None:
            continue
        if isinstance(f, pd.DataFrame) and (not f.empty):
            # drop DataFrames that are all-NaN (no real values anywhere)
            if not f.isna().all().all():
                keep.append(f)
    if not keep:
        # return an empty frame with union of columns to keep downstream code happy
        cols = pd.Index([])
        for f in frames:
            if isinstance(f, pd.DataFrame):
                cols = cols.union(f.columns)
        return pd.DataFrame(columns=cols)
    return pd.concat(keep, ignore_index=True, copy=False)

def infer_rest_step(
    df: pd.DataFrame,
    step_col: str = "Step Type",
    current_col: str = "Current(mA)",
    abs_threshold: float = 0.5,
    win: int = 5,
) -> pd.DataFrame:
    """If Rest isn't labeled, infer it where |Current(mA)| ~ 0 using rolling median."""
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

# ---- Global time builder (prefers absolute timestamp; else stitches step-local Time) ----
TIMESTAMP_CANDIDATES = [
    "Timestamp", "TimeStamp", "Record Time", "RecordTime", "Date Time", "DateTime",
    "datetime", "Measured Time", "MeasuredTime", "Test Time", "TestTime", "Start Time", "StartTime"
]

def pick_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    for c in TIMESTAMP_CANDIDATES:
        if c in df.columns:
            return c
    lower = {col.lower(): col for col in df.columns}
    for c in TIMESTAMP_CANDIDATES:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def build_global_time_seconds(
    df: pd.DataFrame,
    time_col: Optional[str],              # "Time" (resets) or "Total Time"
    cycle_col: str = "Cycle Index",
    step_col: str  = "Step Type",
) -> pd.Series:
    """
    Return a monotonic, t0-aligned time vector in seconds.
    Priority:
      1) Absolute timestamp column (parsed to datetime)
      2) Total Time (HH:MM:SS-like)
      3) Stitch step-local 'Time' that resets (accumulate per (cycle, step) group)
    """
    d = df

    # 1) Absolute timestamp
    ts_col = pick_timestamp_column(d)
    if ts_col:
        t = pd.to_datetime(d[ts_col], errors="coerce")
        return (t - t.iloc[0]).dt.total_seconds()

    # 2) Total Time
    if time_col and time_col in d.columns and time_col.lower() == "total time":
        td = pd.to_timedelta(d[time_col].astype(str), errors="coerce")
        return (td - td.iloc[0]).dt.total_seconds()

    # 3) Stitch step-local 'Time'
    raw = d[time_col] if (time_col and time_col in d.columns) else None
    if raw is None:
        return pd.Series(np.zeros(len(d)), index=d.index, dtype="float64")

    # choose grouping key
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

def pick_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first column that looks like an absolute timestamp."""
    candidates = [
        "Timestamp", "DateTime", "Datetime", "DATE TIME", "Start Time",
        "Record Time", "RecordTime", "System Time", "Local Time"
    ]
    cols_lc = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols_lc:
            return cols_lc[name.lower()]
    # heuristic: any column with many parseable datetimes
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce")
        if s.notna().mean() > 0.8:
            return c
    return None

def insert_line_breaks_vq(df: pd.DataFrame, cap_col: str, v_col: str) -> pd.DataFrame:
    """Insert NaNs after capacity resets to force V–Q breaks."""
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

def insert_line_breaks_vt(df: pd.DataFrame, t_col: str, v_col: str) -> pd.DataFrame:
    """Insert NaNs at cycle changes and at Rest transitions."""
    if df.empty or t_col not in df.columns or v_col not in df.columns:
        return df
    d = df.reset_index(drop=True).copy()
    breaks = set()
    n = len(d)
    if "Cycle Index" in d.columns:
        cyc = d["Cycle Index"].values
        for i in range(n - 1):
            if pd.notna(cyc[i]) and pd.notna(cyc[i + 1]) and cyc[i + 1] != cyc[i]:
                breaks.add(i)
    if "Step Type" in d.columns:
        stype = d["Step Type"].astype(str)
        for i in range(n - 1):
            a, b = stype.iloc[i], stype.iloc[i + 1]
            if (a == "Rest" and b != "Rest") or (a != "Rest" and b == "Rest"):
                breaks.add(i)
    if not breaks:
        return d
    pieces, last = [], 0
    for i in sorted(breaks):
        pieces.append(d.iloc[last:i + 1])
        nan_row = {c: (np.nan if c in [t_col, v_col] else d.iloc[i].get(c)) for c in d.columns}
        pieces.append(pd.DataFrame([nan_row]))
        last = i + 1
    pieces.append(d.iloc[last:])
    return _concat_nonempty(pieces)

def compute_ce(df: pd.DataFrame, cell_type: str = "cathode") -> pd.DataFrame:
    """Cycle-wise CE; prefers specific-cap columns, otherwise totals."""
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

def insert_line_breaks_generic(
    df: pd.DataFrame, x_col: str, y_col: str,
    *,
    seg_cycle: bool = False,
    seg_step: bool = False,
    seg_cap_reset: bool = False,
    seg_current_flip: bool = False,
    seg_x_reverse: bool = False,
    x_rev_eps: float = 0.0,
    cap_col_name: Optional[str] = None
) -> pd.DataFrame:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return df

    # validate capacity column request
    if seg_cap_reset:
        if not isinstance(cap_col_name, str) or cap_col_name not in df.columns:
            seg_cap_reset = False  # disable if unusable

    # ensure unique columns (avoid pandas concat alignment errors)
    d = df.loc[:, ~pd.Index(df.columns).duplicated()].reset_index(drop=True).copy()
    break_pos: List[int] = []

    # 1) Cycle changes
    if seg_cycle and "Cycle Index" in d.columns:
        cyc = d["Cycle Index"].values
        for i in range(len(d) - 1):
            if pd.notna(cyc[i]) and pd.notna(cyc[i+1]) and cyc[i+1] != cyc[i]:
                break_pos.append(i)

    # 2) Step-type changes
    if seg_step and "Step Type" in d.columns:
        stype = d["Step Type"].astype(str)
        for i in range(len(d) - 1):
            if stype.iloc[i+1] != stype.iloc[i]:
                break_pos.append(i)

    # 3) Capacity resets to 0
    if seg_cap_reset:
        try:
            cap_series = pd.to_numeric(d[cap_col_name], errors="coerce").fillna(0)
            idxs = cap_series[(cap_series.shift(-1) == 0) & (cap_series > 0)].index.tolist()
            break_pos.extend(idxs)
        except Exception:
            pass  # if anything odd happens, just skip this rule

    # 4) Current sign flips
    if seg_current_flip and "Current(mA)" in d.columns:
        cur = pd.to_numeric(d["Current(mA)"], errors="coerce")
        sgn = np.sign(cur)
        for i in range(len(d) - 1):
            if not np.isnan(sgn.iloc[i]) and not np.isnan(sgn.iloc[i+1]) and sgn.iloc[i+1] != sgn.iloc[i]:
                break_pos.append(i)

    # 5) X reversals (direction change)
    if seg_x_reverse:
        x_num = pd.to_numeric(d[x_col], errors="coerce")
        if x_num.isna().all():
            x_num = pd.to_timedelta(d[x_col].astype(str), errors="coerce").dt.total_seconds()
        dx = x_num.diff()
        for i in range(1, len(d)):
            if pd.notna(dx.iloc[i]) and abs(dx.iloc[i]) > x_rev_eps and dx.iloc[i] * dx.iloc[i-1] < 0:
                break_pos.append(i-1)

    if not break_pos:
        return d

    pieces = []
    last = 0
    for idx in sorted(set(break_pos)):
        pieces.append(d.iloc[last:idx+1])
        # force a gap for x/y only
        nan_row = {c: (np.nan if c in [x_col, y_col] else d.iloc[idx].get(c)) for c in d.columns}
        pieces.append(pd.DataFrame([nan_row]))
        last = idx + 1
    pieces.append(d.iloc[last:])
    return _concat_nonempty(pieces)

def compute_dcir_for_ndax(
    df: pd.DataFrame,
    cell_id: str,
    pulse_length_s: float,
    soc_mode: str = "design",           # "design" or "data"
    soc_levels=None,                    # e.g. [80, 50, 20, 5] for design mode
    pulses_per_soc: int = 2,
    pre_rest_window_s: float = 60.0,    # last 60 s of rest
    inst_window_s=(0.5, 1.5),           # instant = 0.5–1.5 s into pulse (like your scripts)
    end_window_s: float = 5.0,          # last 5 s of pulse
    pulse_tol_s: float = 2.0,           # duration tolerance
    current_threshold_A: float = 0.001, # reject tiny-current noise
) -> pd.DataFrame:
    """
    DCIR from NDAX-like DataFrame.

    df must have (after normalize_neware_headers / infer_rest_step):
      - 'Step_Index'
      - 'Step Type'         (Rest / Chg / DChg / CCCV_Chg / CC_DChg / CC_Chg, etc.)
      - 'Time'              (step-local seconds)
      - 'Voltage(V)' or 'Voltage'
      - 'Current(mA)'
      - optional: 'Charge_Capacity(mAh)', 'Discharge_Capacity(mAh)'

    Returns one row per DCIR pulse with:
      Cell_ID, Pulse_Direction, SoC_label, pulse_duration_s,
      DCIR_<pulse_length>s_Ohm, DCIR_inst_Ohm
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    cols = d.columns

    # --- column mapping (matches your NDAX frames) ---
    if "Step_Index" not in cols:
        raise ValueError("Expected 'Step_Index' column but did not find it.")

    step_index_col = "Step_Index"
    status_col = "Step Type" if "Step Type" in cols else "Status"
    time_col = "Time"
    volt_col = "Voltage(V)" if "Voltage(V)" in cols else "Voltage"
    cur_col = "Current(mA)"

    # numeric helpers
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d["Current_A"] = pd.to_numeric(d[cur_col], errors="coerce") / 1000.0

    # ---- summarise each Step_Index ----
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

    # ---- pulse detection: duration ~ pulse_length, not Rest, non-zero current ----
    is_pulse_dur = step_summary["duration_s"].between(
        pulse_length_s - pulse_tol_s,
        pulse_length_s + pulse_tol_s,
    )
    enough_I = step_summary["I_mean"].abs() >= current_threshold_A

    pulse_mask = (~is_rest) & is_pulse_dur & enough_I
    pulse_steps = step_summary.index[pulse_mask].tolist()

    dcir_end_col = f"DCIR_{int(round(pulse_length_s))}s_Ohm"

    if not pulse_steps:
        return pd.DataFrame(
            columns=[
                "Cell_ID",
                "Pulse_Direction",
                "SoC_label",
                "pulse_duration_s",
                dcir_end_col,
                "DCIR_inst_Ohm",
            ]
        )

    # ---- SOC labelling helpers ----

    # Design-based SOC: assign labels in pulse order
    soc_labels_by_step = {}
    if soc_mode == "design" and soc_levels:
        expanded = []
        for lvl in soc_levels:
            expanded.extend([lvl] * pulses_per_soc)
        for i, step in enumerate(sorted(pulse_steps)):
            soc_labels_by_step[step] = expanded[i] if i < len(expanded) else expanded[-1]

    # Data-based SOC from capacity
    q_ref = None
    if soc_mode == "data":
        if "Discharge_Capacity(mAh)" in cols:
            q_ref = pd.to_numeric(d["Discharge_Capacity(mAh)"], errors="coerce").max()
        if (q_ref is None or not math.isfinite(q_ref) or q_ref <= 0) and \
           "Charge_Capacity(mAh)" in cols:
            q_ref = pd.to_numeric(d["Charge_Capacity(mAh)"], errors="coerce").max()
        if q_ref is not None and (not math.isfinite(q_ref) or q_ref <= 0):
            q_ref = None

    rest_steps = step_summary.index[is_rest]
    inst_start, inst_end = inst_window_s

    records = []

    for step in sorted(pulse_steps):
        step_info = step_summary.loc[step]

        # ---------- find preceding Rest step ----------
        prior_rests = rest_steps[rest_steps < step]
        if len(prior_rests) == 0:
            continue
        rest_step = prior_rests[-1]

        df_rest = d[d[step_index_col] == rest_step]
        df_pulse = d[d[step_index_col] == step]
        if df_rest.empty or df_pulse.empty:
            continue

        # ---------- pre-pulse window: last pre_rest_window_s of Rest ----------
        rt_max = df_rest[time_col].max()
        rt_min = df_rest[time_col].min()
        rest_start = max(rt_min, rt_max - pre_rest_window_s)
        rest_win = df_rest[df_rest[time_col] >= rest_start]
        if rest_win.empty:
            rest_win = df_rest

        v_pre = pd.to_numeric(rest_win[volt_col], errors="coerce").mean()
        i_pre_mA = pd.to_numeric(rest_win[cur_col], errors="coerce").mean()

        # ---------- instantaneous window (early in the pulse) ----------
        inst = df_pulse[
            (df_pulse[time_col] >= inst_start) & (df_pulse[time_col] <= inst_end)
        ]
        if inst.empty:
            # fallback: first few points if time grid is weird
            inst = df_pulse.head(min(3, len(df_pulse)))

        v_inst = pd.to_numeric(inst[volt_col], errors="coerce").mean()
        i_inst_mA = pd.to_numeric(inst[cur_col], errors="coerce").mean()

        # ---------- end-of-pulse window (last end_window_s) ----------
        pt_max = df_pulse[time_col].max()
        pt_min = df_pulse[time_col].min()
        end_start = max(pt_min, pt_max - end_window_s)
        end_win = df_pulse[df_pulse[time_col] >= end_start]
        if end_win.empty:
            end_win = df_pulse.iloc[[-1]]

        v_end = pd.to_numeric(end_win[volt_col], errors="coerce").mean()
        i_end_mA = pd.to_numeric(end_win[cur_col], errors="coerce").mean()

        # ---------- ΔV / ΔI (match your SAFT/POCO scripts) ----------
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

        # Pulse direction from sign of current
        pulse_i_mean = df_pulse["Current_A"].mean()
        if pulse_i_mean > 0:
            direction = "charge"
        elif pulse_i_mean < 0:
            direction = "discharge"
        else:
            direction = "unknown"

        # ---------- SOC label ----------
        soc_label = None

        if soc_mode == "design" and soc_levels:
            soc_label = soc_labels_by_step.get(step, None)

        elif soc_mode == "data" and q_ref and q_ref > 0:
            first_row = df_pulse.iloc[0]
            if direction == "discharge" and "Discharge_Capacity(mAh)" in cols:
                q = float(first_row["Discharge_Capacity(mAh)"])
                if math.isfinite(q):
                    soc_label = 100.0 * (1.0 - q / q_ref)
            elif direction == "charge" and "Charge_Capacity(mAh)" in cols:
                q = float(first_row["Charge_Capacity(mAh)"])
                if math.isfinite(q):
                    soc_label = 100.0 * (q / q_ref)
            if soc_label is not None:
                soc_label = round(soc_label, 1)

        records.append(
            {
                "Cell_ID": cell_id,
                "Pulse_Direction": direction,
                "SoC_label": soc_label,
                "pulse_duration_s": round(float(step_info["duration_s"]), 3),
                dcir_end_col: dcir_end,
                "DCIR_inst_Ohm": dcir_inst,
            }
        )

    return pd.DataFrame.from_records(records)

# ----------------------------
# NDAX-only loader
# ----------------------------
@st.cache_data(show_spinner=False)
def load_ndax(file) -> pd.DataFrame:
    if nda is None:
        st.error("NewareNDA is not installed. Run: python -m pip install NewareNDA")
        raise RuntimeError("NewareNDA missing")
    name = getattr(file, "name", "uploaded")
    if Path(name).suffix.lower() != ".ndax":
        st.error("This app supports only .ndax files.")
        raise RuntimeError("Unsupported file type")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ndax") as tmp:
        tmp.write(file.getbuffer())
        tmp_path = tmp.name
    try:
        if build_ndax_df is not None:
            df = build_ndax_df(tmp_path, cycle_mode="auto")
        else:
            df = nda.read(tmp_path)
    except Exception as e:
        st.error(f"Failed to read NDAX: {e}")
        raise
    # Normalize & infer Rest for better segmentation later
    df = normalize_neware_headers(df)
    df = infer_rest_step(df)
    return df

# ----------------------------
# Sidebar upload (NDAX only)
# ----------------------------
st.sidebar.header("Upload NDAX files only")
uploaded_files = st.sidebar.file_uploader(
    "Drop Neware .ndax files (multiple allowed)",
    type=["ndax"], accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload one or more NDAX files to begin.")
    st.stop()

with st.sidebar.expander("CE options", expanded=True):
    cell_type_sel = st.radio(
        "Cell type (for CE direction)",
        ["anode", "cathode", "full"],
        index=1,  # default to cathode
        horizontal=True
    )

palette_name = st.sidebar.selectbox("Palette", list(PALETTES.keys()), index=0)
palette = PALETTES[palette_name]

# Parse files
all_frames: List[pd.DataFrame] = []
source_names: List[str] = []
with st.spinner("Parsing NDAX files…"):
    for f in uploaded_files:
        df = load_ndax(f)
        df["__file"] = f.name
        all_frames.append(df)
        source_names.append(f.name)

frames = [df for df in all_frames if df is not None and not df.empty]
if not frames:
    st.error("All uploaded NDAX files were empty.")
    st.stop()
raw = _concat_nonempty(frames)
G = detect_columns(list(raw.columns))

# All distinct file names present
files_all = sorted(raw["__file"].astype(str).unique().tolist())

st.sidebar.header("Files to plot")
# Initialize (or refresh) the checkbox state if file set changed
if "file_checks" not in st.session_state or set(st.session_state.file_checks.keys()) != set(files_all):
    st.session_state.file_checks = {f: True for f in files_all}

c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("Select all"):
        st.session_state.file_checks = {f: True for f in files_all}
with c2:
    if st.button("Select none"):
        st.session_state.file_checks = {f: False for f in files_all}

# Render one checkbox per file
for f in files_all:
    st.session_state.file_checks[f] = st.sidebar.checkbox(
        f, value=st.session_state.file_checks.get(f, True)
    )

# The active selection
selected_sources = [f for f, on in st.session_state.file_checks.items() if on]

if not selected_sources:
    st.info("Select at least one file in the sidebar to plot.")
    st.stop()

# Filter once and use `data` everywhere below instead of `raw`
data = raw[raw["__file"].isin(selected_sources)].copy()


# Build per-file palette
selected_sources = sorted(list({f.name for f in uploaded_files}))
color_map = {src: palette[i % len(palette)] for i, src in enumerate(selected_sources)}

# Preview
with st.expander("Preview parsed data"):
    st.dataframe(raw.drop(columns="__file").head(900))

with st.expander("active mass (read-only)"):
    lines = []
    for src, df_src in zip(source_names, all_frames):
        am = getattr(df_src, "attrs", {}).get("active_mass_g", None)
        if isinstance(am, (int, float)) and am > 0:
            lines.append(f"• {src}: active_mass_g = {am:.6f} g")
    st.markdown("\n".join(lines) if lines else "_No active_mass_g found in uploaded files._")

with st.sidebar.expander("Formatting", expanded=True):
    show_markers = st.checkbox("Show markers", False)
    marker_size = st.number_input("Marker size", 1, 20, 4)
    line_width = st.number_input("Line width", 1.0, 10.0, 2.5, 0.5)
    show_grid = st.checkbox("Show grid", True)

def family_from_filename(name: str) -> str:
    stem = Path(str(name)).stem.lower()
    for sep in ["_", "-", " "]:
        if sep in stem:
            stem = stem.split(sep)[0]; break
    return re.sub(r"\d+$", "", stem) or stem

# annotate families once
if "__family" not in raw.columns:
    raw["__family"] = raw["__file"].apply(family_from_filename)

families_in_data = sorted(raw["__family"].unique().tolist())
files_in_data    = sorted(raw["__file"].unique().tolist())

# ---- global color mode (applies to all tabs) ----
with st.sidebar.expander("Color mode", expanded=True):
    color_mode_global = st.radio(
        "Color by",
        ["Per file", "Filename family"],
        index=0,
        horizontal=True
    )

# palette mapping for both modes
color_map_file = {src: palette[i % len(palette)] for i, src in enumerate(files_in_data)}
color_map_fam  = {fam: palette[i % len(palette)] for i, fam in enumerate(families_in_data)}

def color_for_src(src: str) -> str:
    if color_mode_global == "Per file":
        return color_map_file.get(src, palette[0])
    return color_map_fam.get(family_from_filename(src), palette[0])

# ----------------------------
# Tabs
# ----------------------------
xy_tab, vt_tab, vq_tab, cap_tab, ce_tab, dcir_tab, = st.tabs([
    "XY Builder", "Voltage–Time", "Voltage–Capacity", "Capacity vs Cycle", "Capacity & CE","DCIR",])
#--------------------------------------
# ---------- XY Builder ---------------
# -------------------------------------
with xy_tab:
    st.subheader("Free-form XY plot builder")

    all_cols = [c for c in data.columns if c != "__file"]
    numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(data[c])]

    # sensible defaults
    if "xy_x" not in st.session_state:
        st.session_state.xy_x = G.get("time") if G.get("time") in all_cols else (all_cols[0] if all_cols else None)
    if "xy_y" not in st.session_state or not st.session_state.xy_y:
        defaults = [c for c in [G.get("voltage"), G.get("capacity")] if c in numeric_cols]
        st.session_state.xy_y = defaults[:1] if defaults else (numeric_cols[:1] if numeric_cols else [])

    def family_from_filename(name: str) -> str:
        stem = Path(str(name)).stem.lower()
        for sep in ["_", "-", " "]:
            if sep in stem:
                stem = stem.split(sep)[0]; break
        return re.sub(r"\d+$", "", stem) or stem

    # ---- Compact, aligned XY controls ----
    row1 = st.columns([1.1, 1.2, 0.9])   # X, Y, tools
    with row1[0]:
        x_col = st.selectbox(
        "X axis",
        all_cols,
        index=(all_cols.index(st.session_state.xy_x) if st.session_state.xy_x in all_cols else 0),
        key="xy_x_select",
        )
    # Auto-align if X is the detected time column (hide the toggle)
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
        rolling = st.number_input("Rolling mean window (pts)", 1, 9999, 1, 1)
        use_global_colors_xy = st.checkbox(
        "Use global colors here",
        value=False,
        help="If on, XY uses the sidebar Color by mode; if off, random colors.",
        )

    # Y limits (side-by-side)
    row2 = st.columns([1, 1, 0.6])
    with row2[0]:
                y_min = st.text_input("Y min (blank=auto)", "")
    with row2[1]:
                y_max = st.text_input("Y max (blank=auto)", "")

    # X limits (side-by-side)
    row3 = st.columns([1, 1, 0.6])
    with row3[0]:
            x_min = st.text_input("X min (blank=auto)", "")
    with row3[1]:
            x_max = st.text_input("X max (blank=auto)", "")

    # persist selections
    st.session_state.xy_x = x_col
    st.session_state.xy_y = y_cols

    if not y_cols:
        st.warning("Pick at least one Y column to plot.")
        st.stop()

    plot_df = data.dropna(subset=[x_col] + y_cols).copy()
        # build time vector if aligning
    if align_t0 and is_time_x:
        plot_df["_x"] = build_global_time_seconds(plot_df, time_col=G["time"], cycle_col="Cycle Index", step_col="Step Type")
        x_used = "_x"
    else:
        x_used = x_col

    # color mapping
    # color mapping — rely on global helper + palette
    # (raw["__family"] was already created globally)
    files_in_view = plot_df["__file"].astype(str).unique().tolist()

    # optional smoothing
    if rolling > 1:
        plot_df = plot_df.sort_values(["__file", x_used])
        for y in y_cols:
            if y in plot_df.columns:
                plot_df[y] = plot_df.groupby("__file")[y].transform(lambda s: s.rolling(rolling, min_periods=1).mean())

    fig = go.Figure()
    added = False

    cap_like = ["Spec. Cap.(mAh/g)", "Capacity(mAh)", "Chg. Spec. Cap.(mAh/g)", "DChg. Spec. Cap.(mAh/g)"]
    is_time_plot = (x_used == "_x") or (x_col == G.get("time"))
    cyc_col = G.get("cycle")

    for y in y_cols:
        for src in files_in_view:
            s = plot_df[plot_df["__file"] == src].copy()
            if s.empty or x_used not in s.columns or y not in s.columns:
                continue

            # SPECIAL: Cycle Index on X with capacity on Y → aggregate max per cycle
            if cyc_col and cyc_col in s.columns and x_used == cyc_col and y in cap_like:
                s["_cyc"] = pd.to_numeric(s[cyc_col], errors="coerce")
                grp = (s.dropna(subset=["_cyc", y])
                         .groupby("_cyc", as_index=False)[y]
                         .max()
                         .sort_values("_cyc"))
                if grp.empty:
                    continue
                s2 = grp.rename(columns={"_cyc": x_used})

            # TIME on X in XY builder → keep continuous
            elif is_time_plot:
                s2 = s

            # Otherwise: apply generic line breaks (cycle/step/capacity-reset)
            else:
                cap_name = next((c for c in cap_like if c in s.columns), None)
                order = [x_used, y, "__file", "Cycle Index", "Step Type", "Current(mA)", cap_name]
                cols_needed = []
                for c in order:
                    if isinstance(c, str) and c in s.columns and c not in cols_needed:
                        cols_needed.append(c)
                s2 = insert_line_breaks_generic(
                    s[cols_needed], x_used, y,
                    seg_cycle=("Cycle Index" in s.columns),
                    seg_step=("Step Type" in s.columns),
                    seg_cap_reset=(isinstance(cap_name, str) and cap_name in s.columns and cap_name != x_used),
                    cap_col_name=cap_name,
                    seg_current_flip=False,
                    seg_x_reverse=False
                )

            if s2.empty:
                continue

            if use_global_colors_xy:
                c = color_for_src(src)  # global helper from the sidebar section
                fig.add_trace(go.Scatter(
                x=s2[x_used], y=s2[y],
                mode="lines",
                name=pretty_src(src),
                line=dict(color=c, width=line_width),
                marker=dict(size=marker_size)
                ))
            else:
                fig.add_trace(go.Scatter(
                x=s2[x_used], y=s2[y],
                mode="lines",
                name=pretty_src(src),
                line=dict(width=line_width),   # no color → Plotly default cycle
                marker=dict(size=marker_size)
                ))      
            added = True

    if not added:
        st.warning("No data drawn — check your column choices or filters.")
    else:
        # axes/legend styling
        try:
            if x_min != "": fig.update_xaxes(range=[float(x_min), float(x_max) if x_max != "" else None])
        except Exception: pass
        try:
            if y_min != "": fig.update_yaxes(range=[float(y_min), float(y_max) if y_max != "" else None])
        except Exception: pass
        fig.update_layout(template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        if show_grid:
            fig.update_xaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)
            fig.update_yaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)
        st.plotly_chart(fig, use_container_width=True)


with vt_tab:
    st.subheader("Voltage–Time")
    tcol, vcol = G["time"], G["voltage"]
    if not tcol or not vcol or tcol not in data.columns or vcol not in data.columns:
        st.warning("Couldn’t detect time/voltage. Check NDAX headers.")
    else:
        dfv = data.dropna(subset=[tcol, vcol]).copy()

        unit = st.selectbox("Time units", ["seconds", "minutes", "hours"], 0, key="vt_time_unit")
        DIV = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}
        ABBR = {"seconds": "s", "minutes": "min", "hours": "h"}

        fig_vt = go.Figure()
        for src in selected_sources:
            s = dfv[dfv["__file"] == src].copy()
            if s.empty:
                continue

            # Build global, monotonic seconds using YOUR helper
            s["_t"] = build_global_time_seconds(
            s, time_col=tcol, cycle_col="Cycle Index", step_col="Step Type"
            )

            s = s.dropna(subset=["_t", vcol]).sort_values("_t")
            if s.empty:
                continue

            fig_vt.add_trace(go.Scatter(
                x=s["_t"] / DIV[unit],
                y=s[vcol],
                name=pretty_src(src),
                mode=("lines+markers" if show_markers else "lines"),
                line=dict(color=color_for_src(src), width=line_width),
                marker=dict(size=marker_size)
            ))

        fig_vt.update_layout(
            template="plotly_white",
            xaxis_title=f"Time ({ABBR[unit]})",
            yaxis_title=vcol,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
       

        if show_grid:
            fig_vt.update_xaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)
            fig_vt.update_yaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)

        style_for_ppt(fig_vt)
        st.plotly_chart(fig_vt, use_container_width=False, config=CAMERA_CFG)
        add_ppt_download(fig_vt, filename_base="voltage_time")

# ---------- Voltage–Capacity ----------
with vq_tab:
    st.subheader("Voltage–Capacity")
    ccol, vcol = G["capacity"], G["voltage"]
    if not ccol or not vcol or ccol not in data.columns or vcol not in data.columns:
        st.warning("Couldn’t detect capacity/voltage. Check NDAX headers.")
    else:
        base = data.dropna(subset=[ccol, vcol]).copy()
        base = infer_rest_step(base)
        cyc = G["cycle"] if G["cycle"] in base.columns else None
        if cyc:
            cycles_available = sorted(pd.unique(base[cyc].dropna()))
            sel_cycle = st.selectbox("Cycle", ["All"] + cycles_available, 0, key="vq_single_cycle")
            if sel_cycle != "All":
                base = base[base[cyc] == sel_cycle]
            if cycles_available:
                cmin, cmax = int(min(cycles_available)), int(max(cycles_available))
                rng = st.slider("Cycle range (optional)", cmin, cmax, (cmin, cmax), 1, key="vq_range")
                if rng != (cmin, cmax) and sel_cycle == "All":
                    lo, hi = rng
                    base = base[base[cyc].between(lo, hi)]
                    st.caption(f"Showing cycles {lo}–{hi}")

        fig_vq = go.Figure()
        for src in selected_sources:
            s = base[base["__file"] == src]
            if s.empty:
                continue
    # define the columns we actually need
            cols = [ccol, vcol, "__file"] + [c for c in ["Cycle Index", "Step Type"] if c in s.columns]
            plot_df = insert_line_breaks_vq(s[cols], cap_col=ccol, v_col=vcol)
            fig_vq.add_trace(go.Scatter(
                x=plot_df[ccol], y=plot_df[vcol], name=pretty_src(src),
                mode=("lines+markers" if show_markers else "lines"),
                line=dict(color=color_for_src(src), width=line_width),
                marker=dict(size=marker_size)
            ))
        x_label = f"{ccol}" if "mAh/g" in ccol else ccol
        fig_vq.update_layout(template="plotly_white", xaxis_title=x_label, yaxis_title=vcol,
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        if show_grid:
            fig_vq.update_xaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)
            fig_vq.update_yaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)
        

        style_for_ppt(fig_vq)
        st.plotly_chart(fig_vq, use_container_width=False, config=CAMERA_CFG)
        add_ppt_download(fig_vq, filename_base="voltage_capacity")

# ---------- Capacity vs Cycle ----------
with cap_tab:
    st.subheader("Capacity vs Cycle")
    cyc = G["cycle"]
    cap_col = None
    for cand in ["Spec. Cap.(mAh/g)", "DChg. Spec. Cap.(mAh/g)", "Chg. Spec. Cap.(mAh/g)", "Capacity(mAh)"]:
        if cand in data.columns:
            cap_col = cand
            break
    if not cyc or cyc not in data.columns or cap_col is None:
        st.info("Need Cycle Index and a capacity column.")
    else:
        fig_cap = go.Figure()
        for src in selected_sources:
            s = data[data["__file"] == src]
            cap_cycle = s.groupby(cyc)[cap_col].max().reset_index()
            if cap_cycle.empty: 
                continue
            c = color_for_src(src)
            fig_cap.add_trace(go.Scatter(
                x=cap_cycle[cyc], y=cap_cycle[cap_col], name=pretty_src(src),
                mode=("lines+markers" if show_markers else "lines"),
                line=dict(color=c, width=line_width),
                marker=dict(
                    size=marker_size, symbol="circle-open", color=c,
                    line=dict(color=c, width=max(1, int(line_width - 1)))
                )       
            ))
        y_label = "Specific capacity (mAh/g)" if "mAh/g" in cap_col else "Capacity (mAh)"
        fig_cap.update_layout(template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig_cap.update_xaxes(title_text="Cycle", showgrid=show_grid, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)
        fig_cap.update_yaxes(title_text=y_label, showgrid=show_grid, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)
        style_for_ppt(fig_cap)
        st.plotly_chart(fig_cap, use_container_width=False, config=CAMERA_CFG)
        add_ppt_download(fig_cap, filename_base="Capacity vs cycle")

# ---------- Capacity & CE ----------
with ce_tab:
    st.subheader("Capacity & Coulombic Efficiency vs Cycle")
    cyc = G["cycle"]
    cap_col = None
    for cand in ["Spec. Cap.(mAh/g)", "DChg. Spec. Cap.(mAh/g)", "Chg. Spec. Cap.(mAh/g)", "Capacity(mAh)"]:
        if cand in data.columns:
            cap_col = cand
            break
    if not cyc or cyc not in data.columns or cap_col is None:
        st.info("Need Cycle Index and capacity column for CE plot.")
    else:
        fig_ce = make_subplots(specs=[[{"secondary_y": True}]])
        ce_cell_type = "cathode" if cell_type_sel == "full" else cell_type_sel
        for src in selected_sources:
            sub = data[data["__file"] == src]
            if sub.empty: 
                continue
            c = color_for_src(src)
            ce_df = compute_ce(sub, cell_type=ce_cell_type)
            cap_cycle = sub.groupby(cyc)[cap_col].max().reset_index()
            if not cap_cycle.empty:
                fig_ce.add_trace(go.Scatter(
                    x=cap_cycle[cyc], y=cap_cycle[cap_col], name=f"Cap — {src}",
                    mode=("lines+markers" if show_markers else "lines"),
                    line=dict(color=c, width=line_width),
                    marker=dict(size=marker_size, symbol="circle-open", color=c,
                                line=dict(color=c, width=max(1, int(line_width - 1))))
                ), secondary_y=False)
            if not ce_df.empty:
                fig_ce.add_trace(go.Scatter(
                    x=ce_df["cycle"], y=ce_df["ce"], name=f"CE — {src}",
                    mode=("lines+markers" if show_markers else "lines"),
                    line=dict(color=c, dash="dash", width=line_width),
                    marker=dict(size=marker_size, symbol="diamond", color=c)
                ), secondary_y=True)
        y_left = "Specific capacity (mAh/g)" if "mAh/g" in (cap_col or "") else "Capacity (mAh)"

            #UI controls
        show_grid = st.checkbox("Show grid", value=True, key="ce_show_grid")
        grid_side = st.radio(
            "Y-grid on",
            ["left", "right", "both", "none"],
            index=0,  # default = left
            key="ce_grid_side"
        )
        
        fig_ce.update_yaxes(title_text=y_left, secondary_y=False)
        fig_ce.update_yaxes(title_text="CE (%)", range=[90, 105], secondary_y=True)
        fig_ce.update_xaxes(title_text="Cycle")
        fig_ce.update_layout(template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02))

        if show_grid and grid_side != "none":
            # X-axis grid ON
            fig_ce.update_xaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)

            # Left Y grid
            fig_ce.update_yaxes(
                showgrid=(grid_side in ["left", "both"]),
                gridcolor=NV_COLORDICT["nv_gray3"],
                gridwidth=0.5,
                secondary_y=False
            )
            # Right Y grid
            fig_ce.update_yaxes(
                showgrid=(grid_side in ["right", "both"]),
                gridcolor=NV_COLORDICT["nv_gray3"],
                gridwidth=0.5,
                secondary_y=True
            )
        else:
            # turn all grids off
            fig_ce.update_xaxes(showgrid=False)
            fig_ce.update_yaxes(showgrid=False, secondary_y=False)
            fig_ce.update_yaxes(showgrid=False, secondary_y=True)
        style_for_ppt(fig_ce)
        st.plotly_chart(fig_ce, use_container_width=False, config=CAMERA_CFG)
        add_ppt_download(fig_ce, filename_base="CE & Capacity")

# ---------- DCIR  ----------
#----------------------------
with dcir_tab:
    st.subheader("DCIR calculator")

    if "dcir_results" not in st.session_state:
        st.session_state["dcir_results"] = None
        st.session_state["dcir_summary"] = None

    files_here = sorted(data["__file"].astype(str).unique().tolist())
    if not files_here:
        st.info("No NDAX data available for DCIR.")
    else:
        left, right = st.columns([1.2, 1])

        with left:
            pulse_length_s = st.number_input(
                "DCIR pulse length [s]",
                min_value=1.0,
                value=18.0,
                step=1.0,
            )

            soc_mode_label = st.radio(
                "SOC labelling mode",
                ["By test design (e.g. 80/50/20/5%)", "Estimate from data (capacity-based)"],
                index=0,
            )

            if soc_mode_label.startswith("By test"):
                soc_mode = "design"
                soc_levels_text = st.text_input(
                    "SOC levels (%) in pulse order",
                    value="80, 50, 20, 5",
                    help="Example: 80, 50, 20, 5 with 2 pulses per SOC → 8 pulses total.",
                )
                pulses_per_soc = st.number_input(
                    "Pulses per SOC level",
                    min_value=1,
                    max_value=10,
                    value=2,
                    step=1,
                )
                soc_levels = [
                    float(x.strip())
                    for x in soc_levels_text.split(",")
                    if x.strip()
                ]
            else:
                soc_mode = "data"
                soc_levels = None
                pulses_per_soc = 2  # ignored in data mode

            run_btn = st.button("Compute DCIR")

        with right:
            st.markdown(
                """
**How this works :**

- Pre-rest OCV = mean _Voltage_ over last **60 s** of the Rest step.
- **Instantaneous DCIR** = ΔV/ΔI using an early time window (e.g. 0.5–1.5 s).
- **End-of-pulse DCIR** = ΔV/ΔI using last **5 s** of the pulse.
- Pulses are detected as non-Rest steps with duration ≈ your pulse length.
                """
            )

        if run_btn:
            all_results = []

            for src in files_here:
                df_src = data[data["__file"] == src].copy()
                if df_src.empty:
                    continue

                try:
                    res = compute_dcir_for_ndax(
                        df=df_src,
                        cell_id=src,
                        pulse_length_s=pulse_length_s,
                        soc_mode=soc_mode,
                        soc_levels=soc_levels,
                        pulses_per_soc=pulses_per_soc,
                    )
                except Exception as e:
                    st.warning(f"Skipping {src} due to error: {e}")
                    continue

                if res is not None and not res.empty:
                    all_results.append(res)

            # ----- UPDATE SESSION STATE, don't display directly here -----
            if all_results:
                dcir_results = pd.concat(all_results, ignore_index=True)

                dcir_end_col = f"DCIR_{int(round(pulse_length_s))}s_Ohm"

                wide_summary = None
                if (
                    "SoC_label" in dcir_results.columns
                    and dcir_end_col in dcir_results.columns
                    and "DCIR_inst_Ohm" in dcir_results.columns
                ):
                    tmp = dcir_results.copy()

                    # nice SoC strings like "80%"
                    def soc_fmt(x):
                        s = str(x)
                        try:
                            v = float(s.replace("%", ""))
                            return f"{v:g}%"
                        except Exception:
                            return s

                    tmp["SoC_fmt"] = tmp["SoC_label"].apply(soc_fmt)

                    metric_map = {
                        dcir_end_col: "18s",
                        "DCIR_inst_Ohm": "inst",
                    }

                    long = tmp.melt(
                        id_vars=["Cell_ID", "Pulse_Direction", "SoC_fmt"],
                        value_vars=list(metric_map.keys()),
                        var_name="metric",
                        value_name="DCIR",
                    )
                    long["metric_suffix"] = long["metric"].map(metric_map)

                    # column name like "80%_inst" or "80%_18s"
                    long["column_name"] = long["SoC_fmt"] + "_" + long["metric_suffix"]

                    wide = long.pivot_table(
                        index=["Cell_ID", "Pulse_Direction"],
                        columns="column_name",
                        values="DCIR",
                        aggfunc="mean",
                    ).reset_index()

                    value_cols = [
                        c for c in wide.columns
                        if c not in ["Cell_ID", "Pulse_Direction"]
                    ]

                    import math

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
                else:
                    wide_summary = None

                # store in session_state so it survives reruns
                st.session_state["dcir_results"] = dcir_results
                st.session_state["dcir_summary"] = wide_summary

            else:
                st.session_state["dcir_results"] = None
                st.session_state["dcir_summary"] = None
                st.info(
                    "No DCIR pulses detected in the selected NDAX files. "
                    "Check the pulse length or SOC design inputs."
                )

    # ----- OUTSIDE if run_btn, always display from session_state -----
    dcir_results = st.session_state.get("dcir_results")
    wide_summary = st.session_state.get("dcir_summary")

    if dcir_results is not None:
        st.subheader("DCIR results (raw, one row per pulse)")
        st.dataframe(dcir_results)

        if wide_summary is not None:
            st.subheader(
                f"DCIR summary (instant + {int(round(pulse_length_s))}s, SoC as columns)"
            )
            st.dataframe(wide_summary)

        # Excel: raw + summary on one file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            dcir_results.to_excel(writer, index=False, sheet_name="DCIR_raw")
            if wide_summary is not None:
                wide_summary.to_excel(
                    writer, index=False, sheet_name="DCIR_summary"
                )

        output.seek(0)
        st.download_button(
            label="⬇️ Download DCIR tables (Excel)",
            data=output,
            file_name="dcir_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    else:
        st.info(
                    "No DCIR pulses detected in the selected files. "
                    "Check the pulse length or SOC design inputs."
                )


# ---------- Box plots ----------
# with box_tab:
#     st.subheader("Distribution box plots")
#     num_cols = [c for c in data.columns if c != "__file" and pd.api.types.is_numeric_dtype(data[c])]
#     if not num_cols:
#         st.info("Need at least one numeric column.")
#     else:
#         y_box = st.selectbox(
#             "Numeric column for Y", num_cols,
#             index=(num_cols.index("Spec. Cap.(mAh/g)") if "Spec. Cap.(mAh/g)" in num_cols else 0)
#         )

#         # choose x/color field from global mode
#         if color_mode_global == "Per file":
#             group_field = "__file"
#             cmap = color_map_file
#         else:
#             # ensure __family exists (done earlier)
#             group_field = "__family"
#             cmap = color_map_fam

#         dfb = data.dropna(subset=[y_box]).copy()
#         fig_box = px.box(
#             dfb, x=group_field, y=y_box, color=group_field,
#             points="all", color_discrete_map=cmap
#         )
#         fig_box.update_layout(template="plotly_white",
#                               xaxis_title=("File" if group_field == "__file" else "Filename family"))
#         if show_grid:
#             fig_box.update_xaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)
#             fig_box.update_yaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], gridwidth=0.5)
#         st.plotly_chart(fig_box, use_container_width=True)


st.success("Loaded. Use the tabs above to explore your NDAX data.")

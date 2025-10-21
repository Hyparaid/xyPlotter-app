import io
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =============================
# App config
# =============================
st.set_page_config(
    page_title="Battery Cell Data — Visualizer",
    page_icon="🔋",
    layout="wide",
)

st.title("🔋 PRELI BATTERY CELL DATA —  VISUALIZER")
st.caption("Built by the Preli team.")

# =============================
# Helpers
# =============================
NV_COLORDICT = {
    "white": "#ffffff",
    "black": "#000000",
    "nv_power_green": "#44a27a",
    "nv_silver": "#8A8D8F",
    "nv_gray1": "#f3f3f3",
    "nv_gray2": "#e2e2e2",
    "nv_gray3": "#c0c0c0",
    "nv_gray4": "#9b9b9b",
    "nv_gray5": "#6e6e6d",
    "nv_blue1": "#a1d8d7",
    "nv_blue2": "#44acb1",
    "nv_blue3": "#21898b",
    "nv_blue4": "#006663",
    "nv_blue5": "#163f46",
    "nv_red": "#d84437",
    "nv_orange": "#f78618",
    "nv_amber": "#f7a600",
}

PALETTES = {
    "Preli long": ["#44a27a","#163f46","#a1d8d7","#e7298a",'#0383a3','#013824','#c7e9b4','#7fcdbb', '#d7301f','#990000',"#9b9b9b","#c0c0c0",'#034e7b', "#6C88AD","#5C9BBA"],
    "Preli blues": [NV_COLORDICT[c] for c in ["nv_blue1", "nv_blue2", "nv_blue3", "nv_blue4", "nv_blue5"]],
    "Preli greens": ["#c9f3e2", "#8ad5b2", "#00966c", "#1e644b", "#174335"],
    "Preli greys": [NV_COLORDICT[c] for c in ["nv_gray1", "nv_gray2", "nv_gray3", "nv_gray4", "nv_gray5", "black"]],
    "Preli Warm": ["#f78618", "#f7a600", "#d84437", "#990000"],
}

@st.cache_data(show_spinner=False)
def load_tabular(file, sheet: Optional[str] = None) -> pd.DataFrame:
    name = getattr(file, "name", "uploaded")
    suffix = Path(name).suffix.lower()
    if suffix in [".csv", ".txt", ".tsv"]:
        try:
            df = pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin-1")
        return df
    if suffix in [".xlsx", ".xlsm", ".xls"]:
        xls = pd.ExcelFile(file)
        sheet_to_use = sheet or xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet_to_use)
        return df
    raise ValueError("Unsupported file type — upload CSV or XLSX.")


def detect_columns(columns: List[str]) -> Dict[str, Optional[str]]:
    """Deterministic mapping tuned to your CSV schema (no guessing heuristics).
    Expected headers from Neware-style exports.
    """
    cols = {c.lower(): c for c in columns}
    def pick(*names):
        for n in names:
            ln = n.lower()
            if ln in cols:
                return cols[ln]
        return None
    return {
        "time": pick("Total Time", "Time"),
        "voltage": pick("Voltage(V)"),
        # Prefer specific capacity first
        "capacity": pick("Spec. Cap.(mAh/g)", "Capacity(mAh)"),
        "cycle": pick("Cycle Index"),
        "step": pick("Step Type"),
    }


def to_seconds_like(x: pd.Series) -> pd.Series:
    """Return a t0-aligned seconds vector from seconds/hours/datetimes without mutating the dataframe."""
    s = x.copy()
    if np.issubdtype(s.dtype, np.number):
        return s - s.iloc[0]
    try:
        t = pd.to_datetime(s)
        return (t - t.iloc[0]).dt.total_seconds()
    except Exception:
        s_num = pd.to_numeric(s, errors="coerce")
        return ((s_num - s_num.iloc[0]) * 3600.0) if (s_num.notna().any() and s_num.max() < 1e6) else (s_num - s_num.iloc[0])


def compute_ce(df: pd.DataFrame, cell_type: str = "cathode") -> pd.DataFrame:
    """Cycle-wise Coulombic Efficiency.
    Prefers specific capacity (mAh/g) if available, otherwise falls back to mAh.
    Uses either dedicated charge/discharge specific-capacity columns or Step Type filters.
    """
    if "Cycle Index" not in df.columns:
        return pd.DataFrame()

    has_step = "Step Type" in df.columns
    sc_chg = "Chg. Spec. Cap.(mAh/g)" if "Chg. Spec. Cap.(mAh/g)" in df.columns else None
    sc_dch = "DChg. Spec. Cap.(mAh/g)" if "DChg. Spec. Cap.(mAh/g)" in df.columns else None
    sc_any = "Spec. Cap.(mAh/g)" if "Spec. Cap.(mAh/g)" in df.columns else None
    cap_mAh = "Capacity(mAh)" if "Capacity(mAh)" in df.columns else None

    out = []
    for c in pd.unique(df["Cycle Index"].dropna()):
        sub = df[df["Cycle Index"] == c]
        q_chg = np.nan
        q_dch = np.nan

        if sc_chg and sc_dch:
            q_chg = pd.to_numeric(sub[sc_chg], errors="coerce").max()
            q_dch = pd.to_numeric(sub[sc_dch], errors="coerce").max()
        elif sc_any and has_step:
            chg = sub[sub["Step Type"] == "CC Chg"].get(sc_any)
            dch = sub[sub["Step Type"] == "CC DChg"].get(sc_any)
            q_chg = pd.to_numeric(chg, errors="coerce").max() if chg is not None else np.nan
            q_dch = pd.to_numeric(dch, errors="coerce").max() if dch is not None else np.nan
        elif cap_mAh and has_step:
            chg = sub[sub["Step Type"] == "CC Chg"].get(cap_mAh)
            dch = sub[sub["Step Type"] == "CC DChg"].get(cap_mAh)
            q_chg = pd.to_numeric(chg, errors="coerce").max() if chg is not None else np.nan
            q_dch = pd.to_numeric(dch, errors="coerce").max() if dch is not None else np.nan
        elif cap_mAh:
            q_chg = pd.to_numeric(sub.get(cap_mAh), errors="coerce").max()
            q_dch = q_chg  # fallback; CE will be 100

        if pd.isna(q_chg) or pd.isna(q_dch) or q_dch == 0 or q_chg == 0:
            ce = np.nan
        else:
            ce = (q_dch / q_chg * 100.0) if cell_type == "cathode" else (q_chg / q_dch * 100.0)
        out.append({"cycle": c, "ce": ce, "q_chg": q_chg, "q_dch": q_dch})

    return pd.DataFrame(out).sort_values("cycle")

def insert_line_breaks(df: pd.DataFrame, cap_col: str, v_col: str) -> pd.DataFrame:
    """Insert a NaN row after any index where capacity transitions from >0 to 0 to force a line break."""
    if df.empty or cap_col not in df.columns or v_col not in df.columns:
        return df
    d = df.reset_index(drop=True).copy()
    cap = pd.to_numeric(d[cap_col], errors="coerce").fillna(0)
    transitions = cap[(cap.shift(-1) == 0) & (cap > 0)].index
    if len(transitions) == 0:
        return d
    pieces = []
    last = 0
    for idx in transitions:
        pieces.append(d.iloc[last:idx+1])
        nan_row = {c: (np.nan if c in [cap_col, v_col] else d.iloc[idx].get(c)) for c in d.columns}
        pieces.append(pd.DataFrame([nan_row]))
        last = idx + 1
    pieces.append(d.iloc[last:])
    return pd.concat(pieces, ignore_index=True)

def insert_line_breaks_time_voltage(df: pd.DataFrame, t_col: str, v_col: str) -> pd.DataFrame:
    """Insert NaN rows to break lines for Voltage–Time style plots.
    Breaks at cycle changes and at transitions into/out of 'Rest'.
    """
    if df.empty or t_col not in df.columns or v_col not in df.columns:
        return df
    d = df.reset_index(drop=True).copy()
    # Try to sort by the time-like column
    try:
        sort_key = pd.to_numeric(d[t_col], errors="coerce")
        if sort_key.isna().all():
            sort_key = pd.to_timedelta(d[t_col].astype(str), errors="coerce").dt.total_seconds()
        d = d.assign(_sort=sort_key).sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    except Exception:
        pass

    break_pos = []
    n = len(d)
    if "Cycle Index" in d.columns:
        cyc = d["Cycle Index"].values
        for i in range(n - 1):
            if pd.notna(cyc[i]) and pd.notna(cyc[i+1]) and cyc[i+1] != cyc[i]:
                break_pos.append(i)
    if "Step Type" in d.columns:
        stype = d["Step Type"].astype(str)
        for i in range(n - 1):
            a, b = stype.iloc[i], stype.iloc[i+1]
            if (a == "Rest" and b != "Rest") or (a != "Rest" and b == "Rest"):
                break_pos.append(i)

    if not break_pos:
        return d

    pieces = []
    last = 0
    for idx in sorted(set(break_pos)):
        pieces.append(d.iloc[last:idx+1])
        nan_row = {c: (np.nan if c in [t_col, v_col] else d.iloc[idx].get(c)) for c in d.columns}
        pieces.append(pd.DataFrame([nan_row]))
        last = idx + 1
    pieces.append(d.iloc[last:])
    return pd.concat(pieces, ignore_index=True)


def insert_line_breaks_generic(df: pd.DataFrame, x_col: str, y_col: str,
                               *,
                               seg_cycle: bool = False,
                               seg_step: bool = False,
                               seg_cap_reset: bool = False,
                               seg_current_flip: bool = False,
                               seg_x_reverse: bool = False,
                               x_rev_eps: float = 0.0,
                               cap_col_name: Optional[str] = None) -> pd.DataFrame:
    """Generic line-break inserter for arbitrary XY combinations.
    Applies one or more segmentation rules and inserts NaN rows after the selected indices.
    """
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return df

    d = df.reset_index(drop=True).copy()
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
    if seg_cap_reset and cap_col_name and cap_col_name in d.columns:
        cap = pd.to_numeric(d[cap_col_name], errors="coerce").fillna(0)
        idxs = cap[(cap.shift(-1) == 0) & (cap > 0)].index.tolist()
        break_pos.extend(idxs)

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
            if pd.notna(dx.iloc[i]) and abs(dx.iloc[i]) > x_rev_eps:
                if dx.iloc[i] * dx.iloc[i-1] < 0:  # sign change
                    break_pos.append(i-1)

    if not break_pos:
        return d

    pieces = []
    last = 0
    for idx in sorted(set(break_pos)):
        pieces.append(d.iloc[last:idx+1])
        nan_row = {c: (np.nan if c in [x_col, y_col] else d.iloc[idx].get(c)) for c in d.columns}
        pieces.append(pd.DataFrame([nan_row]))
        last = idx + 1
    pieces.append(d.iloc[last:])
    return pd.concat(pieces, ignore_index=True)

# =============================
# Sidebar — Upload & Options (no manual column mapping)
# =============================
st.sidebar.header("Upload files")
uploaded_files = st.sidebar.file_uploader(
    "Drop CSV file (multiple allowed)", type=["csv"], accept_multiple_files=True
)

palette_name = st.sidebar.selectbox("Palette", list(PALETTES.keys()), index=0)
palette = PALETTES[palette_name]
cell_type = st.sidebar.radio("Cell type (for CE direction)", ["anode", "cathode", "full"], index=1, horizontal=True)

all_frames: List[pd.DataFrame] = []
if uploaded_files:
    with st.spinner("Parsing files…"):
        for f in uploaded_files:
            df = load_tabular(f)
            df["__file"] = f.name
            all_frames.append(df)

if not all_frames:
    st.info("Upload one or more files to begin.")
    st.stop()

# Merge raw frames (do not add derived columns)
raw = pd.concat(all_frames, ignore_index=True)

# Auto‑detect columns
colnames = list(raw.columns)
G = detect_columns(colnames)

st.sidebar.header("Display options")
files = sorted(raw["__file"].unique().tolist())
# Initialize checkboxes
if "file_checks" not in st.session_state or set(st.session_state.file_checks.keys()) != set(files):
    st.session_state.file_checks = {f: True for f in files}

b1, b2 = st.sidebar.columns(2)
with b1:
    if st.button("Select all"):
        st.session_state.file_checks = {f: True for f in files}
with b2:
    if st.button("Select none"):
        st.session_state.file_checks = {f: False for f in files}

st.sidebar.write("**Files**")
for f in files:
    st.session_state.file_checks[f] = st.sidebar.checkbox(f, value=st.session_state.file_checks.get(f, True))

selected_sources = [f for f, checked in st.session_state.file_checks.items() if checked]
row_limit = st.sidebar.slider("Preview rows", 50, 500, 200, 50)
apply_downsample = st.sidebar.checkbox("Downsample rendering (every Nth)", value=True)
N_ds = st.sidebar.number_input("N (keep every Nth row)", min_value=1, max_value=200, value=5, step=1)

if len(selected_sources) == 0:
    st.info("Select at least one file in the sidebar to plot.")
    st.stop()

# Build a per-file color map from the selected palette (for all tabs EXCEPT XY Builder)
color_map = {src: palette[i % len(palette)] for i, src in enumerate(selected_sources)}

work = raw[raw["__file"].isin(selected_sources)].copy()
if apply_downsample and N_ds > 1:
    work = work.iloc[::N_ds, :]

# Build a per-file color map from the selected palette (used in VT, VQ, CE, Box)
color_map = {src: palette[i % len(palette)] for i, src in enumerate(selected_sources)}

# Global formatting controls
with st.sidebar.expander("Formatting", expanded=False):
    show_markers = st.checkbox("Show markers", value=False)
    marker_size = st.number_input("Marker size", min_value=1, max_value=20, value=4)
    line_width = st.number_input("Line width", min_value=1.0, max_value=10.0, value=2.5, step=0.5)
    show_grid = st.checkbox("Show grid", value=True)

# Segmentation (line breaks) controls for XY Builder
# Segmentation flags (no UI): always-on generic segmentation in XY Builder
seg_cycle = True
seg_step = True
seg_cap_reset = True
seg_current_flip = True
seg_x_reverse = True
x_rev_eps = 0.0

# Preview only original columns (hide helper __file)
with st.expander("Preview parsed data (first rows)"):
    orig_cols = [c for c in work.columns if c != "__file"]
    st.dataframe(work[orig_cols].head(row_limit))

# =============================
# Tabs
# =============================
xy_tab, vt_tab, vq_tab, ce_tab, box_tab = st.tabs([
    "XY Builder", "Voltage–Time", "Voltage–Capacity", "Capacity & CE", "Box plots"
])

# ---------- XY Builder (USE PLOTLY DEFAULT COLORS) ----------
with xy_tab:
    st.subheader("Free‑form XY plot builder")
    all_cols = [c for c in work.columns if c != "__file"]
    numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(work[c])]

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        x_col = st.selectbox("X axis", all_cols, index=(all_cols.index(G["time"]) if G["time"] in all_cols else 0))
        x_log = st.checkbox("Log X", value=False)
        align_t0 = st.checkbox("Treat X as time and align t₀", value=(G["time"] == x_col))
        x_min = st.text_input("X min (blank=auto)", value="")
        x_max = st.text_input("X max (blank=auto)", value="")
    with c2:
        y_cols = st.multiselect("Y axis (one or more)", numeric_cols, default=[c for c in [G["voltage"], G["capacity"]] if c in numeric_cols][:1])
        y_log = st.checkbox("Log Y", value=False)
        y_min = st.text_input("Y min (blank=auto)", value="")
        y_max = st.text_input("Y max (blank=auto)", value="")
    with c3:
        color_by = st.selectbox("Color by", ["__file"] + all_cols, index=0)
        rolling = st.number_input("Rolling mean window (pts)", min_value=1, max_value=9999, value=1, step=1)

    plot_df = work.dropna(subset=[x_col] + y_cols).copy()
    if align_t0:
        try:
            plot_df["_x"] = to_seconds_like(plot_df[x_col])
            x_used = "_x"
        except Exception:
            x_used = x_col
    else:
        x_used = x_col

    if rolling > 1:
        plot_df = plot_df.sort_values(["__file", x_used])
        for y in y_cols:
            plot_df[y] = plot_df.groupby("__file")[y].transform(lambda s: s.rolling(rolling, min_periods=1).mean())

    fig = go.Figure()
    cap_name = G.get("capacity")
    # Always apply generic segmentation across any X/Y combo, grouped by 'color_by'
    groups = plot_df[color_by].astype(str).unique() if color_by else [None]
    for y in y_cols:
        for gval in groups:
            sdf = plot_df if gval is None else plot_df[plot_df[color_by].astype(str) == str(gval)]
            if sdf.empty:
                continue
            cols = [x_used, y, color_by] if color_by else [x_used, y]
            # include helpful columns if present
            for extra in ["__file", "Cycle Index", "Step Type", cap_name]:
                if extra and extra in sdf.columns and extra not in cols:
                    cols.append(extra)
            s2 = sdf[cols].dropna(subset=[x_used, y]).copy()
            s2 = insert_line_breaks_generic(
                s2, x_col=x_used, y_col=y,
                seg_cycle=seg_cycle, seg_step=seg_step,
                seg_cap_reset=seg_cap_reset, seg_current_flip=seg_current_flip,
                seg_x_reverse=seg_x_reverse, x_rev_eps=x_rev_eps,
                cap_col_name=cap_name,
            )
            trace_name = f"{y} — {gval}" if gval is not None else y
            fig.add_trace(go.Scatter(x=s2[x_used], y=s2[y], mode="lines", name=trace_name))

    if x_log:
        fig.update_xaxes(type="log")
    if y_log:
        fig.update_yaxes(type="log")
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
    st.plotly_chart(fig, use_container_width=True)

# ---------- Voltage–Time (USE color_map) ----------
with vt_tab:
    st.subheader("Voltage–Time")
    tcol, vcol = G["time"], G["voltage"]
    if not tcol or not vcol or tcol not in work.columns or vcol not in work.columns:
        st.warning("Couldn’t detect time/voltage. Use the XY Builder tab.")
    else:
        dfv = work.dropna(subset=[tcol, vcol]).copy()
        # Local time unit control for this tab only
        unit = st.selectbox("Time units", ["seconds", "minutes", "hours"], index=0, key="vt_time_unit")
        _DIV = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}
        _ABBR = {"seconds": "s", "minutes": "min", "hours": "h"}
        ts = pd.to_timedelta(dfv[tcol].astype(str))
        dfv["_t"] = ((ts - ts.iloc[0]).dt.total_seconds()) / _DIV[unit]
        fig_vt = go.Figure()
        for src in selected_sources:
            s = dfv[dfv["__file"] == src].sort_values(["_t"]).copy()
            if s.empty:
                continue
            mode = "lines+markers" if show_markers else "lines"
            fig_vt.add_trace(go.Scatter(
                x=s["_t"], y=s[vcol], name=src, mode=mode,
                line=dict(color=color_map.get(src), width=line_width),
                marker=dict(size=marker_size)
            ))
        fig_vt.update_layout(
            template="plotly_white",
            xaxis_title=f"Time ({_ABBR[unit]}, aligned)", yaxis_title=vcol,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        if show_grid:
            fig_vt.update_xaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], griddash="dash", gridwidth=0.5)
            fig_vt.update_yaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], griddash="dash", gridwidth=0.5)
        else:
            fig_vt.update_xaxes(showgrid=False)
            fig_vt.update_yaxes(showgrid=False)
        st.plotly_chart(fig_vt, use_container_width=True)

# ---------- Voltage–Capacity (USE color_map + line breaks) ----------
with vq_tab:
    st.subheader("Voltage–Capacity")
    ccol, vcol = G["capacity"], G["voltage"]
    if not ccol or not vcol or ccol not in work.columns or vcol not in work.columns:
        st.warning("Couldn’t detect capacity/voltage. Use the XY Builder tab.")
    else:
        df_base = work.dropna(subset=[ccol, vcol]).copy()
        cyc = G["cycle"] if G["cycle"] in df_base.columns else None
        # Existing single-cycle selection (kept as-is)
        if cyc:
            cycles_available = sorted(pd.unique(df_base[cyc].dropna()))
            sel_cycle = st.selectbox("Cycle", ["All"] + cycles_available, index=0, key="vq_single_cycle")
            if sel_cycle != "All":
                df_base = df_base[df_base[cyc] == sel_cycle]
        # NEW: optional cycle-range picker (does not affect the single-cycle selector above)
        if cyc:
            cmin, cmax = int(min(cycles_available)), int(max(cycles_available))
            # Initialize session state once
            if "vq_range" not in st.session_state:
                st.session_state.vq_range = (cmin, cmax)
            r1, r2 = st.columns([3,1])
            with r1:
                new_range = st.slider("Cycle range (optional)", min_value=cmin, max_value=cmax,
                                       value=st.session_state.vq_range, step=1, key="vq_range_slider")
            with r2:
                apply_clicked = st.button("Apply range", key="vq_apply_btn")
                reset_clicked = st.button("Reset", key="vq_reset_btn")
            if reset_clicked:
                st.session_state.vq_range = (cmin, cmax)
                st.session_state.vq_range_active = False
            if apply_clicked:
                st.session_state.vq_range = new_range
                st.session_state.vq_range_active = True
            # Apply active range filter (in addition to potential single-cycle filter above)
            if st.session_state.get("vq_range_active", False) and sel_cycle == "All":
                lo, hi = st.session_state.vq_range
                df_base = df_base[df_base[cyc].between(lo, hi)]
                st.caption(f"Showing cycles {lo}–{hi}")

        fig_vq = go.Figure()
        for src in selected_sources:
            sdf = df_base[df_base["__file"] == src]
            if sdf.empty:
                continue
            plot_df = insert_line_breaks(sdf[[ccol, vcol, "__file"]], cap_col=ccol, v_col=vcol)
            fig_vq.add_trace(go.Scatter(x=plot_df[ccol], y=plot_df[vcol], name=src,
                                        mode=("lines+markers" if show_markers else "lines"),
                                        line=dict(color=color_map.get(src), width=line_width),
                                        marker=dict(size=marker_size)))
        x_label = f"{ccol}" if "mAh/g" in ccol else ccol
        fig_vq.update_layout(template="plotly_white", xaxis_title=x_label, yaxis_title=vcol,
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        if show_grid:
            fig_vq.update_xaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], griddash="dash", gridwidth=0.5)
            fig_vq.update_yaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], griddash="dash", gridwidth=0.5)
        else:
            fig_vq.update_xaxes(showgrid=False)
            fig_vq.update_yaxes(showgrid=False)
        st.plotly_chart(fig_vq, use_container_width=True)

# ---------- Capacity & CE (USE color_map) ----------
with ce_tab:
    st.subheader("Capacity & Coulombic Efficiency vs Cycle (dual‑axis)")
    cyc = G["cycle"]
    if not cyc or cyc not in work.columns:
        st.info("No cycle index detected; map via XY Builder instead.")
    else:
        # Pick the capacity column for the left axis (prefer mAh/g)
        cap_col = None
        for cand in ["Spec. Cap.(mAh/g)", "DChg. Spec. Cap.(mAh/g)", "Chg. Spec. Cap.(mAh/g)", "Capacity(mAh)"]:
            if cand in work.columns:
                cap_col = cand
                break
        if cap_col is None:
            st.info("No capacity column found for Capacity vs Cycle.")
        else:
            # Cycle range slider
            cycles_avail = sorted(pd.unique(work[cyc].dropna()))
            if len(cycles_avail) == 0:
                st.info("No cycles available to plot.")
            else:
                cmin, cmax = int(min(cycles_avail)), int(max(cycles_avail))
                sel_min, sel_max = st.slider("Cycle range", min_value=cmin, max_value=cmax, value=(cmin, cmax), step=1)
                mask_range = work[cyc].between(sel_min, sel_max)
                work_rng = work[mask_range]

                fig_ce = make_subplots(specs=[[{"secondary_y": True}]])
                # Compute CE (respects specific capacity when present)
                ce_cell_type = ("cathode" if cell_type == "full" else cell_type)
                for src in selected_sources:
                    sub = work_rng[work_rng["__file"] == src]
                    if sub.empty:
                        continue
                    ce_df = compute_ce(sub, cell_type=ce_cell_type)
                    cap_cycle = sub.groupby(cyc)[cap_col].max().reset_index()
                    if not cap_cycle.empty:
                        # Capacity trace (left axis) — HOLLOW markers
                        fig_ce.add_trace(
                            go.Scatter(
                                x=cap_cycle[cyc], y=cap_cycle[cap_col], name=f"Cap — {src}",
                                mode=("lines+markers" if show_markers else "lines"),
                                line=dict(color=color_map.get(src), width=line_width),
                                marker=dict(
                                    size=marker_size,
                                    symbol="circle-open",
                                    color=color_map.get(src),
                                    line=dict(color=color_map.get(src), width=max(1, int(line_width - 1)))
                                ),
                            ),
                            secondary_y=False,
                        )
                        # CE trace (right axis) — SOLID markers, dashed line
                        fig_ce.add_trace(
                            go.Scatter(
                                x=ce_df["cycle"], y=ce_df["ce"], name=f"CE — {src}",
                                mode=("lines+markers" if show_markers else "lines"),
                                line=dict(color=color_map.get(src), dash="dash", width=line_width),
                                marker=dict(size=marker_size, symbol="circle", color=color_map.get(src)),
                            ),
                            secondary_y=True,
                        )


                y_left_label = "Specific capacity (mAh/g)" if "mAh/g" in cap_col else "Capacity (mAh)"
                fig_ce.update_yaxes(title_text=y_left_label, secondary_y=False)
                fig_ce.update_yaxes(title_text="CE (%)", range=[90, 105], secondary_y=True)
                fig_ce.update_xaxes(title_text="Cycle")
                fig_ce.update_layout(template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02))
                if show_grid:
                    fig_ce.update_xaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], griddash="dash", gridwidth=0.5)
                    fig_ce.update_yaxes(showgrid=True, gridcolor=NV_COLORDICT["nv_gray3"], griddash="dash", gridwidth=0.5)
                else:
                    fig_ce.update_xaxes(showgrid=False)
                    fig_ce.update_yaxes(showgrid=False)
                st.plotly_chart(fig_ce, use_container_width=True)

# ---------- Box plots (USE color_map) ----------
with box_tab:
    st.subheader("Distribution box plots")
    all_cols = [c for c in work.columns if c != "__file"]
    num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(work[c])]

    def _family_from_filename(name: str) -> str:
        stem = Path(str(name)).stem
        # Split on common delimiters and take the first token
        for sep in ["_", "-", " "]:
            if sep in stem:
                stem = stem.split(sep)[0]
                break
        # Strip trailing digits
        return stem.rstrip("0123456789") or Path(str(name)).stem

    if not num_cols:
        st.info("Need at least one numeric column.")
    else:
        y_box = st.selectbox(
            "Numeric column for Y",
            num_cols,
            index=(num_cols.index(G["capacity"]) if G["capacity"] in num_cols else 0),
            key="box_y_col",
        )

        # Grouping mode: per-file vs filename family (e.g., linlib_1, linlib_2 → linlib)
        grp_mode = st.radio("Group by", ["Per file", "Filename family"], index=0, horizontal=True, key="box_grp_mode")
        dfb = work.dropna(subset=[y_box]).copy()
        if grp_mode == "Filename family":
            dfb["__group"] = dfb["__file"].apply(_family_from_filename)
            groups = sorted(dfb["__group"].unique())
            color_map_box = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
            fig_box = px.box(dfb, x="__group", y=y_box, color="__group", points="all", color_discrete_map=color_map_box)
        else:
            groups = sorted(dfb["__file"].unique())
            color_map_box = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
            fig_box = px.box(dfb, x="__file", y=y_box, color="__file", points="all", color_discrete_map=color_map_box)

        fig_box.update_layout(template="plotly_white")
        # Standardized dashed grid
        fig_box.update_xaxes(showgrid=show_grid, gridcolor=NV_COLORDICT["nv_gray3"], griddash="dash", gridwidth=0.5)
        fig_box.update_yaxes(showgrid=show_grid, gridcolor=NV_COLORDICT["nv_gray3"], griddash="dash", gridwidth=0.5)
        st.plotly_chart(fig_box, use_container_width=True)

st.success("Loaded. Use the tabs above to explore your data. Save images via the Plotly 'camera' icon.")


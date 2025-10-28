# ndax_min_builder.py (updated core)
from __future__ import annotations
import re, zipfile, pathlib
import pandas as pd
import numpy as np

def _extract_scq_from_ndax_path(ndax_path: str | pathlib.Path) -> float | None:
    ndax_path = pathlib.Path(ndax_path)
    with zipfile.ZipFile(ndax_path, "r") as zf:
        if "Step.xml" not in zf.namelist():
            return None
        raw = zf.read("Step.xml")
    for enc in ("utf-8","gb18030","gb2312","utf-16","latin-1"):
        try:
            text = raw.decode(enc); break
        except Exception:
            continue
    else:
        text = raw.decode("utf-8", errors="ignore")
    m = re.search(r'SCQ\s+Value="([\d\.,Ee+-]+)"', text)
    return float(m.group(1).replace(",", ".")) if m else None

def _parse_status_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add Step Mode + Step Type from Status (e.g., 'CCCV_Chg' → Mode='CCCV', Type='Chg').
       Also marks rest-like steps as Step Type = 'Rest'."""
    if "Status" not in df.columns:
        return df

    s_norm = df["Status"].astype(str).str.strip()
    parts = s_norm.str.rsplit("_", n=1)
    step_mode = parts.str[0]
    tail = parts.str[1].fillna("")

    rest_keywords = ("rest", "ocv", "pause", "hold", "wait", "idle", "stop")

    def canon_type(full: str, tail_part: str) -> str:
        f = full.lower()
        t = tail_part.lower()
        # 1) Rest-like steps (anywhere in the full string)
        if any(k in f for k in rest_keywords):
            return "Rest"
        # 2) Tail-based charge/discharge
        if ("d" in t and "chg" in t) or ("dis" in t):
            return "DChg"
        if "chg" in t:
            return "Chg"
        # 3) No explicit tail → infer from full string if possible
        if "dchg" in f or "dis" in f:
            return "DChg"
        if "chg" in f:
            return "Chg"
        return "Rest" if any(k in f for k in rest_keywords) else ""  # default

    step_type = [canon_type(full, t) for full, t in zip(s_norm, tail)]

    # (Optional) normalize common mode tokens; keep as-is if you prefer original
    step_mode = (step_mode
                 .str.replace(r"\bCCC\b", "CCC", regex=True)
                 .str.replace(r"\bCCCV\b", "CCCV", regex=True)
                 .str.replace(r"\bOCV\b", "OCV", regex=True))

    df["Step Mode"] = step_mode
    df["Step Type"] = pd.Series(step_type, index=df.index)
    return df

def build_ndax_df(ndax_path: str | pathlib.Path, cycle_mode: str = "auto") -> pd.DataFrame:
    try:
        import NewareNDA
    except Exception as e:
        raise RuntimeError("NewareNDA not installed. pip install NewareNDA") from e

    ndax_path = pathlib.Path(ndax_path)
    df = NewareNDA.read(str(ndax_path), cycle_mode=cycle_mode)

    # 1) parse Status → Step Mode / Step Type
    df = _parse_status_cols(df)

    # 2) mass from SCQ/1e6 (your rule)
    scq = _extract_scq_from_ndax_path(ndax_path)
    active_mass_g = scq / 1_000_000.0 if scq is not None else None

    # 3) locate capacity columns emitted by NewareNDA
    chg_cap = next((c for c in ["Charge_Capacity(mAh)","Chg. Cap.(mAh)","Q_charge(mAh)","ChargeCap(mAh)"] if c in df.columns), None)
    dchg_cap = next((c for c in ["Discharge_Capacity(mAh)","DChg. Cap.(mAh)","Q_discharge(mAh)","DischargeCap(mAh)"] if c in df.columns), None)

    # 4) Capacity(mAh) per Step Type
    cap_series = pd.Series(np.nan, index=df.index, dtype="float64")
    if "Step Type" in df.columns:
        is_chg  = df["Step Type"].eq("Chg")
        is_dchg = df["Step Type"].eq("DChg")
        if chg_cap is not None:
            cap_series = np.where(is_chg,  pd.to_numeric(df[chg_cap],  errors="coerce"), cap_series)
        if dchg_cap is not None:
            cap_series = np.where(is_dchg, pd.to_numeric(df[dchg_cap], errors="coerce"), cap_series)
    else:
        if chg_cap and not dchg_cap:
            cap_series = pd.to_numeric(df[chg_cap], errors="coerce")
        elif dchg_cap and not chg_cap:
            cap_series = pd.to_numeric(df[dchg_cap], errors="coerce")

    df["Capacity(mAh)"] = cap_series

    # 5) Specific capacities (vectorized)
    if active_mass_g and active_mass_g > 0:
        df["Spec. Cap.(mAh/g)"] = df["Capacity(mAh)"] / active_mass_g
        if chg_cap:
            df["Chg. Spec. Cap.(mAh/g)"]  = pd.to_numeric(df[chg_cap],  errors="coerce") / active_mass_g
        if dchg_cap:
            df["DChg. Spec. Cap.(mAh/g)"] = pd.to_numeric(df[dchg_cap], errors="coerce") / active_mass_g

    # keep originals first; append deriveds
    derived = ["Step Mode","Step Type","Capacity(mAh)","Spec. Cap.(mAh/g)","Chg. Spec. Cap.(mAh/g)","DChg. Spec. Cap.(mAh/g)"]
    ordered = [c for c in df.columns if c not in derived] + [c for c in derived if c in df.columns]
    df = df[ordered]

    df.attrs["SCQ_raw"] = scq
    df.attrs["active_mass_g"] = active_mass_g
    return df

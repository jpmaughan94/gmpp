"""Read one raw GMPP file and turn it into the canonical schema.

Everything here is a pure function of the raw file, so the raw archive can be
re-processed at any time.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONFIG = Path(__file__).resolve().parents[2] / "config"

# --------------------------------------------------------------------------- reading

ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def read_raw(path: str | Path) -> pd.DataFrame:
    """Read a CSV/XLSX/ODS with encoding fallback; transpose if projects run across columns."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    elif suffix == ".ods":
        df = pd.read_excel(path, engine="odf", dtype=str)
    else:
        raw = path.read_bytes()
        last_err = None
        for enc in ENCODINGS:
            try:
                df = pd.read_csv(io.StringIO(raw.decode(enc)), dtype=str, keep_default_na=False)
                break
            except (UnicodeDecodeError, pd.errors.ParserError) as e:  # pragma: no cover
                last_err = e
        else:  # pragma: no cover
            raise last_err
    df = _drop_preamble(df)
    if _is_transposed(df):
        df = _transpose(df)
    df.columns = [norm_header(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    return df.replace({"": np.nan}).dropna(how="all")


def norm_header(h) -> str:
    h = str(h).replace("£", "£").replace("\n", " ")
    return re.sub(r"\s+", " ", h).strip().lower()


def _drop_preamble(df: pd.DataFrame) -> pd.DataFrame:
    """Some files have title rows above the header. Find the row containing 'project name'."""
    if any("project name" in norm_header(c) for c in df.columns):
        return df
    for i in range(min(15, len(df))):
        row = [norm_header(v) for v in df.iloc[i].tolist()]
        if any("project name" in v for v in row):
            out = df.iloc[i + 1 :].copy()
            out.columns = df.iloc[i].tolist()
            return out.reset_index(drop=True)
    return df


_FIELD_HINTS = ("department", "delivery confidence", "whole life", "narrative", "description")


def _is_transposed(df: pd.DataFrame) -> bool:
    """Projects run across columns when the first column holds field labels."""
    first = df.iloc[:, 0].astype(str).map(norm_header)
    return sum(any(h in v for h in _FIELD_HINTS) for v in first) >= 2


def _transpose(df: pd.DataFrame) -> pd.DataFrame:
    head = norm_header(df.columns[0])
    t = df.set_index(df.columns[0]).T
    t.insert(0, "project name" if head == "project name" else "_col0", t.index)
    t = t.reset_index(drop=True)
    t.columns = [str(c) for c in t.columns]
    return t


# --------------------------------------------------------------------------- columns

def load_column_config() -> dict:
    return yaml.safe_load((CONFIG / "columns.yaml").read_text())


def map_columns(headers: list[str], snapshot_id: str | None = None, cfg: dict | None = None) -> dict[str, str]:
    """Return {canonical_field: raw_header}. Overrides win; otherwise first unique regex match."""
    cfg = cfg or load_column_config()
    overrides = (cfg.get("overrides") or {}).get(snapshot_id, {}) if snapshot_id else {}
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for field, h in overrides.items():
        mapping[field] = norm_header(h)
        used.add(norm_header(h))
    for field, spec in cfg["fields"].items():
        if field in mapping:
            continue
        excl = [re.compile(x) for x in spec.get("exclude", [])]
        for pat in spec["patterns"]:
            rx = re.compile(pat)
            hits = [h for h in headers if h not in used and rx.search(h) and not any(e.search(h) for e in excl)]
            if len(hits) >= 1:
                # prefer the shortest header: "project name" over "project name (if changed)"
                best = min(hits, key=len)
                mapping[field] = best
                used.add(best)
                break
    return mapping


# --------------------------------------------------------------------------- values

DCA5 = {"green": 1, "amber/green": 2, "amber": 3, "amber/red": 4, "red": 5}
DCA5_LABELS = {v: k.title() for k, v in DCA5.items()}
NON_RATINGS = {"exempt", "reset", "not provided", "data not provided", "n/a", "na", "-", "none"}


def norm_dca(value) -> tuple[str | None, str | None]:
    """Return (label, status). label is one of the five-point labels; status flags exempt/reset/missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, "missing"
    v = re.sub(r"\s+", "", str(value).strip().lower())
    v = v.replace("-", "/").replace("_", "/").replace("&", "/")
    v = {"ag": "amber/green", "ar": "amber/red", "a/g": "amber/green", "a/r": "amber/red", "g": "green", "a": "amber", "r": "red",
         "ambergreen": "amber/green", "amberred": "amber/red"}.get(v, v)
    if v in DCA5:
        return DCA5_LABELS[DCA5[v]], "rated"
    if any(v.startswith(n.replace(" ", "")) for n in NON_RATINGS):
        return None, "exempt" if v.startswith("exempt") else ("reset" if v.startswith("reset") else "missing")
    return None, "unrecognised"


def harmonise_dca(label: str | None, scale: int) -> dict:
    """Put five-point (to Mar 2022) and three-point (from Mar 2023) ratings on comparable footing.

    The IPA moved to Green/Amber/Red for the March 2023 data with new definitions, so
    there is no exact mapping. We keep the raw label and two versions of 'red':
      red_strict = Red only;  red_broad = Red or Amber/Red.
    rag3 maps Amber/Green -> Green and Amber/Red -> Red (sensitivity: use red_strict).
    """
    out = {"dca": label, "dca_scale": scale, "rag3": None, "red_strict": None, "red_broad": None}
    if label is None:
        return out
    l = label.lower()
    out["rag3"] = {"green": "Green", "amber/green": "Green", "amber": "Amber",
                   "amber/red": "Red", "red": "Red"}[l]
    out["red_strict"] = l == "red"
    out["red_broad"] = l in ("red", "amber/red")
    return out


_MONEY_RX = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_money_m(value) -> float | None:
    """Parse a cost cell to £ millions. Handles '£1,234.5m', '1.2bn', '(12.3)', '£12,300,000'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower().replace("£", "").replace("?", "")
    if not s or any(w in s for w in ("exempt", "not provided", "n/a", "tbc", "commercial")):
        return None
    neg = s.startswith("(") and s.endswith(")")
    m = _MONEY_RX.search(s)
    if not m:
        return None
    x = float(m.group().replace(",", ""))
    if "bn" in s or "billion" in s:
        x *= 1000
    elif x > 1e6 and "m" not in s:  # raw pounds
        x /= 1e6
    return -x if neg else x


def parse_date(value) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if not s or s.lower() in NON_RATINGS or "exempt" in s.lower():
        return None
    if re.fullmatch(r"\d{5}(\.0)?", s):  # Excel serial
        return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(float(s)))
    try:
        return pd.to_datetime(s, dayfirst=True, errors="raise")
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- canonical table

TEXT_FIELDS = ["project_id", "project_name", "department", "category", "description",
               "dca_narrative", "schedule_narrative", "budget_narrative", "benefits_narrative"]
MONEY_FIELDS = ["wlc_baseline", "wlc_forecast", "benefits_baseline"]
DATE_FIELDS = ["start_date", "end_date_baseline", "end_date_forecast"]


def to_canonical(df: pd.DataFrame, snapshot_id: str, snapshot_date: str, cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    mapping = map_columns(list(df.columns), snapshot_id, cfg)
    out = pd.DataFrame(index=df.index)
    for f in TEXT_FIELDS:
        out[f] = df[mapping[f]].astype("string").str.strip() if f in mapping else pd.NA
    for f in MONEY_FIELDS:
        out[f] = df[mapping[f]].map(parse_money_m) if f in mapping else np.nan
    for f in DATE_FIELDS:
        out[f] = pd.to_datetime(df[mapping[f]].map(parse_date)) if f in mapping else pd.NaT
    scale = 5 if pd.Timestamp(snapshot_date) < pd.Timestamp("2023-01-01") else 3
    parsed = df[mapping["dca"]].map(norm_dca) if "dca" in mapping else pd.Series([(None, "missing")] * len(df), index=df.index)
    out["dca_raw"] = df[mapping["dca"]] if "dca" in mapping else pd.NA
    out["dca_status"] = parsed.map(lambda t: t[1])
    harm = pd.DataFrame([harmonise_dca(t[0], scale) for t in parsed], index=df.index)
    out = pd.concat([out, harm], axis=1)
    out.insert(0, "snapshot", snapshot_id)
    out.insert(1, "snapshot_date", pd.Timestamp(snapshot_date))
    out = out[out["project_name"].notna() & (out["project_name"].str.len() > 0)].reset_index(drop=True)
    report = {"snapshot": snapshot_id, "rows": len(out), "mapping": mapping,
              "unmapped_fields": sorted(set(cfg["fields"] if cfg else load_column_config()["fields"]) - set(mapping)),
              "dca_status": out["dca_status"].value_counts().to_dict()}
    return out, report

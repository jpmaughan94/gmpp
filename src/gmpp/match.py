"""Link the same project across annual snapshots.

Names drift ("Crossrail" -> "Crossrail Programme"), departments are renamed or split
(BEIS -> DESNZ/DSIT/DBT), and IDs only exist in some years. The approach:

1. exact ID match where both snapshots carry an ID;
2. a similarity score for every remaining pair (normalised-name fuzzy match,
   TF-IDF description similarity, same department lineage);
3. one-to-one assignment (Hungarian algorithm) above a threshold;
4. unmatched projects get a second chance against the snapshot after next
   (projects that skip a year, e.g. exempt or temporarily off the portfolio);
5. hand decisions in config/match_overrides.csv always win.

Links are chained into a stable `project_uid` with union-find.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CONFIG = Path(__file__).resolve().parents[2] / "config"

# Machinery-of-government lineages: departments that are the "same" for matching.
DEPT_FAMILY = {
    "decc": "energy-business", "bis": "energy-business", "beis": "energy-business",
    "desnz": "energy-business", "dsit": "energy-business", "dbt": "energy-business",
    "dit": "energy-business", "dcms": "dcms",
    "dclg": "housing", "mhclg": "housing", "dluhc": "housing",
    "dh": "health", "dhsc": "health", "nhs": "health",
    "fco": "foreign", "dfid": "foreign", "fcdo": "foreign",
    "mod": "defence", "moj": "justice", "ho": "home", "dft": "transport", "dwp": "dwp",
    "hmrc": "hmrc", "hmt": "treasury", "dfe": "education", "defra": "defra", "co": "cabinet",
    "dexeu": "cabinet", "ons": "ons", "uksa": "ons", "nca": "nca", "fsa": "fsa", "cps": "cps",
}
DEPT_ALIASES = {
    "ministry of defence": "mod", "ministry of justice": "moj", "home office": "ho",
    "department for transport": "dft", "department for work and pensions": "dwp",
    "hm revenue and customs": "hmrc", "hm revenue & customs": "hmrc", "hm treasury": "hmt",
    "department for education": "dfe", "cabinet office": "co",
    "department of health and social care": "dhsc", "department of health": "dh",
    "department for environment, food and rural affairs": "defra",
    "department for energy security and net zero": "desnz",
    "department for science, innovation and technology": "dsit",
    "department for business and trade": "dbt",
    "department for business, energy and industrial strategy": "beis",
    "department for levelling up, housing and communities": "dluhc",
    "ministry of housing, communities and local government": "mhclg",
    "department for culture, media and sport": "dcms",
    "department for digital, culture, media and sport": "dcms",
    "foreign, commonwealth and development office": "fcdo",
    "foreign, commonwealth & development office": "fcdo",
    "office for national statistics": "ons", "national crime agency": "nca",
}

STOP = {"the", "programme", "program", "project", "projects", "and", "of", "for", "phase", "a", "to", "in"}


def dept_code(name) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)) or pd.isna(name):
        return ""
    s = re.sub(r"\s+", " ", str(name).strip().lower())
    s = DEPT_ALIASES.get(s, s)
    s = re.sub(r"[^a-z]", "", s) if len(s) <= 8 else s
    return s


def dept_family(name) -> str:
    c = dept_code(name)
    return DEPT_FAMILY.get(c, c)


def norm_name(name: str) -> str:
    s = str(name).lower().replace("&", " and ")
    s = re.sub(r"\(.*?\)", " ", s)          # drop bracketed asides
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in STOP]
    return " ".join(toks)


def name_similarity(a: str, b: str) -> float:
    na, nb = norm_name(a), norm_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return max(fuzz.token_set_ratio(na, nb), fuzz.ratio(na, nb)) / 100


def pair_scores(a: pd.DataFrame, b: pd.DataFrame, w_name=0.6, w_desc=0.25, w_dept=0.15) -> np.ndarray:
    """Similarity matrix (len(a) x len(b)) in [0, 1]."""
    names = np.array([[name_similarity(x, y) for y in b["project_name"]] for x in a["project_name"]])
    da = a["description"].fillna("").astype(str).tolist()
    db = b["description"].fillna("").astype(str).tolist()
    if any(da) and any(db):
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1).fit(da + db)
        desc = cosine_similarity(vec.transform(da), vec.transform(db))
        has = np.outer([bool(x) for x in da], [bool(y) for y in db])
        # where a description is missing, re-weight onto the name
        desc_w = np.where(has, w_desc, 0.0)
    else:
        desc = np.zeros_like(names)
        desc_w = np.zeros_like(names)
    fa = [dept_family(x) for x in a["department"]]
    fb = [dept_family(y) for y in b["department"]]
    dept = np.array([[1.0 if x and x == y else 0.0 for y in fb] for x in fa])
    name_w = w_name + (w_desc - desc_w)
    return (name_w * names + desc_w * desc + w_dept * dept) / (name_w + desc_w + w_dept)


def load_overrides(path: Path | None = None) -> pd.DataFrame:
    path = path or CONFIG / "match_overrides.csv"
    if not path.exists():
        return pd.DataFrame(columns=["snapshot_a", "name_a", "snapshot_b", "name_b", "decision", "note"])
    return pd.read_csv(path, dtype=str).fillna("")


def match_pair(a: pd.DataFrame, b: pd.DataFrame, threshold=0.72, review_band=0.6,
               overrides: pd.DataFrame | None = None) -> pd.DataFrame:
    """One-to-one links between snapshot a and snapshot b. Returns rows with indices, score and method."""
    links = []
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    for d in (a, b):
        if "row_key" not in d:
            d["row_key"] = d["snapshot"] + "#" + d.index.astype(str)
    free_a, free_b = set(a.index), set(b.index)
    sa, sb = a["snapshot"].iat[0], b["snapshot"].iat[0]

    # 0. hand decisions
    forbid = set()
    if overrides is not None and len(overrides):
        ov = overrides[(overrides.snapshot_a == sa) & (overrides.snapshot_b == sb)]
        for _, r in ov.iterrows():
            ia = a.index[a.project_name == r.name_a]
            ib = b.index[b.project_name == r.name_b]
            if len(ia) and len(ib):
                if r.decision == "link" and ia[0] in free_a and ib[0] in free_b:
                    links.append((ia[0], ib[0], 1.0, "override"))
                    free_a.discard(ia[0]); free_b.discard(ib[0])
                elif r.decision == "nolink":
                    forbid.add((ia[0], ib[0]))

    # 1. exact IDs
    if a["project_id"].notna().any() and b["project_id"].notna().any():
        idb = {str(v).strip().lower(): i for i, v in b["project_id"].items() if pd.notna(v) and i in free_b}
        for i, v in a["project_id"].items():
            k = str(v).strip().lower() if pd.notna(v) else None
            if i in free_a and k and k in idb and idb[k] in free_b and (i, idb[k]) not in forbid:
                links.append((i, idb[k], 1.0, "id"))
                free_a.discard(i); free_b.discard(idb[k])

    # 2-3. similarity + assignment
    ra, rb = sorted(free_a), sorted(free_b)
    if ra and rb:
        S = pair_scores(a.loc[ra], b.loc[rb])
        for (i, j) in forbid:
            if i in ra and j in rb:
                S[ra.index(i), rb.index(j)] = 0
        r_idx, c_idx = linear_sum_assignment(-S)
        for r, c in zip(r_idx, c_idx):
            s = float(S[r, c])
            if s >= threshold:
                links.append((ra[r], rb[c], s, "auto"))
            elif s >= review_band:
                links.append((ra[r], rb[c], s, "review"))

    out = pd.DataFrame(links, columns=["ia", "ib", "score", "method"])
    out["snapshot_a"], out["snapshot_b"] = sa, sb
    out["name_a"] = a.loc[out.ia, "project_name"].values if len(out) else []
    out["name_b"] = b.loc[out.ib, "project_name"].values if len(out) else []
    out["dept_a"] = a.loc[out.ia, "department"].values if len(out) else []
    out["dept_b"] = b.loc[out.ib, "department"].values if len(out) else []
    out["row_a"] = a.loc[out.ia, "row_key"].values if len(out) else []
    out["row_b"] = b.loc[out.ib, "row_key"].values if len(out) else []
    return out.sort_values("score").reset_index(drop=True)


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def link_snapshots(panel: pd.DataFrame, accept_review=False, **kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign project_uid across all snapshots. Returns (panel with uid, links table)."""
    panel = panel.copy()
    panel["row_key"] = panel["snapshot"] + "#" + panel.groupby("snapshot").cumcount().astype(str)
    snaps = sorted(panel["snapshot"].unique())
    overrides = load_overrides()
    by = {s: panel[panel.snapshot == s].reset_index(drop=True) for s in snaps}
    all_links = []
    matched_b: dict[str, set] = {s: set() for s in snaps}
    matched_a: dict[str, set] = {s: set() for s in snaps}
    ok = {"auto", "id", "override"} | ({"review"} if accept_review else set())
    for k in range(len(snaps) - 1):
        sa, sb = snaps[k], snaps[k + 1]
        L = match_pair(by[sa], by[sb], overrides=overrides, **kw)
        all_links.append(L)
        good = L[L.method.isin(ok)]
        matched_a[sa] |= set(good.row_a); matched_b[sb] |= set(good.row_b)
    # skip-one-snapshot pass for leftovers
    for k in range(len(snaps) - 2):
        sa, sc = snaps[k], snaps[k + 2]
        A = by[sa][~by[sa].row_key.isin(matched_a[sa])]
        C = by[sc][~by[sc].row_key.isin(matched_b[sc])]
        if len(A) and len(C):
            L = match_pair(A, C, overrides=overrides, **kw)
            L["method"] = L["method"].replace({"auto": "auto_skip"})
            all_links.append(L)
            good = L[L.method.isin(ok | {"auto_skip"})]
            matched_a[sa] |= set(good.row_a); matched_b[sc] |= set(good.row_b)
    links = pd.concat(all_links, ignore_index=True) if all_links else pd.DataFrame()
    uf = _UF()
    for rk in panel.row_key:
        uf.find(rk)
    for _, r in links[links.method.isin(ok | {"auto_skip"})].iterrows():
        uf.union(r.row_a, r.row_b)
    roots = panel.row_key.map(uf.find)
    # stable uid: first snapshot + normalised name of the earliest row in the chain
    first = panel.assign(root=roots).sort_values("snapshot_date").groupby("root").head(1)
    uid = {r.root: f"{r.snapshot}:{norm_name(r.project_name)[:40].replace(' ', '-')}" for r in first.itertuples()}
    panel["project_uid"] = roots.map(uid)
    return panel, links

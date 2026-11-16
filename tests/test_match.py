import pandas as pd

from gmpp.match import dept_family, link_snapshots, match_pair, name_similarity, norm_name


def snap(sid, date, rows):
    df = pd.DataFrame(rows, columns=["project_name", "department", "description"])
    df["project_id"] = pd.NA
    df.insert(0, "snapshot", sid)
    df.insert(1, "snapshot_date", pd.Timestamp(date))
    return df


def test_norm_name_strips_noise():
    assert norm_name("The Crossrail Programme (Elizabeth Line)") == "crossrail"
    assert name_similarity("Crossrail", "Crossrail Programme") == 1.0


def test_department_lineage():
    assert dept_family("BEIS") == dept_family("DESNZ") == dept_family("Department for Business and Trade")
    assert dept_family("DCLG") == dept_family("DLUHC")
    assert dept_family("MoD") != dept_family("DfT")


A = snap("2019-09", "2019-09-30", [
    ("Crossrail", "DfT", "New east-west railway through London"),
    ("Smart Meter Implementation Programme", "BEIS", "Roll out smart gas and electricity meters to homes"),
    ("Universal Credit Programme", "DWP", "Replace six legacy benefits with a single payment"),
    ("Dreadnought", "MoD", "Successor ballistic missile submarines"),
])
B = snap("2021-03", "2021-03-31", [
    ("Universal Credit", "DWP", "Single payment replacing six legacy benefits"),
    ("Smart Metering Implementation Programme", "BEIS", "Smart gas and electricity meters for homes"),
    ("Crossrail Programme", "DfT", "Elizabeth line east-west railway"),
    ("New Hospital Programme", "DHSC", "Build 40 new hospitals"),
])
C = snap("2022-03", "2022-03-31", [
    ("Smart Metering Implementation Programme", "DESNZ", "Smart meters rollout"),
    ("Dreadnought Programme", "MOD", "Successor submarines to Vanguard class"),
    ("New Hospital Programme", "DHSC", "Build 40 new hospitals by 2030"),
])


def test_match_pair_one_to_one():
    L = match_pair(A, B)
    good = L[L.method == "auto"]
    pairs = set(zip(good.name_a, good.name_b))
    assert ("Crossrail", "Crossrail Programme") in pairs
    assert ("Universal Credit Programme", "Universal Credit") in pairs
    assert ("Smart Meter Implementation Programme", "Smart Metering Implementation Programme") in pairs
    assert not any(n == "Dreadnought" for n in good.name_a)       # nothing to match in B
    assert good.ib.is_unique and good.ia.is_unique


def test_link_snapshots_chains_and_skips_a_year():
    panel, _links = link_snapshots(pd.concat([A, B, C], ignore_index=True))
    uid = panel.groupby("project_uid").snapshot.apply(set)
    smart = panel[panel.project_name.str.startswith("Smart")].project_uid.unique()
    assert len(smart) == 1 and len(uid[smart[0]]) == 3                # BEIS -> DESNZ handled
    dread = panel[panel.project_name.str.startswith("Dreadnought")].project_uid.unique()
    assert len(dread) == 1                                            # linked across the 2021 gap
    nhp = panel[panel.project_name == "New Hospital Programme"].project_uid.unique()
    assert len(nhp) == 1

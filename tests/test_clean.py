import numpy as np
import pandas as pd
import pytest

from gmpp.clean import (
    harmonise_dca,
    load_column_config,
    map_columns,
    norm_dca,
    norm_header,
    parse_date,
    parse_money_m,
    read_raw,
    to_canonical,
)


@pytest.mark.parametrize("raw,label,status", [
    ("Amber/Red", "Amber/Red", "rated"), ("AMBER / GREEN", "Amber/Green", "rated"),
    ("Amber-Red", "Amber/Red", "rated"), ("A/G", "Amber/Green", "rated"), ("Red", "Red", "rated"),
    ("Exempt", None, "exempt"), ("Reset", None, "reset"), (np.nan, None, "missing"),
    ("purple", None, "unrecognised"),
])
def test_norm_dca(raw, label, status):
    assert norm_dca(raw) == (label, status)


def test_harmonise_keeps_two_red_definitions():
    h = harmonise_dca("Amber/Red", 5)
    assert h["rag3"] == "Red" and h["red_broad"] and not h["red_strict"]
    assert harmonise_dca("Amber/Green", 5)["rag3"] == "Green"
    assert harmonise_dca(None, 3)["rag3"] is None


@pytest.mark.parametrize("raw,val", [
    ("£1,234.5m", 1234.5), ("1.2bn", 1200), ("(12.3)", -12.3), ("£12,300,000", 12.3),
    ("Exempt", None), ("", None), ("456", 456.0), ("?21.1m", 21.1),
])
def test_parse_money(raw, val):
    got = parse_money_m(raw)
    assert (got is None and val is None) or got == pytest.approx(val)


def test_parse_date_formats():
    assert parse_date("31/03/2026") == pd.Timestamp("2026-03-31")
    assert parse_date("44651") == pd.Timestamp("2022-03-31")
    assert parse_date("Exempt") is None


def test_map_columns_avoids_narrative_for_rating():
    headers = [norm_header(h) for h in [
        "Project Name", "Department", "IPA Delivery Confidence Assessment",
        "Departmental commentary on actions planned or taken on the IPA RAG rating",
        "Total Baseline Whole Life Costs (£m)", "Departmental narrative on schedule",
        "Description / Aims"]]
    m = map_columns(headers, cfg=load_column_config())
    assert m["dca"] == "ipa delivery confidence assessment"
    assert m["dca_narrative"].startswith("departmental commentary")
    assert m["wlc_baseline"].startswith("total baseline")
    assert m["schedule_narrative"] == "departmental narrative on schedule"


def test_read_raw_handles_cp1252_and_transposed(tmp_path):
    p = tmp_path / "t.csv"
    text = "Project Name,Alpha,Beta\nDepartment,DfT,MoD\nDelivery Confidence Assessment,Red,Amber\nTotal Baseline Whole Life Costs (£m),£10m,20\n"
    p.write_bytes(text.encode("cp1252"))
    df = read_raw(p)
    assert list(df["project name"]) == ["Alpha", "Beta"]
    out, _rep = to_canonical(df, "2019-09", "2019-09-30")
    assert out["dca"].tolist() == ["Red", None] or out["dca"].tolist() == ["Red", "Amber"]
    assert out["wlc_baseline"].tolist() == [10.0, 20.0]
    assert out["dca_scale"].iat[0] == 5


def test_three_point_scale_after_2023(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("Project Name,Department,Delivery Confidence Assessment\nA,DfT,Amber\n")
    out, _ = to_canonical(read_raw(p), "2024-03", "2024-03-31")
    assert out["dca_scale"].iat[0] == 3 and out["rag3"].iat[0] == "Amber"

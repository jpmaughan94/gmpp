# Red, Amber, Green: what goes wrong with Britain's biggest projects?

A pipeline and analysis of the **Government Major Projects Portfolio (GMPP)**:
twelve annual transparency snapshots (Sep 2013 to Mar 2026) stitched into one
project-level panel, plus the April 2026 reset list.

## Questions
1. **What language comes before a slide to red?** NLP on the departmental narratives behind each delivery confidence assessment (DCA).
2. **Do cost and schedule slip after a red rating?** A quasi-experimental comparison of projects rated red with similar projects that weren't.
3. **What happened to the projects dropped in the April 2026 reset** (200+ → 81)?

## Pipeline
```
config/sources.yaml ──fetch──▶ data/raw/ (immutable, SHA-256 manifest)
                    ──clean──▶ canonical schema per snapshot
                    ──match──▶ project_uid across years ──▶ data/processed/panel.parquet
```

```bash
pip install -e ".[dev]"
python -m gmpp.fetch      # downloads from gov.uk (needs normal internet access)
python -m gmpp.build      # panel.parquet, links.csv, links_to_review.csv, build_report.json
pytest -q
```

## Data notes
- **Rating scale changed in 2023.** Up to Mar 2022 the DCA is five-point (Green, Amber/Green, Amber, Amber/Red, Red); from Mar 2023 it is three-point (Green, Amber, Red) with new definitions. The panel keeps the raw label and two definitions of red: `red_strict` (Red only) and `red_broad` (Red or Amber/Red). Results should hold under both.
- **Missing snapshots.** Sep 2015 was only published as ~20 departmental files (not yet included). There is no Sep 2020 snapshot: reporting moved to March.
- **Matching projects across years.** Exact IDs where available; otherwise a weighted score (normalised name fuzzy match, TF-IDF description similarity, same department *lineage*, e.g. BEIS → DESNZ/DSIT/DBT), then one-to-one Hungarian assignment. Projects that skip a year get a second pass. Borderline pairs go to `links_to_review.csv`; hand decisions live in `config/match_overrides.csv` and always win.
- **Costs** are parsed to £ millions, in nominal terms.

Source: NISTA / IPA / MPA annual reports on major projects, published under the Open Government Licence v3.0.

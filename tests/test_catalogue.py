"""
onFHIR-Feast catalogue resolution: newest-version selection, the exact
version pin, issued-date tie-breaks, and the step-by-step preflight
checklist on broken layouts.
"""

import json
import os

import pytest

from pipeline.catalogue import (
    CatalogueError,
    resolve_data_dir,
    select_entry,
)


def _entry(entry_id, version, issued, name="Study1", resource="study1-fs"):
    return {
        "id": entry_id,
        "name": name,
        "issued": issued,
        "population": {"url": f"https://datatools4heart.eu/cohorts/study1|{version}"},
        "featureSet": {
            "url": f"https://datatools4heart.eu/feature-sets/study1|{version}",
            "pipeline": {"reference": f"FeatureSet/{resource}/_history/1"},
        },
    }


CATALOGUE = {"entries": [
    _entry("run-old", "1.4", "2026-01-01T00:00:00Z"),
    _entry("run-21", "2.1", "2026-05-01T00:00:00Z"),
    _entry("run-22a", "2.2", "2026-08-01T00:00:00Z"),
    _entry("run-22b", "2.2", "2026-09-01T00:00:00Z"),  # rerun of 2.2
    _entry("other", "9.9", "2026-09-01T00:00:00Z", name="OtherStudy"),
]}


def test_newest_version_wins_and_issued_breaks_ties():
    e = select_entry(CATALOGUE, "Study1")
    assert e["id"] == "run-22b"  # 2.2 beats 2.1/1.4; newest issued of the 2.2 pair


def test_version_pin_selects_exactly_that_version():
    assert select_entry(CATALOGUE, "Study1", version="2.1")["id"] == "run-21"
    assert select_entry(CATALOGUE, "Study1", version="1.4")["id"] == "run-old"


def test_version_pin_miss_names_available_versions():
    with pytest.raises(CatalogueError, match="3.0"):
        select_entry(CATALOGUE, "Study1", version="3.0")


def test_other_studies_never_leak_into_selection():
    e = select_entry(CATALOGUE, "Study1")
    assert e["name"] == "Study1"


def _make_tree(tmp_path, entry):
    (tmp_path / "catalogue.json").write_text(json.dumps({"entries": [entry]}))
    data_dir = tmp_path / "study1-fs" / entry["id"]
    data_dir.mkdir(parents=True)
    (data_dir / "part-00000-abc.snappy.parquet").write_bytes(b"PARQUET-STAND-IN")
    return data_dir


def test_resolve_data_dir_happy_path(tmp_path):
    entry = _entry("run-22b", "2.2", "2026-09-01T00:00:00Z")
    data_dir = _make_tree(tmp_path, entry)
    r = resolve_data_dir(str(tmp_path), "Study1")
    assert r.data_dir == str(data_dir)
    assert all(s.passed for s in r.steps)


def test_resolve_data_dir_reports_missing_catalogue_without_raising(tmp_path):
    r = resolve_data_dir(str(tmp_path), "Study1")
    assert r.data_dir is None
    assert any(not s.passed and "catalogue.json" in s.name for s in r.steps)


def test_resolve_data_dir_reports_missing_part_files(tmp_path):
    entry = _entry("run-22b", "2.2", "2026-09-01T00:00:00Z")
    data_dir = _make_tree(tmp_path, entry)
    os.remove(os.path.join(data_dir, "part-00000-abc.snappy.parquet"))
    r = resolve_data_dir(str(tmp_path), "Study1")
    assert r.data_dir is None
    assert any(not s.passed and "part-" in s.name for s in r.steps)


def test_resolve_data_dir_without_root():
    r = resolve_data_dir(None)
    assert r.data_dir is None and not r.steps[0].passed

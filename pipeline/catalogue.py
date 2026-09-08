"""
Resolves the local part-*.parquet extract for a study from the
onFHIR-Feast catalogue.json, so an operator no longer has to know (or
be handed) the exact per-run "id" folder to point the pipeline at.

Expected layout under CDM_ROOT_PATH:

    <CDM_ROOT_PATH>/catalogue.json
    <CDM_ROOT_PATH>/<featureset-resource-name>/<id>/part-*.parquet

catalogue.json is onFHIR-Feast's per-run manifest: one entry per
(population, featureSet) run, named after the study (e.g. "Study1"),
each carrying a versioned `population.url` / `featureSet.url`
("...|MAJOR.MINOR") and its own `id`. Reruns and feature-set revisions
show up as SEPARATE entries sharing the same `name` -- this picks the
newest one for a given study by featureSet version (ties broken by
`issued`), not by the entry's `result` path, since that path is
wherever onFHIR-Feast happened to write it on the machine that produced
the catalogue and is not portable to this one.
"""

import glob
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime

CATALOGUE_FILENAME = "catalogue.json"
DEFAULT_STUDY_NAME = "Study1"

# Matches the "|MAJOR.MINOR[.PATCH...]" version suffix FHIR canonical
# URLs use, e.g. "https://datatools4heart.eu/feature-sets/study1|2.1".
_VERSION_RE = re.compile(r"\|(\d+(?:\.\d+)*)\s*$")

# "FeatureSet/<resource-name>/_history/<n>" -- <resource-name> is the
# on-disk folder under CDM_ROOT_PATH (e.g. "study1-fs").
_FEATURESET_REF_RE = re.compile(r"^FeatureSet/(.+)/_history/\d+$")


class CatalogueError(Exception):
    """Raised for a hard failure resolving the catalogue (bad path, bad
    JSON, no such study). Step-by-step failures during resolve_data_dir
    are reported via ResolveResult.steps instead, so callers can show a
    full checklist rather than stopping at the first problem."""


@dataclass
class CheckStep:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ResolveResult:
    steps: list = field(default_factory=list)
    data_dir: str | None = None
    entry: dict | None = None

    @property
    def ok(self) -> bool:
        return self.data_dir is not None

    def failure_summary(self) -> str:
        return "\n".join(f"  - {s.name}: {s.detail}" for s in self.steps if not s.passed)


def _parse_version(url: str | None) -> tuple:
    if not url:
        return ()
    m = _VERSION_RE.search(url)
    if not m:
        return ()
    return tuple(int(p) for p in m.group(1).split("."))


def _parse_issued(entry: dict) -> datetime:
    issued = entry.get("issued")
    if not issued:
        return datetime.min
    try:
        return datetime.fromisoformat(issued.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def featureset_resource_name(entry: dict) -> str:
    """The FeatureSet resource id from featureSet.pipeline.reference
    ("FeatureSet/<name>/_history/<n>") -- the on-disk folder name under
    CDM_ROOT_PATH. Derived from the entry rather than hardcoded so a
    differently-named study/feature-set still resolves correctly."""
    ref = entry.get("featureSet", {}).get("pipeline", {}).get("reference", "")
    m = _FEATURESET_REF_RE.match(ref)
    if not m:
        raise CatalogueError(
            f"Could not derive the FeatureSet resource name from featureSet.pipeline.reference "
            f"{ref!r} (expected 'FeatureSet/<name>/_history/<n>')."
        )
    return m.group(1)


def _parse_version_string(version: str) -> tuple:
    """Parses a user-supplied version pin ("1.4") the same way
    _parse_version parses the "|MAJOR.MINOR" URL suffix, for an exact
    tuple comparison -- so "1.4" matches featureSet url "...|1.4" and
    nothing else (not "1.40", not "1.4.0")."""
    try:
        return tuple(int(p) for p in version.strip().lstrip("|").split("."))
    except ValueError:
        raise CatalogueError(f"Invalid version {version!r} -- expected dot-separated integers, e.g. '1.4'.")


def select_entry(catalogue: dict, study_name: str, version: str | None = None) -> dict:
    """The catalogue entry named `study_name`.

    With `version` given (e.g. "1.4"), returns the entry whose
    featureSet version matches EXACTLY -- for deliberately pinning to a
    specific historical version rather than always the newest one (an
    older public_domains.json/release calibrated against it, reproducing
    a past run, etc.). `issued` breaks a tie if more than one entry
    happens to share that exact version. Raises naming every version
    actually available if none match.

    Without `version`, returns the newest entry, ranked by featureSet
    version first (the "|MAJOR.MINOR" suffix -- what actually changes
    between reruns, per the catalogue's own versioning) and `issued` as
    a tiebreak. Raises if `study_name` has no entries at all either way.
    """
    entries = [e for e in catalogue.get("entries", []) if e.get("name") == study_name]
    if not entries:
        available = sorted({e.get("name") for e in catalogue.get("entries", []) if e.get("name")})
        raise CatalogueError(f"No catalogue entry named {study_name!r} found. Available: {available}")

    if version is not None:
        wanted = _parse_version_string(version)
        matched = [e for e in entries
                   if _parse_version(e.get("featureSet", {}).get("url")) == wanted]
        if not matched:
            available_versions = sorted(
                ".".join(map(str, v)) for v in
                {_parse_version(e.get("featureSet", {}).get("url")) for e in entries} if v
            )
            raise CatalogueError(
                f"No {study_name!r} entry with featureSet version {version!r}. "
                f"Available version(s): {available_versions}"
            )
        entries = matched

    entries.sort(key=lambda e: (_parse_version(e.get("featureSet", {}).get("url")), _parse_issued(e)),
                 reverse=True)
    return entries[0]


def select_latest_entry(catalogue: dict, study_name: str) -> dict:
    """Back-compat alias for select_entry(catalogue, study_name) -- the
    newest entry, no version pin."""
    return select_entry(catalogue, study_name)


def resolve_data_dir(cdm_root_path: str | None, study_name: str = DEFAULT_STUDY_NAME,
                      study_version: str | None = None) -> ResolveResult:
    """Full resolution, step by step: CDM_ROOT_PATH -> catalogue.json ->
    the `study_name` entry (newest, or the exact `study_version` pin if
    given) -> its on-disk folder -> part-*.parquet files inside it.
    Never raises -- each stage is recorded as a CheckStep so a caller
    (preflight) can print the whole checklist even when an early stage
    fails; `result.data_dir` is set only once every stage has passed."""
    r = ResolveResult()

    if not cdm_root_path:
        r.steps.append(CheckStep("CDM_ROOT_PATH set", False, "not provided"))
        return r
    r.steps.append(CheckStep("CDM_ROOT_PATH set", True, cdm_root_path))

    if not os.path.isdir(cdm_root_path):
        r.steps.append(CheckStep("CDM_ROOT_PATH is a directory", False, cdm_root_path))
        return r

    catalogue_path = os.path.join(cdm_root_path, CATALOGUE_FILENAME)
    if not os.path.isfile(catalogue_path):
        r.steps.append(CheckStep(f"{CATALOGUE_FILENAME} found", False, catalogue_path))
        return r
    r.steps.append(CheckStep(f"{CATALOGUE_FILENAME} found", True, catalogue_path))

    try:
        with open(catalogue_path) as f:
            catalogue = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        r.steps.append(CheckStep(f"{CATALOGUE_FILENAME} parsed", False, f"{type(e).__name__}: {e}"))
        return r

    label = f"'{study_name}'" + (f" version {study_version!r}" if study_version else "")
    try:
        entry = select_entry(catalogue, study_name, version=study_version)
    except CatalogueError as e:
        r.steps.append(CheckStep(f"{label} entry found", False, str(e)))
        return r
    r.entry = entry
    fs_url = entry.get("featureSet", {}).get("url")
    pop_url = entry.get("population", {}).get("url")
    r.steps.append(CheckStep(
        f"{label} entry resolved" + ("" if study_version else " (newest)"), True,
        f"id={entry.get('id')} featureSet={fs_url} population={pop_url} issued={entry.get('issued')}",
    ))

    entry_id = entry.get("id")
    if not entry_id:
        r.steps.append(CheckStep("entry has an id", False, "missing 'id' field"))
        return r

    try:
        resource_name = featureset_resource_name(entry)
    except CatalogueError as e:
        r.steps.append(CheckStep("featureSet resource name derived", False, str(e)))
        return r

    data_dir = os.path.join(cdm_root_path, resource_name, entry_id)
    if not os.path.isdir(data_dir):
        r.steps.append(CheckStep(f"{resource_name}/{entry_id} directory exists", False, data_dir))
        return r
    r.steps.append(CheckStep(f"{resource_name}/{entry_id} directory exists", True, data_dir))

    part_files = sorted(glob.glob(os.path.join(data_dir, "part-*.parquet")))
    if not part_files:
        r.steps.append(CheckStep("part-*.parquet files present", False, f"none found in {data_dir}"))
        return r
    r.steps.append(CheckStep("part-*.parquet files present", True, f"{len(part_files)} file(s)"))

    r.data_dir = data_dir
    return r

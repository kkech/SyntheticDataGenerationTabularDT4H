"""
UC1 data pipeline entrypoint.

Runs load_data -> profile_data -> preprocess -> profile_preprocessed_data
-> generate -> evaluate -> utility -> privacy in order, skipping any step already marked completed
(tracked in pipeline_status.json) unless explicitly forced.

What the generate step runs is a config-driven RUN PLAN (seeds x
epsilons x models -- see PipelineConfig.resolved_run_plan), so changing
the experiment is a config change.

Usage:
    python main.py                          # run everything not yet done
    python main.py --force                  # rerun every step
    python main.py --force-step preprocess   # rerun just one step (repeatable)
    python main.py --only preprocess          # run just this step (repeatable;
                                               #   still respects its own completed
                                               #   status unless also forced)
    python main.py --status                 # print current step-completion status

All console output (including warnings and tracebacks, which shell
redirection alone would miss) is teed to logs.txt by default; override
with --log <path>.
"""

import argparse
import time
from datetime import datetime

from pipeline.config import PipelineConfig
from pipeline.logging_setup import start_logging, stop_logging
from pipeline.state import PipelineState, plan_cascade_invalidations
from pipeline.steps.load_data import LoadDataStep
from pipeline.steps.profile_data import ProfileDataStep
from pipeline.steps.preprocess import PreprocessStep
from pipeline.steps.profile_preprocessed_data import ProfilePreprocessedDataStep
from pipeline.steps.generate import GenerateStep
from pipeline.steps.evaluate import EvaluateStep
from pipeline.steps.coherence import CoherenceStep
from pipeline.steps.survival import SurvivalStep
from pipeline.steps.privacy import PrivacyStep
from pipeline.steps.utility import UtilityStep
from pipeline.steps.attacks import AttacksStep
from pipeline.steps.figures import FiguresStep
from pipeline.steps.release_docs import ReleaseDocsStep

STEPS = [
    LoadDataStep(),
    ProfileDataStep(),
    PreprocessStep(),
    ProfilePreprocessedDataStep(),
    GenerateStep(),
    EvaluateStep(),
    CoherenceStep(),
    SurvivalStep(),
    UtilityStep(),
    PrivacyStep(),
    AttacksStep(),
    FiguresStep(),
    ReleaseDocsStep(),
]

# All steps after 'generate' are ANALYSIS steps: they read the generated
# CSVs and the train/holdout split but never regenerate anything -- so
# they can be rerun cheaply on existing outputs.
ANALYSIS_STEPS = ["evaluate", "coherence", "survival", "utility", "privacy",
                  "attacks", "figures", "release_docs"]


def run_pipeline(
    config: PipelineConfig | None = None,
    force: bool = False,
    force_steps: list[str] | None = None,
    only: list[str] | None = None,
    no_cascade: bool = False,
    retry_failed_steps: list[str] | None = None,
) -> None:
    config = config or PipelineConfig()
    state = PipelineState(config.status_path)
    force_steps = set(force_steps or [])
    # Distinct from force_steps: queues an already-completed step to run
    # again WITHOUT wiping its output directory first, for a resumable
    # step (GenerateStep) whose own reconciliation only re-executes the
    # runs that did NOT succeed last time (a bad public domain entry
    # aside, a per-run failure is usually transient -- CUDA OOM under too
    # much --parallel contention, a flaky node) rather than the whole
    # multi-hour plan. --force-step generate would also rerun it, but
    # wipes everything first and redoes every run, successes included.
    retry_failed_steps = set(retry_failed_steps or [])

    steps = [s for s in STEPS if only is None or s.name in only]
    if not steps:
        raise ValueError(f"No matching step(s) for --only {only}. Known steps: {[s.name for s in STEPS]}")

    import os
    import shutil

    # STALE-MIX GUARD. Rerunning an upstream step (forced or simply not
    # completed yet) makes every completed step AFTER it stale: its outputs
    # were computed from the OLD upstream data, and a later analysis-only
    # run would evaluate old synthetic CSVs against a new split/encoding
    # with nothing detecting the mix. So every completed step later in
    # pipeline order than the earliest step queued to run is invalidated
    # (marked pending) up front. Steps in the current selection then rerun
    # now; steps outside it (e.g. under --only) rerun on the next full run.
    # A retry-failed-steps rerun belongs in this too: it can add newly-
    # successful runs to generate's output, which analysis steps after it
    # have not seen yet.
    queued = {s.name for s in steps
              if force or s.name in force_steps or s.name in retry_failed_steps
              or not state.is_completed(s.name)}
    step_names = [s.name for s in STEPS]
    stale = plan_cascade_invalidations(step_names, queued, state.is_completed)
    if stale and not no_cascade:
        earliest = min(queued, key=step_names.index)
        print(f"🔁 Cascade: '{earliest}' will rerun, so the completed outputs of every "
              f"later step were built from data this run is about to replace.")
        for name in stale:
            state.mark_pending(name, note=f"invalidated: upstream step '{earliest}' reran")
            print(f"   invalidated '{name}' (marked pending -- its outputs would mix "
                  f"old results with the new '{earliest}' outputs)")
        print(f"   Pass --no-cascade to keep downstream steps marked completed instead.")
    elif stale and no_cascade:
        print(f"⚠️  --no-cascade: leaving {len(stale)} completed downstream step(s) as-is "
              f"({', '.join(stale)}). Their outputs may mix generations (old results vs "
              f"the upstream data this run regenerates) -- verifying consistency is on you.")

    to_run = []
    for step in steps:
        should_force = force or step.name in force_steps
        if state.is_completed(step.name) and not should_force and step.name not in retry_failed_steps:
            print(f"⏭️  Skipping '{step.name}' (already completed). "
                  f"Use --force or --force-step {step.name} to rerun.")
            continue
        to_run.append(step)

    # Mark every step this run WILL execute as pending up front, so a
    # status check mid-run never shows a stale 'completed' from a
    # previous run for a step that is queued to be redone.
    for step in to_run:
        state.mark_pending(step.name)

    for step in to_run:
        should_force = force or step.name in force_steps
        step_out = config.step_dir(step.name)
        # A rerun replaces the step's outputs wholesale: delete the old
        # ones first so nothing stale can survive next to fresh files --
        # UNLESS the step opted into resuming (GenerateStep) and this is
        # not an explicit --force/--force-step: it was merely never
        # marked completed (interrupted, crashed, rebooted mid-run), and
        # the step's own run() reconciles with what's already there
        # instead of redoing a multi-hour campaign from scratch.
        if step.resumable and not should_force:
            if os.path.isdir(step_out) and step.name in retry_failed_steps:
                print(f"↩️  Retrying only the failed run(s) in '{step.name}' -- "
                      f"{step_out} is kept, not wiped; already-successful runs are reused.")
            elif os.path.isdir(step_out):
                print(f"↩️  '{step.name}' was not marked completed (interrupted?) -- "
                      f"resuming: {step_out} is kept, not wiped.")
        elif os.path.isdir(step_out):
            if step.name == "load_data" and not os.path.isdir(config.transfer_folder):
                raise FileNotFoundError(
                    f"Refusing to delete {step_out} before rerunning load_data: the transfer "
                    f"folder {config.transfer_folder} is not available to rebuild it from."
                )
            shutil.rmtree(step_out)
            print(f"🧹 Deleted previous outputs of '{step.name}' ({step_out}) for a clean rerun.")

        state.mark_running(step.name)
        started = time.time()
        print(f"\n{'=' * 70}\n▶️  RUNNING '{step.name}' (started {datetime.now().strftime('%H:%M:%S')})\n{'=' * 70}")
        try:
            step.run(config)
        except BaseException as e:
            # BaseException so a Ctrl-C or SIGTERM mid-step is recorded
            # too, instead of leaving the status file claiming 'running'
            # while the step actually died half-way.
            state.mark_failed(step.name, f"{type(e).__name__}: {e}")
            print(f"❌ Step '{step.name}' "
                  f"{'interrupted' if isinstance(e, (KeyboardInterrupt, SystemExit)) else 'failed'} "
                  f"after {time.time() - started:.0f}s: {e}")
            raise
        state.mark_completed(step.name)
        print(f"✅ '{step.name}' completed in {time.time() - started:.0f}s.")


def preflight(config: PipelineConfig | None = None, min_free_gb: float = 5.0) -> bool:
    """Everything a long run needs, checked in seconds. Returns True if
    the run can proceed."""
    import importlib
    import os
    import shutil

    config = config or PipelineConfig()
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        mark = "✅" if passed else "❌"
        print(f"  {mark} {name}" + (f" -- {detail}" if detail else ""))
        ok = ok and passed

    # Computed early so the mbi check below (and every check further down
    # that already relies on it) can ask "does the run actually need this"
    # rather than checking every dependency unconditionally -- a machine
    # excluding aim/mst entirely (main.py --synthesizers/--dp-only) should
    # not fail preflight over a backend it will never touch.
    plan = config.resolved_run_plan()

    print("Preflight checks:")
    for mod in ("polars", "pandas", "numpy", "scipy", "sdv", "snsynth", "torch",
                "cloudpickle", "sklearn"):
        try:
            m = importlib.import_module(mod)
            check(f"import {mod}", True, getattr(m, "__version__", ""))
        except Exception as e:
            check(f"import {mod}", False, f"{type(e).__name__}: {e}")

    # MST/AIM generation imports mbi -> jax, and jax requires numpy>=2;
    # anonymeter's install can silently downgrade numpy to 1.26 and this
    # is the check that catches it BEFORE a multi-day campaign rather
    # than 10 hours in (see requirements.txt, numpy note).
    needs_mbi = any(s["synthesizer"] in ("aim", "mst") for s in plan)
    if needs_mbi:
        try:
            importlib.import_module("mbi")
            check("import mbi (MST/AIM backend)", True)
        except Exception as e:
            check("import mbi (MST/AIM backend)", False,
                  f"{type(e).__name__}: {e} -- every MST/AIM run would fail; if this is "
                  f"a numpy<2 downgrade (e.g. from installing anonymeter), fix: "
                  f"`pip install numpy==2.2.6` (see requirements.txt) -- otherwise this "
                  f"is unrelated to numpy (read the exception above).")
    else:
        print("  ⏭️  import mbi (MST/AIM backend) -- skipped, no aim/mst run in the "
              "resolved plan (--synthesizers/--dp-only excluded them)")

    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            check("CUDA GPU", True, f"{torch.cuda.get_device_name(0)}, {free/1e9:.1f} GB free of {total/1e9:.1f} GB")
        else:
            # Distinguish "this machine has no GPU" from "the machine has a
            # GPU but the installed torch wheel targets a newer CUDA than
            # the driver supports" -- the wheel/driver mismatch otherwise
            # reads as missing hardware and sends a partner site down the
            # wrong path (found at a site with 8 idle GPUs and a torch
            # wheel one CUDA generation ahead of the driver).
            driver = None
            try:
                import subprocess
                smi = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10)
                if smi.returncode == 0 and smi.stdout.strip():
                    driver = smi.stdout.strip().splitlines()[0]
            except Exception:
                pass
            wheel_cuda = getattr(torch.version, "cuda", None)
            if driver:
                check("CUDA GPU", False,
                      f"nvidia-smi sees a GPU (driver {driver}) but "
                      f"torch {torch.__version__} (built for CUDA {wheel_cuda or 'CPU-only'}) "
                      f"cannot use it -- the installed wheel does not match the driver. "
                      f"Fix: install a torch build for your driver's CUDA generation, "
                      f"e.g. `pip install 'torch==2.5.*'` (CUDA 12.x wheels run on "
                      f"drivers >= 525). CPU training works but is many times slower "
                      f"and the README runtimes will not hold.")
            else:
                print("  ⚠️  no CUDA GPU -- ctgan/tvae/dpctgan will train on CPU (much slower); "
                      "gaussian_copula/mst/aim unaffected")
    except Exception:
        pass

    have_transfer = os.path.isdir(config.transfer_folder)
    have_loaded = os.path.exists(config.local_full_dataset_path)
    check("input data", have_transfer or have_loaded,
          config.transfer_folder if have_transfer else config.local_full_dataset_path)
    check("metadata.json", os.path.exists(config.metadata_path) or have_transfer, config.metadata_path)

    free_gb = shutil.disk_usage(os.path.dirname(config.output_dir) or ".").free / 1e9
    check(f"disk space >= {min_free_gb:g} GB", free_gb >= min_free_gb,
          f"{free_gb:.1f} GB free")

    from pipeline.steps.generate.synthesizers import REGISTRY

    unknown = sorted({s["synthesizer"] for s in plan} - set(REGISTRY))
    check("synthesizers registered", not unknown,
          ", ".join(sorted({s["synthesizer"] for s in plan})))

    # DP runs are bounded by the reviewed public domain declaration; a
    # missing or unreviewed file would abort the generate step anyway,
    # but a long run should be refused HERE, before it detaches.
    dp_in_plan = any(s.get("epsilon") is not None for s in plan)
    if dp_in_plan:
        domains = None
        try:
            from pipeline.steps.generate.synthesizers.smartnoise_models import load_public_domains

            domains, sha = load_public_domains(config.public_domains_path)
            check("public domains reviewed (DP runs)", True,
                  f"{len(domains)} ranges, sha256 {sha[:12]}...")
        except Exception as e:
            check("public domains reviewed (DP runs)", False,
                  f"{e} -- run `python make_public_domains.py`, review every range, "
                  f"set \"reviewed\": true")

        # Report BEFORE any run starts how many datapoints actually sit
        # outside their declared domain -- whether that's zero, a handful
        # of documented entry artifacts, or enough to suggest something
        # upstream is actually broken (this number is the difference
        # between "fine to clip" and "go investigate the data first").
        # A separate try/except from the check above: a failure here must
        # never read back as "public domains reviewed: False" when the
        # domains loaded just fine.
        if domains is not None and os.path.exists(config.train_output_path):
            try:
                import json

                import pandas as pd

                from pipeline.steps.generate.synthesizers.smartnoise_models import (
                    coarse_observed_span,
                    compute_domain_report,
                )
                from pipeline.steps.preprocess.transforms import NUMERIC_ENCODING_FILENAME

                train_df = pd.read_parquet(config.train_output_path)
                continuous = [c for c in train_df.columns if pd.api.types.is_numeric_dtype(train_df[c])]
                enc_path = os.path.join(config.step_dir("preprocess"), NUMERIC_ENCODING_FILENAME)
                encoding = {}
                if os.path.exists(enc_path):
                    with open(enc_path) as f:
                        encoding = json.load(f)
                report = compute_domain_report(train_df, continuous, domains, encoding)
                violating = {c: r for c, r in report["columns"].items() if r["violates"]}
                if violating:
                    total = sum(r["n_below"] + r["n_above"] for r in violating.values())
                    print(f"  ⚠️  {len(violating)} continuous column(s), {total} datapoint(s) total, "
                          f"fall outside their declared public domain -- these will FAIL the run "
                          f"unless --clip-to-domain is set (which clips them to the declared "
                          f"bound, not to anything derived from the data):")
                    for c, r in sorted(violating.items(), key=lambda kv: -(kv[1]["n_below"] + kv[1]["n_above"])):
                        n = r["n_below"] + r["n_above"]
                        # The sentinel tag is the key diagnostic: a column
                        # flagged as NOT sentinel-encoded whose observed
                        # minimum sits near -(0.25 x range) below zero is a
                        # train parquet carrying sentinels the CURRENT
                        # encoding map does not know about -- i.e. stale,
                        # mixed preprocess artifacts, not bad data.
                        tag = ("sentinel-encoded per current map" if c in encoding
                               else "NOT sentinel-encoded per current map")
                        print(f"      {c}: {n}/{r['n_total']} datapoint(s) outside "
                              f"[{r['lower']:g}, {r['pub_hi']:g}] "
                              f"(observed {coarse_observed_span(r)}, coarsened; {tag})")
                else:
                    print("  ✅ every continuous column's training values fall within its declared public domain")
            except Exception as e:
                print(f"  ⚠️  Could not check training values against the public domain: "
                      f"{type(e).__name__}: {e}")

    # Rough per-run durations measured on this project's own full-scale
    # runs (T4, 211 columns), for a total-duration expectation only.
    est_minutes = {"gaussian_copula": 1, "tvae": 7, "ctgan": 14,
                   "dpctgan": 42, "mst": 170, "aim": 60}
    total_min = sum(est_minutes.get(s["synthesizer"], 60) for s in plan)
    by_model: dict[str, int] = {}
    for s in plan:
        by_model[s["synthesizer"]] = by_model.get(s["synthesizer"], 0) + 1
    print(f"\n  Run plan: {len(plan)} run(s) -- "
          + ", ".join(f"{m} x{n}" for m, n in by_model.items()))
    print(f"  DP epsilon sweep: {', '.join(f'{e:g}' for e in config.dp_epsilons)} | "
          f"seeds {config.variance_seeds} | AIM on top {config.aim_max_columns} columns")
    print(f"  Rough duration estimate: ~{total_min/60:.0f}h "
          f"(measured per-model costs; AIM is the unknown, bounded at "
          f"{config.aim_timeout_seconds/3600:.0f}h/run)")
    print(f"  Default per-run timeout: {config.synthesizer_timeout_seconds/3600:.1f}h | "
          f"base seed {config.seed} | holdout fraction {config.holdout_fraction:.0%}")
    print(f"  Ordering is cheapest/most-reliable first, so a late timeout costs only the tail of the run.")
    print("\n" + ("Preflight PASSED -- ready for a long run." if ok else "Preflight FAILED -- fix the items above first."))
    return ok


def print_status(config: PipelineConfig | None = None) -> None:
    config = config or PipelineConfig()
    state = PipelineState(config.status_path)
    summary = state.summary()
    print(f"Pipeline status ({config.status_path}):")
    for step in STEPS:
        info = summary.get(step.name)
        if info is None:
            print(f"  {step.name}: never run")
        elif info.get("completed"):
            print(f"  {step.name}: ✅ completed at {info.get('completed_at')}")
        elif info.get("running"):
            print(f"  {step.name}: 🔄 running (started {info.get('started_at')})")
        elif info.get("pending"):
            print(f"  {step.name}: ⏳ pending (queued in the current run)")
        else:
            print(f"  {step.name}: ❌ failed at {info.get('failed_at')} -- {info.get('error')}")


def write_deployment_manifest(config: PipelineConfig, model: str) -> str:
    """After a --model run, write ONE stable, fixed-name JSON file naming
    exactly where the synthetic CSV and every analysis report ended up --
    so a downstream program consuming this deployment's output only ever
    needs to know this single file's name, never this repo's internal
    run-id/step-directory naming conventions.

    Every report path is checked for actual existence (None if absent --
    e.g. a step skipped via --only) rather than assumed, since '--model'
    does not restrict which steps run, only what generate's plan
    contains. Reads the real generation summary rather than assuming
    success, so the manifest is honest even if called after a failure."""
    import json as _json
    import os
    from datetime import timezone

    def _existing(path: str) -> str | None:
        return path if os.path.exists(path) else None

    run_record = None
    summary_path = os.path.join(config.step_dir("generate"), "DT4H_Generation_Summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            gen_summary = _json.load(f)
        matches = [r for r in gen_summary.get("runs", [])
                  if r.get("synthesizer") == model or r.get("base_synthesizer") == model]
        run_record = matches[0] if matches else None

    reports = {}
    for step_name, filename in (("evaluate", "DT4H_Evaluation"), ("coherence", "DT4H_Coherence_Audit"),
                                ("survival", "DT4H_Survival_Fidelity"), ("utility", "DT4H_Utility_TSTR"),
                                ("privacy", "DT4H_Privacy_Assessment"), ("attacks", "DT4H_Privacy_Attacks")):
        step_dir = config.step_dir(step_name)
        reports[step_name] = {
            "md": _existing(os.path.join(step_dir, f"{filename}.md")),
            "json": _existing(os.path.join(step_dir, f"{filename}.json")),
        }
    reports["figures_dir"] = _existing(config.step_dir("figures"))
    release_dir = config.step_dir("release_docs")
    reports["datasheet"] = _existing(os.path.join(release_dir, "DT4H_Datasheet.md"))
    reports["codebook"] = _existing(os.path.join(release_dir, "DT4H_Codebook.md"))
    if run_record and run_record.get("run_id"):
        reports["release_label"] = _existing(
            os.path.join(release_dir, "labels", f"DT4H_Label_{run_record['run_id']}.json"))

    manifest = {
        "model": model,
        "run_id": run_record.get("run_id") if run_record else None,
        "status": run_record.get("status") if run_record else None,
        "epsilon": run_record.get("epsilon") if run_record else None,
        "seed": run_record.get("seed") if run_record else None,
        "synthetic_csv": run_record.get("output_path") if run_record else None,
        "output_dir": os.path.abspath(config.output_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports": reports,
    }
    manifest_path = os.path.join(config.output_dir, "DT4H_Deployment_Manifest.json")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        _json.dump(manifest, f, indent=2)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="UC1 data preparation pipeline.")
    parser.add_argument("--force", action="store_true", help="Rerun every step, even if already completed.")
    parser.add_argument("--force-step", action="append", default=[],
                         help="Rerun this step even if already completed (repeatable).")
    parser.add_argument("--retry-failed-step", action="append", default=[],
                         help="Rerun this ALREADY-COMPLETED step, but (for a resumable step -- "
                              "currently just 'generate') WITHOUT wiping its output directory "
                              "first: already-successful runs are reused as-is and only the "
                              "runs that failed last time are re-executed. For a batch where "
                              "some runs failed for a transient reason (CUDA OOM under too much "
                              "--parallel contention, a flaky node) rather than every run being "
                              "wrong. --force-step generate also reruns it, but wipes everything "
                              "and redoes every run, successes included -- use that instead when "
                              "you actually want a clean restart. Repeatable, though only "
                              "'generate' does anything special with it today.")
    parser.add_argument("--only", action="append", default=None,
                         help="Run only these step(s) (repeatable).")
    parser.add_argument("--no-cascade", action="store_true",
                         help="Do NOT invalidate completed downstream steps when an upstream "
                              "step reruns. WARNING: this can leave analysis outputs computed "
                              "from a previous generation of the data sitting next to fresh "
                              "upstream outputs -- mixing generations is then on you.")
    parser.add_argument("--analysis", action="store_true",
                         help="Rerun ALL analysis steps (evaluate through release_docs) over the "
                              "existing generated outputs. No regeneration -- cheap and safe.")
    parser.add_argument("--extended", action="store_true",
                         help="Append the roadmap runs to the generation plan (quantile-"
                              "transform variants, TVAE capacity sweep, indicator-encoding "
                              "ablation, AIM 40-column sweep, MST epsilon=0.5 anchor, the "
                              "native diffusion baseline ddpm x3 seeds, PATE-CTGAN x3 "
                              "epsilons). The base plan is unchanged; see "
                              "PipelineConfig.extended_plan.")
    parser.add_argument("--data-dir", metavar="PATH",
                         help="Directory holding the input part-*.parquet files (and, "
                              "unless --metadata is given, the metadata file). Overrides "
                              "the configured transfer folder -- this is how the pipeline "
                              "points at a new site's extract.")
    parser.add_argument("--metadata", metavar="PATH",
                         help="Explicit path to the feature-set metadata JSON, for when "
                              "it does not live inside --data-dir. Copied to "
                              "output/profile_data/metadata.json for the downstream steps.")
    synth_filter = parser.add_mutually_exclusive_group()
    synth_filter.add_argument("--synthesizers", metavar="NAME[,NAME...]",
                         help="Restrict the generate step's run plan to only these synthesizer "
                              "families (comma-separated, e.g. 'dpctgan,ctgan,tvae,gaussian_copula') "
                              "-- the SAME plan resolved_run_plan() would build (respects "
                              "--extended), just filtered down to the names given. Useful for "
                              "excluding a backend this machine can't run (e.g. aim/mst when "
                              "mbi/jax won't import) or testing one model family in isolation. "
                              "NOTE: --force-step generate (or --force) still deletes ALL of "
                              "output/generate/ first -- this only controls what gets "
                              "regenerated afterward, not what survives from a prior run. "
                              "Mutually exclusive with --dp-only.")
    synth_filter.add_argument("--dp-only", action="store_true",
                         help="Shortcut for --synthesizers <every DP-registered family present "
                              "in the resolved plan> -- derived from each synthesizer's is_dp "
                              "flag in the registry (pipeline/steps/generate/synthesizers), not "
                              "a hardcoded list. Mutually exclusive with --synthesizers.")
    synth_filter.add_argument("--model", choices=["mst", "tvae"],
                         help="Deployment shortcut: run the FULL pipeline (every step) with the "
                              "generate plan replaced by a single run of this one model -- "
                              "'mst' at epsilon=15 (the paper's DP operating point), or plain "
                              "'tvae' -- instead of the full multi-model/multi-epsilon sweep. "
                              "For running one model at a time, in separate invocations, without "
                              "needing to know this repo's run-plan/epsilon vocabulary. Writes "
                              "DT4H_Deployment_Manifest.json in --output-dir naming exactly where "
                              "the synthetic CSV and every report ended up. Pair with "
                              "--output-dir so each invocation's output is self-contained. "
                              "Mutually exclusive with --synthesizers/--dp-only.")
    parser.add_argument("--output-dir", metavar="PATH",
                         help="Root directory for every step's output (default: ./output). Also "
                              "redirects pipeline_status.json into it (normally at the repo "
                              "root), so a run against one --output-dir never thinks a step is "
                              "already done because SOME OTHER --output-dir's run marked it so -- "
                              "each output directory is a fully self-contained, independently "
                              "resumable pipeline run. Set this to give each deployment (site, "
                              "model, environment) its own output location.")
    parser.add_argument("--status", action="store_true", help="Print step-completion status and exit.")
    parser.add_argument("--min-free-gb", type=float, default=None,
                         help="Override the preflight free-disk requirement (GB). The v3 "
                              "campaign needs ~2 GB (slim backup + DP re-fit outputs), not "
                              "the full-campaign 5 GB.")
    parser.add_argument("--clip-to-domain", action="store_true",
                        help="Instead of failing when a continuous column's training "
                             "values fall outside its declared public domain, clip those "
                             "cells TO THE DECLARED BOUND (never to anything derived from "
                             "the data) and record the clip in the run's provenance. For "
                             "documented entry artifacts; run --preflight first to see "
                             "how many cells are affected.")
    parser.add_argument("--preflight", action="store_true",
                         help="Verify libraries, GPU, inputs, disk and config, then exit. "
                              "Run this before a long run.")
    parser.add_argument("--log", default="logs.txt",
                         help="File to tee all console output (stdout, stderr and warnings) to. "
                              "Default: logs.txt. Pass --log '' to disable.")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.analysis:
        args.only = list(ANALYSIS_STEPS)
        args.force = True

    import os

    # One config from the flags, used identically by preflight and the
    # run, so the pre-launch check always previews what would execute.
    cfg_kwargs = {}
    if args.extended:
        cfg_kwargs["extended_plan"] = True
    if args.clip_to_domain:
        cfg_kwargs["clip_to_domain"] = True
    if args.data_dir:
        cfg_kwargs["transfer_folder"] = args.data_dir
    if args.metadata:
        cfg_kwargs["metadata_source"] = args.metadata
    if args.output_dir:
        cfg_kwargs["output_dir"] = args.output_dir
        # status_path does NOT derive from output_dir on its own (unlike
        # metadata_path/train_output_path/etc in __post_init__) -- it
        # defaults to a fixed path at the repo root, shared by every
        # invocation from this checkout regardless of --output-dir. Two
        # --model runs pointed at two different --output-dir values would
        # otherwise share ONE status file: the second run would see
        # 'generate' already completed (by the first model) and skip it
        # entirely. Deriving it here instead makes each --output-dir a
        # fully independent, self-contained pipeline run.
        cfg_kwargs["status_path"] = os.path.join(args.output_dir, "pipeline_status.json")
    cfg = PipelineConfig(**cfg_kwargs) if cfg_kwargs else None

    if args.model:
        cfg = cfg or PipelineConfig()
        # One explicit run, in the exact dict shape resolved_run_plan()
        # itself produces (run_id, synthesizer, seed, epsilon, columns,
        # timeout_seconds) -- PipelineConfig.run_plan is the documented
        # override point ("Set run_plan explicitly ... to override the
        # generated plan entirely").
        if args.model == "mst":
            cfg.run_plan = ({"run_id": "mst_eps15_seed0", "synthesizer": "mst", "seed": 0,
                             "epsilon": 15.0, "columns": None, "timeout_seconds": None},)
        else:
            cfg.run_plan = ({"run_id": "tvae_seed0", "synthesizer": "tvae", "seed": 0,
                             "epsilon": None, "columns": None, "timeout_seconds": None},)
        print(f"--model {args.model}: generate plan replaced with a single run "
              f"({cfg.run_plan[0]['run_id']}).")

    if args.synthesizers or args.dp_only:
        cfg = cfg or PipelineConfig()
        plan = cfg.resolved_run_plan()
        if args.dp_only:
            from pipeline.steps.generate.step import GenerateStep

            wanted = {name for name in {s["synthesizer"] for s in plan} if GenerateStep._is_dp(name)}
        else:
            wanted = {n.strip() for n in args.synthesizers.split(",") if n.strip()}
        filtered = [s for s in plan if s["synthesizer"] in wanted]
        if not filtered:
            parser.error(
                f"--synthesizers/--dp-only matched no run in the resolved plan "
                f"(wanted: {sorted(wanted)}). Available synthesizer families: "
                f"{sorted({s['synthesizer'] for s in plan})}"
            )
        cfg.run_plan = filtered
        print(f"Filtered run plan to {len(filtered)}/{len(plan)} run(s) "
              f"(synthesizers: {sorted(wanted)}): " + ", ".join(s["run_id"] for s in filtered))

    # Analysis-only runs write reports and figures (tens of MB), not
    # models and datasets -- a full campaign's 5 GB headroom would block
    # them pointlessly on a tight disk.
    analysis_only = bool(args.only) and set(args.only) <= set(ANALYSIS_STEPS)
    min_free = args.min_free_gb if args.min_free_gb is not None else (1.0 if analysis_only else 5.0)

    if args.preflight:
        raise SystemExit(0 if preflight(cfg, min_free_gb=min_free) else 1)

    # ./run_job.sh stop (and plain `kill`) send SIGTERM, which by default
    # ends the process without unwinding Python -- leaving the status
    # file claiming a step is still 'running'. Raise instead, so the
    # normal failure path records the interrupted step and the log gets
    # its closing line.
    import signal

    def _on_sigterm(signum, frame):
        raise SystemExit("terminated (SIGTERM)")

    signal.signal(signal.SIGTERM, _on_sigterm)

    # Tee everything to a log file so a failing run can be shared whole,
    # rather than only the stdout half that shell redirection captures.
    handle = start_logging(args.log) if args.log else None
    try:
        run_pipeline(config=cfg, force=args.force, force_steps=args.force_step,
                     only=args.only, no_cascade=args.no_cascade,
                     retry_failed_steps=args.retry_failed_step)
        if args.model:
            manifest_path = write_deployment_manifest(cfg, args.model)
            print(f"\n📦 Deployment manifest written -> {manifest_path}")
    finally:
        if handle:
            stop_logging(handle)
            print(f"\nFull log written to {args.log}")


if __name__ == "__main__":
    main()

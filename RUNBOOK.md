# Runbook: MST (ε=15) and TVAE deployment

Two decided models, run one at a time, in separate invocations, each
producing a self-contained output package (synthetic CSV + every
evaluation report) at a location you choose. Written for an operator
running this at a partner site who is not expected to know this
repo's run-plan/epsilon vocabulary.

Branch: `feature/synthesizer-filter` (`--model`/`--output-dir` are not
yet on `main`).

## 1. One-time setup, per machine

```bash
git clone <repo-url>
cd SyntheticDataGenerationTabularDT4H
git checkout feature/synthesizer-filter

python -m venv .synthenv && source .synthenv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install lifelines anonymeter    # optional evaluators
pip install numpy==2.2.6            # AFTER anonymeter -- restores numpy 2 (see requirements.txt)
```

**Review the public domain file — required for MST, not for TVAE.**
MST is differentially private and refuses to fit until a human has
signed off on `public_domains.json`: open it, read every numeric
range against your own clinical knowledge, then set `"reviewed": true`
(and record `reviewed_by`/`reviewed_at`). It ships `reviewed: false` on
purpose — another site's sign-off does not transfer. TVAE carries no
DP guarantee and does not read this file at all; you can run
`--model tvae` before this review is done.

## 2. Before every run: preflight

```bash
python main.py --data-dir /path/to/your/part-parquet-folder --preflight
```

Fix anything it reports before starting a real run. One thing specific
to this deployment: the `import mbi (MST/AIM backend)` check only runs
if the resolved plan actually contains `mst`/`aim` — so it fires for
`--model mst` but is silently skipped for `--model tvae`. If `mbi`/`jax`
can't import on a given machine (a known failure: jaxlib built for a
CPU instruction set — AVX — the machine doesn't have), **that machine
can still run `--model tvae`, just not `--model mst`.**

## 3. Run one model, into its own output directory

```bash
# MST, epsilon=15
python main.py --data-dir /path/to/your/part-parquet-folder \
  --model mst --output-dir /path/to/mst_results

# TVAE, run separately (now, later, on a different day -- doesn't matter)
python main.py --data-dir /path/to/your/part-parquet-folder \
  --model tvae --output-dir /path/to/tvae_results
```

Add `--metadata /path/to/metadata.json` if it does not live inside
`--data-dir`.

**Always give each invocation its own `--output-dir`.** Every step's
output (including the internal `pipeline_status.json` completion
tracker) lives inside it, so two different `--output-dir` values are
two fully independent runs — running MST does not affect, skip, or
interfere with TVAE's run in any way, and vice versa.

For a long run that must survive a disconnect, use the existing job
runner with the same flags:

```bash
./run_job.sh start --model mst --output-dir /path/to/mst_results --data-dir /path/to/your/part-parquet-folder
./run_job.sh status      # any time
./run_job.sh follow      # live log (Ctrl-C detaches; job keeps running)
```

## 4. What you get

Everything lands under `--output-dir`, one subfolder per pipeline
step (`generate/`, `evaluate/`, `coherence/`, `survival/`, `utility/`,
`privacy/`, `attacks/`, `figures/`, `release_docs/`). You should not
need to know that layout — read one file instead:

```
<output-dir>/DT4H_Deployment_Manifest.json
```

```json
{
  "model": "mst",
  "run_id": "mst_eps15_seed0",
  "status": "ok",
  "epsilon": 15.0,
  "seed": 0,
  "synthetic_csv": "<output-dir>/generate/DT4H_Synthetic_mst_eps15_seed0.csv",
  "output_dir": "<output-dir>",
  "generated_at": "2026-...Z",
  "reports": {
    "evaluate":  {"md": "...", "json": "..."},
    "coherence": {"md": "...", "json": "..."},
    "survival":  {"md": "...", "json": "..."},
    "utility":   {"md": "...", "json": "..."},
    "privacy":   {"md": "...", "json": "..."},
    "attacks":   {"md": "...", "json": "..."},
    "figures_dir": "...",
    "datasheet": "...",
    "codebook": "...",
    "release_label": "..."
  }
}
```

Every path in it is checked to actually exist before being written —
a downstream program should treat a `null` there as "that report was
not produced this run" (e.g. a step skipped with `--only`), not as a
bug. `synthetic_csv` is the file to hand to anything consuming the
generated data; the `reports` block is everything needed to judge
whether that CSV is fit to use.

# Autoresearch Program: GMM-TIDE Config Search

This repo adapts Karpathy's autoresearch loop to configuration search, not broad code mutation. The agent should improve GMM-TIDE experiment configs by reading existing reports, proposing a small next grid, optionally dry-running the Kaggle staging path, then waiting for human approval before any real Kaggle push.

## Objective

Find CelebA-HQ GMM-TIDE configs that lower `fid128_best`. Use these tie-breakers in order: `fid32_best`, `fid4_best`, `valid_loss`, and lower operational risk.

## Editable Scope

Default editable files:

- `configs/autoresearch/*.json`
- `reports/autoresearch_*.json`
- `reports/autoresearch_*.md`
- `scripts/autoresearch_config_search.py` when the harness itself needs improvement
- `README.md` when commands or behavior change

Do not edit `train.py`, model code, notebooks, or Kaggle credential handling unless the human explicitly changes the research scope from config search to code search.

## Loop

1. Rank current evidence:

   ```bash
   ./.venv/bin/python scripts/autoresearch_config_search.py rank \
     --results 'reports/*results*.json' \
     --results 'reports/*metrics*.json' \
     --results 'reports/latest_*.json' \
     --results 'reports/tide_selected*.json' \
     --output reports/autoresearch_rank.json
   ```

2. Generate the next bounded grid:

   ```bash
   ./.venv/bin/python scripts/autoresearch_config_search.py propose \
     --results 'reports/*results*.json' \
     --results 'reports/*metrics*.json' \
     --results 'reports/latest_*.json' \
     --results 'reports/tide_selected*.json' \
     --template-grid configs/gmm_tide_fm_next10_grid.json \
     --output-grid configs/autoresearch/gmm_tide_fm_autoresearch_grid.json \
     --label "$(date -u +%Y%m%d)" \
     --budget 6
   ```

3. Validate the generated grid locally:

   ```bash
   ./.venv/bin/python -m json.tool configs/autoresearch/gmm_tide_fm_autoresearch_grid.json >/dev/null
   ./.venv/bin/python - <<'PY'
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path("scripts").resolve()))
   from submit_gmm_tide_fm_jobs import load_grid
   jobs = load_grid(Path("configs/autoresearch/gmm_tide_fm_autoresearch_grid.json"))
   print(f"loaded {len(jobs)} jobs")
   PY
   ```

4. Only when the human explicitly asks to submit, run a dry-run first:

   ```bash
   ./.venv/bin/python scripts/submit_gmm_tide_fm_jobs.py \
     --grid-config configs/autoresearch/gmm_tide_fm_autoresearch_grid.json \
     --owners all \
     --exclude-owners kieutung,no1ceboy \
     --accelerator tpu \
     --report-path reports/autoresearch_submit_dryrun.json \
     --dry-run \
     --no-shared-context
   ```

5. If the dry-run report is sound and the human approves actual Kaggle use, submit with the same command minus `--dry-run`. After completion, collect diagnostics with the existing collector and feed the new result report into the next ranking step.

## Research Guardrails

- Keep each candidate batch small: 4 to 8 jobs is the default range.
- Prefer local neighborhoods around the best measured configs before trying unrelated ideas.
- Avoid duplicate config fingerprints even if the run name differs.
- Record why each generated candidate exists in the Markdown companion report.
- Treat Kaggle credentials as sensitive. Never print, copy, commit, or summarize API keys.
- Do not commit until documentation and any repo-specific release/checklist requirements have been handled.

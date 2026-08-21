# Agent Handoff Brief — IEEE-CIS Fraud Detection Case Study

Place this file (or its contents) at the repo root as `CLAUDE.md` so a Claude
Code cloud agent picks it up automatically as project context, or paste it
directly as the first message when starting the cloud agent session.

---

## Context

This repo is a graduate data-science case study on the IEEE-CIS Fraud
Detection dataset (Kaggle). The grading rubric and full requirements are in
`IMPLEMENTATION_PLAN.md` at the repo root — **read that file first**, it is
the source of truth for scope, phase order, and deliverables. Do not deviate
from its phase structure without flagging the deviation back to me.

## Objective

Execute Phases 0–9 from `IMPLEMENTATION_PLAN.md` in order, producing:
- working, tested code under `src/`
- a populated `notebooks/02_eda.ipynb`
- markdown/notebook write-ups under `reports/final/` for every phase that
  requires one (DGP, feature dictionary, leakage audit, comparative analysis)
- a final stitched report at `reports/final/REPORT.md`

## Data access

Data is provided via Google Drive, **not** the Kaggle API. Full provenance,
expected filenames, and a checksum table live in `data/raw/README.md`
(tracked in git, unlike the CSVs themselves) — read that file first and
verify your local copy against it before touching Phase 0's schema checks.
If the checksum table in that file is still blank (first download), fill it
in after verifying the files look right, then commit it so future sessions
don't have to re-verify from scratch.

Most cloud agent sandboxes do **not** have unauthenticated internet access,
and Drive folder links generally require either a logged-in Google session
or an explicit connector/tool to fetch — a bare HTTP fetch to a
`drive.google.com` URL will very likely fail or return an HTML login page,
not the CSVs. Two ways to get the agent unblocked:

1. **Preferred:** download the five files locally yourself and place them in
   `data/raw/` *before* starting the agent session. Tell the agent in your
   opening message that the files are already present, so it skips any
   download step entirely and goes straight to Phase 0's schema checks.
2. **If a Google Drive connector/tool is available** in the agent's
   environment, it can be used to pull the files directly — but confirm the
   tool is actually connected before relying on it; don't assume.

If the agent cannot access the files by either path, it should say so
explicitly and stop, rather than substituting synthetic data silently.

## Hard constraints (do not violate)

1. **Never commit raw data.** `data/raw/`, `data/processed/`, `data/interim/`
   are already gitignored except for `.gitkeep`. Do not modify `.gitignore`
   to include them.
2. **Data leakage is the single highest-risk failure mode for this
   assignment** (explicitly flagged as critical in the source brief). Any
   feature that aggregates across rows must be computed using only
   information available strictly before that row's `TransactionDT`. Flag
   and self-review every feature in `src/features/build_features.py`
   against this rule before moving to Phase 4.
3. **Train/test split must be chronological**, not random — `TransactionDT`
   defines relative order (it is a timedelta from an unknown reference, not
   a real timestamp — do not derive true calendar features like actual
   hour-of-day from it without labeling them as assumptions). Do not use
   `sklearn`'s default shuffled `train_test_split` or standard k-fold
   without a time-aware variant.
4. **No Transformer model, at all.** Not required for this dataset option
   and explicitly out of scope — do not build one, even as a stretch goal,
   even if extra time remains. Redirect any spare time to Phase 8 (leakage
   audit) or Phase 9 (write-up polish) instead.
5. `transaction` and `identity` files join on `TransactionID` via a **left
   join from `transaction`** — not all transactions have an identity row.
   Treat "no identity match" as a potential feature, not just a null to fill.
6. **Every feature and every modeling choice needs a one-line "why" comment
   or docstring** — the assignment is graded partly on justification, not
   just on working code. Code without rationale is an incomplete deliverable
   here, even if it runs correctly.
7. Keep `requirements.txt` in sync with anything you actually import; don't
   let the environment drift from what's declared.

## Researching other participants' approaches (optional, if network access permits)

The competition overview page
(https://www.kaggle.com/competitions/ieee-fraud-detection/overview) links to
a discussion board and public notebooks/kernels where other participants —
including top-ranked solutions — describe their feature engineering and
modeling choices. Consulting these for inspiration is fine, subject to the
rules below; whether it's *possible* depends on what's allowlisted for this
agent session.

**If you (the person running the agent) can configure a domain allowlist,
add:**
- `www.kaggle.com` — covers the competition overview, `/discussion`, and
  `/code` (public notebooks) under `ieee-fraud-detection`. This one domain
  is sufficient for the agent to read discussion threads and published
  solution notebooks directly on Kaggle.

**Do not blanket-allow beyond that.** Discussion posts sometimes link out to
personal blogs, GitHub repos, or YouTube write-ups of solutions — those are
on different domains and would each need individual review before adding,
since a broad allowlist (e.g., all of `github.com`) is a much wider surface
than this task needs. If the agent hits a linked external domain that isn't
allowlisted, it should note what it wanted to check and move on rather than
stall on it.

**Rules if the agent does read external sources:**
- Use them for inspiration/sanity-checking only (e.g., "did others also find
  the `V*` block redundant" or "what aggregation windows did the winning
  solution use") — cite what you drew from them.
- Any technique adopted must be independently justified in this project's
  own write-up, not justified by "a top Kaggle solution did this."
- Don't copy code verbatim from public kernels — reimplement in this repo's
  own style, so the leakage-safety guarantees in Phase 8 can be verified
  against this project's actual pipeline rather than inherited unchecked.

## Working style

- Work phase by phase, in the order given in `IMPLEMENTATION_PLAN.md`.
  Commit after each phase with a message referencing the phase number
  (e.g. `feat(phase-3): feature engineering + leakage-checked aggregations`).
- After each phase, write its deliverable (per the plan's "Deliverable" line)
  before moving to the next phase — don't batch all code first and
  write-ups later; the write-ups are graded as heavily as the code (30%
  presentation weight).
- If Kaggle credentials are not available in this environment, stub the data
  loading step, note it clearly in `data/raw/README.md`, and proceed using a
  small synthetic sample with the same schema so the rest of the pipeline
  can still be built and tested — flag this limitation prominently in the
  final report rather than silently working around it.
- Run `pytest src/tests` before each phase commit if tests exist for that
  phase's code; add a minimal smoke test for any new `src/` module
  (loads without error, expected output shape/columns).
- If a design decision in `IMPLEMENTATION_PLAN.md` seems to conflict with
  something you discover in the actual data (e.g., a column doesn't exist,
  a described pattern isn't present), stop and note the discrepancy in your
  final summary rather than silently improvising a replacement plan.

## Definition of done

- All 10 phases have both code (where applicable) and a written deliverable.
- `reports/final/REPORT.md` reads as one coherent narrative end-to-end, not
  a concatenation of disconnected phase notes.
- `git log` shows one commit per phase with descriptive messages.
- Final message back to me: a short summary of the comparative results
  (Phase 9 table) and any flagged limitations or deviations from the plan.
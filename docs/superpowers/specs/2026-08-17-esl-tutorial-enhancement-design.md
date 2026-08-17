# ESL Tutorial Enhancement — Design Spec

**Date:** 2026-08-17
**Status:** Approved
**Goal:** Professionalize the ESL (The Elements of Statistical Learning) scaffolding and produce real, runnable, teaching-quality content for all 175 sections across 23 chapters.

## 1. Goal & Scope

Transform the current project — an auto-generated scaffolding of ~175 empty `.py`/`.md` pairs plus a single `main.py` generator — into a professional, well-structured learning resource.

Scope: **All 23 chapters, full sweep, tiered depth.**
- Every section: a Chinese `.md` summary + a runnable `.py` example.
- Selected "deep" sections: additionally a richer `.ipynb` notebook walkthrough.

## 2. Architecture

Modernize the generator from the single `main.py` into a config-driven, template-based system.

- **Structure config:** `config/structure.yaml` holds the full chapter → section hierarchy (no longer hardcoded in Python). Each section node may carry metadata:
  - `deep: true` → mark for notebook + rich treatment.
  - `summary` → short description used in generation and README status.
- **Generator:** `scripts/generate_docs.py` (or refactored `main.py`) — reads config, generates/regenerates:
  - `.md` summary stubs (filled with content where authored)
  - `.py` code file per section
  - `.ipynb` notebook for deep sections
  - `README.md` with a status-driven table of contents
- **Templates:** `scripts/templates/*` — consistent docstring/markdown structure for every generated artifact.
- **Python hygiene:** type hints, docstrings, `logging`, CLI flags (`--init`, `--regenerate`), `if __name__ == "__main__"`.

### Supporting files
- `requirements.txt` — numpy, scipy, scikit-learn, pandas, matplotlib, jupyter/nbformat.
- `scripts/` — generator + helper utilities.
- `config/` — structure + deep-section metadata.

## 3. Content Model

### Per section (all 175)
- **`.md`** (Chinese): definition, intuition, key formulas in LaTeX, role in the chapter, worked-link to the `.py`/`.ipynb`. Teaching narrative in Chinese with English technical terms where natural.
- **`.py`** (runnable): clean, dependency-consistent code; synthetic data where real datasets are unavailable; a `main()`/entry; figures saved to a `figures/` convention with Chinese-friendly labels; clear comments. Runs standalone.

### Deep sections (notebooks)
- **`.ipynb`**: narrative markdown + code cells + saved outputs, a true walkthrough. Runnable in Jupyter.

## 4. Deep-Section List (config-flagged)

- 4.5 Local Methods in High Dimensions (kNN)
- 4.9 Model Selection & Bias–Variance
- 5.2 Linear Regression & Least Squares
- 5.4 Shrinkage Methods (ridge/lasso)
- 5.5 Methods Using Derived Input Directions (PCR/PLS)
- 6.3 Linear Discriminant Analysis
- 6.4 Logistic Regression
- 7.2 Piecewise Polynomials & Splines
- 7.4 Smoothing Splines
- 9.10 Cross-Validation
- 9.11 Bootstrap
- 10.5 EM Algorithm
- 10.7 Bagging
- 11.2 Tree-Based Methods
- 11.4 MARS
- 12.1 Boosting
- 12.4 AdaBoost (Exponential Loss)
- 12.10 Gradient Boosting
- 13.3 Neural Networks
- 14.2/14.3 SVM
- 15.3 k-NN Classifiers
- 16.3 Cluster Analysis
- 16.5 Principal Components
- 17.2 Random Forests

(The exact set is finalized in the implementation plan; adjust only if a section lacks a natural runnable demo.)

## 5. README & Navigation

Rewrite `README.md`: professional Chinese index with:
- accurate multi-level table of contents
- per-section status badges (✅ summary / 🔬 notebook / ⏳ TODO)
- install + usage instructions

## 6. Figures & Assets Convention

- All generated figures saved under `figures/<chapter>/<section>_*.png`.
- Code produces figures deterministically from synthetic data unless a standard dataset (e.g. iris, digits, spam) is used.

## 7. Error Handling & Verification

- Generator is idempotent and safe to re-run.
- Every `.py` verified runnable before marking ✅ (verification gate: run each file headless, confirm no exceptions).
- Notebooks verified to execute cell-by-cell in Jupyter.

## 8. Testing Strategy

- `scripts/generate_docs.py` self-tests (config parses, structure matches disk).
- A smoke test runs every `.py` and reports failures.
- Markdown/LaTeX sanity checks (balanced `$`/`$$`).

## 9. Packaging & Distribution

- Clean tree, `requirements.txt`, consistent imports.
- Prefer verified, already-run code.

## 10. Delivery Phases (input to implementation plan)

- **Phase 1 — Scaffolding:** config-driven generator, templates, README index with statuses, requirements, generator self-tests.
- **Phase 2+ — Content, chapter by chapter:** for each chapter in sequence: Chinese `.md` summaries, runnable `.py`, and deep `.ipynb` for flagged sections. Priority order for content work: foundations → regression → classification → smoothing → resampling → boosting/trees → neural nets → SVM → unsupervised.
- Continuous verification gate at each phase.
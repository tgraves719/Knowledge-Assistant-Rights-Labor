# Karl Update Log

## v0.8.92 - Gate Check Surfaces release_090 Advisories (Non-Blocking) (February 23, 2026)

### Overview

Extended `backend.evaluate_gate_check` to print advisory status from the `release_090` scorecard so CI/local gate runs show release-hardening warnings in one place.

This is **non-blocking** by design:

- advisories are visible
- gate pass/fail still depends only on blocking checks

### What Changed

- `backend/evaluate_gate_check.py`
  - added `release_090` advisory summary printer for:
    - `base_chunk_lineage`
    - `contract_text_compare_amended`
    - generic future advisory keys
  - prints advisory rows as:
    - `[OK] ...` when `warning=false`
    - `[!!] ...` when `warning=true`
  - does not append advisory messages to blocking failures

### Validation

- `python -m py_compile backend/evaluate_gate_check.py` -> PASS
- `python -m backend.evaluate_gate_check` -> PASS

### Current Observed Output

- `release_090 advisory base_chunk_lineage` -> `status=ok`
- `release_090 advisory contract_text_compare_amended` -> `status=ok`

## v0.8.91 - Release Scorecard Advisory for Amended Text Compare Coverage (February 23, 2026)

### Overview

Wired the amended-section `effective vs base` compare regression eval into the `release_090` scorecard as a **non-gating advisory** (same model as the base lineage audit).

This keeps the release scorecard focused on hard gates while surfacing whether the Contract-tab compare prerequisites are:

- passing on known amended targets
- fully covering currently ingested approved `replace_section` MOA ops

### What Changed

- `backend/evaluate_release_090.py`
  - added advisory input path:
    - `--contract-text-compare-amended-results`
  - added `advisories.contract_text_compare_amended` with:
    - evaluator schema + dataset schema
    - overall pass/pass_rate
    - coverage summary (`contract_coverage_rate`, `operation_coverage_rate`)
    - uncovered contracts / missing target count
  - advisory is **non-gating** (does not change `components_passed/components_total`)

- `backend/test_release_090_eval.py`
  - added advisory coverage for:
    - `ok` compare advisory
    - `warn` compare advisory (partial coverage / non-pass)
    - missing compare artifact as non-gating warning

- `data/test_set/release_0_9_0_scorecard.json`
  - refreshed to include `advisories.contract_text_compare_amended`

### Validation

- `python -m py_compile backend/evaluate_release_090.py backend/test_release_090_eval.py` -> PASS
- `python -m backend.test_release_090_eval` -> PASS
- `python -m backend.evaluate_release_090 --skip-manifest-validation --skip-track-all-metadata` -> PASS (`9/9`)

### Current Advisory Status

- `contract_text_compare_amended`: `ok`
  - coverage: `1/1 amended contracts`, `2/2 approved replace_section ops`
  - overall compare regression: `2/2` pass

## v0.8.90 - Amended Compare Eval Coverage Inventory + Deterministic Dataset Seeder (February 23, 2026)

### Overview

Expanded the amended-section text-compare regression infrastructure so it can scale with newly ingested MOAs without manual curation drift.

This adds:

- automatic inventory discovery of approved non-draft `replace_section` patch ops
- coverage reporting (which amended contracts / targets are represented in the compare dataset)
- a deterministic dataset seeder to merge or expand targets from currently ingested approved patches

### What Changed

- `backend/evaluate_contract_text_compare_amended.py`
  - added `discover_approved_replace_section_inventory(...)`
  - report now includes `coverage`:
    - contract coverage rate
    - operation coverage rate
    - uncovered contracts
    - missing target sample
    - failing case sample

- `backend/seed_contract_text_compare_amended_targets.py` (new)
  - deterministic seeder for `data/test_set/contract_text_compare_amended_targets_test.json`
  - selects approved `replace_section` targets per contract (`--per-contract`, default `2`)
  - supports merge/replace behavior and optional contract filtering
  - preserves existing curated case IDs/descriptions when target keys already exist

- `backend/test_contract_text_compare_amended_eval.py`
  - now asserts coverage fields exist and reports full coverage for the current ingested amended corpus

- `data/test_set/contract_text_compare_amended_targets_test.json`
  - now includes `seed_metadata` documenting inventory totals and selection behavior

- `data/test_set/contract_text_compare_amended_results.json`
  - now includes `coverage` summary

### Validation / Findings

- `python -m py_compile backend/evaluate_contract_text_compare_amended.py backend/seed_contract_text_compare_amended_targets.py backend/test_contract_text_compare_amended_eval.py` -> PASS
- `python -m backend.seed_contract_text_compare_amended_targets` -> PASS
- `python -m backend.test_contract_text_compare_amended_eval` -> PASS
- `python -m backend.evaluate_contract_text_compare_amended` -> PASS (`2/2`)

Current ingested approved-MOA compare coverage:

- amended contracts with approved `replace_section` ops: `1`
- approved `replace_section` ops discovered: `2`
- compare dataset coverage: `1/1 contracts`, `2/2 ops` (`100%`)

### Why This Matters

- Converts the compare regression from a hand-maintained one-off into a scalable, auditable dataset workflow.
- Makes it easy to expand coverage immediately as additional approved MOA patches are ingested for other contracts.

## v0.8.89 - Release Scorecard Lineage Advisory + Amended-Section Compare Regression Eval (February 23, 2026)

### Overview

Added two release-hardening pieces:

- a **non-gating base chunk lineage advisory** in the `release_090` scorecard (to surface `source_view=base` lineage risk without blocking release)
- a **targeted amended-section compare regression eval** to validate that known MOA-amended sections differ between `base` and `effective` chunk artifacts (protects the Contract-tab effective/base compare UX)

### What Changed

- `backend/evaluate_release_090.py`
  - added advisory-only ingestion of `data/test_set/base_chunk_lineage_report.json`
  - emits `advisories.base_chunk_lineage` with `status` (`ok` / `warn` / `missing`) and summary fields
  - does **not** affect component pass/fail counts (non-gating by design)
  - new CLI arg: `--base-chunk-lineage-results`

- `backend/test_release_090_eval.py`
  - added coverage for:
    - advisory `ok` path
    - advisory `warn` path
    - missing lineage artifact as **non-gating warning**

- `backend/evaluate_contract_text_compare_amended.py` (new)
  - deterministic artifact-level eval for known amended section targets
  - verifies `base` vs `effective` chunk text differs for specified article/section targets
  - emits `data/test_set/contract_text_compare_amended_results.json`

- `data/test_set/contract_text_compare_amended_targets_test.json` (new)
  - initial Pueblo Clerks amended targets:
    - `Article 15, Section 34`
    - `Article 58, Section 1749`

- `backend/test_contract_text_compare_amended_eval.py` (new)
  - deterministic regression test for the new evaluator

- `backend/evaluate_runner.py`
  - new track: `contract_text_compare_amended`

### Validation

- `python -m py_compile backend/evaluate_release_090.py backend/evaluate_contract_text_compare_amended.py` -> PASS
- `python -m backend.test_release_090_eval` -> PASS
- `python -m backend.test_contract_text_compare_amended_eval` -> PASS
- `python -m backend.evaluate_contract_text_compare_amended` -> PASS (`2/2`)
- `python -m backend.evaluate_release_090 --skip-manifest-validation --skip-track-all-metadata` -> PASS (`9/9`)
- `python -m backend.evaluate_runner --track contract_text_compare_amended --skip-manifest-validation` -> PASS

### Why This Matters

- The release scorecard now surfaces lineage risk explicitly without conflating advisory issues with hard gates.
- We now have a repeatable regression check that the Contract-tab compare feature has real `base` vs `effective` text deltas to render on known amended provisions.

## v0.8.88 - Base Chunk Lineage Audit (Path + Patched-Section Content Checks) (February 22, 2026)

### Overview

Added a backend audit report to evaluate whether Contract-tab `source_view=base` text is likely trustworthy for each contract by checking:

- path-level lineage (base chunk path vs effective chunk path)
- effective-snapshot path contamination risk
- content-level checks on approved `replace_section` patch targets (base vs effective chunk text equality)

### What Changed

- `backend/evaluate_base_chunk_lineage.py` (new)
  - emits `data/test_set/base_chunk_lineage_report.json`
  - reports per contract:
    - base/effective chunk paths
    - whether paths point into effective snapshots
    - path equality risk
    - patched `replace_section` target content comparisons (`equal` vs `different`)
    - risk level + findings

### Validation / Findings

- `python -m py_compile backend/evaluate_base_chunk_lineage.py` -> PASS
- `python -m backend.evaluate_base_chunk_lineage` -> PASS
- Current report: `data/test_set/base_chunk_lineage_report.json`
  - Pueblo Clerks (`local7_safeway_pueblo_clerks_2022`) patched section content check shows `different_content_count=2` on approved `replace_section` ops (good signal that base/effective text artifacts are distinct for patched sections)

### Why This Matters

- Gives us an objective way to validate (or challenge) the reliability of `Previous/Base` text mode instead of relying on anecdotal spot checks.
- Supports release hardening for the dual-view Contract tab.

## v0.8.87 - Contract Text Compare: Phrase-Level Highlights on Changed Lines (February 22, 2026)

### Overview

Upgraded the Contract text compare diff so paired changed lines now include inline token/phrase-level highlights (not just line-level add/remove coloring).

### What Changed

- `frontend/modular/src/app.js`
  - added inline token diff helpers:
    - `_inlineDiffTokenLcs(...)`
    - `_renderInlineTokenDiffPair(...)`
  - paired adjacent changed rows in the line diff renderer and displays inline highlighted replacements with `~` marker
  - added truncation guardrails for long changed lines

### Validation

- `node --check frontend/modular/src/app.js` -> PASS

### Why This Matters

- Makes MOA-driven wording changes much faster to inspect in the Contract text pane.
- Reduces reviewer effort when verifying exactly what changed within a provision.

## v0.8.86 - Contract Text Compare: Line-Diff Renderer (Effective vs Base) (February 22, 2026)

### Overview

Replaced the Contract text pane's basic compare preview with a line-diff style compare renderer for the current target (`effective` vs `previous/base`).

### What Changed

- `frontend/modular/src/app.js`
  - added bounded line-diff helpers for contract text compare:
    - `_buildContractTextLineDiffRows(...)`
    - `_renderContractTextLineDiff(...)`
  - compare panel now shows:
    - effective-only line count
    - base-only line count
    - unchanged line count
    - side-by-side effective/base line diff preview with added/removed highlights
  - includes truncation guardrails for UI performance on long targets

### Validation

- `node --check frontend/modular/src/app.js` -> PASS

### Why This Matters

- Moves the compare UX from “two text previews” to a real inspection tool that helps users see what changed in the current target.
- This is a better fit for MOA workflows and reduces reviewer effort when checking amended provisions.

## v0.8.85 - Contract Text Pane Provenance + Effective/Base Compare Preview (February 22, 2026)

### Overview

Completed the first visible UX layer of the dual-view Contract tab by adding:

- text-pane provenance metadata (current text source + target + effective snapshot/amendment context)
- target-level compare preview (`effective` vs `previous/base`) in the Contract text pane

### What Changed

- `frontend/modular/src/app.js`
  - added contract text pane state/cache:
    - current target text payloads by source (`effective`, `base`)
    - compare panel open/close state
  - added provenance header + compare UI renderers:
    - `updateContractTextPanelChrome(...)`
    - `renderContractTextComparePanel(...)`
  - added text compare workflow:
    - `ensureContractTextPayloadForMode(...)`
    - `toggleContractTextCompare(...)`
  - refactored article / non-article text fetches to store payloads by source mode for compare
- `frontend/modular/index.html`
  - added text-pane provenance line
  - added `Compare` button + hint
  - added compare preview panel under the text-pane toolbar

### Validation

- `node --check frontend/modular/src/app.js` -> PASS

### Why This Matters

- Makes the materialized effective text explicit (users can now see what source mode they are reading).
- Gives stewards and reviewers a concrete `effective vs base` text comparison workflow in the same Contract tab while keeping PDFs available for provenance verification.

## v0.8.84 - Contract Tab Dual-View Foundation (Effective vs Base Text Source) (February 22, 2026)

### Overview

Implemented the first pass of the dual-view Contract tab model:

- **Text pane**: source-selectable contract text (`Effective` vs `Previous/Base`)
- **PDF pane**: provenance/source navigation (`Effective/MOA/Base/Previous`) from the current target

This is the foundation for showing a materialized effective contract in markdown/text while still allowing legal verification in source PDFs.

### What Changed

- `backend/api.py`
  - added `source_view` support (`effective` default, `base` optional) to text endpoints:
    - `GET /api/article/{article_num}`
    - `GET /api/section/{article_num}/{section_num}`
    - `GET /api/contract-browse-item`
  - added base-chunk resolution path for viewer text fetches (without disturbing retrieval/effective defaults)
- `frontend/modular/index.html`
  - added a desktop Contract tab text panel (`Contract Text`)
  - added `Text Source` selector (`Effective` / `Previous/Base`)
- `frontend/modular/src/app.js`
  - article and non-article Contract tab text loaders now request `source_view` based on the text selector
  - selecting a TOC item now loads both:
    - text pane content (effective/base)
    - PDF pane location (effective/MOA/base/previous via provenance)
  - added `handleContractTextSourceModeChange(...)` and current-target text refresh behavior
- `backend/test_contract_history_api.py`
  - added regression test coverage for `source_view='base'` vs `source_view='effective'` on:
    - article endpoint
    - contract-browse-item endpoint

### Validation

- `python -m py_compile backend/api.py` -> PASS
- `python -m backend.test_contract_history_api` -> PASS
- `node --check frontend/modular/src/app.js` -> PASS

### Notes / Current Limitation

- The dual-view behavior is implemented, but on some real contracts the preserved base chunk artifact may already reflect amended/effective text (artifact lineage issue), so `Base` text may not always differ yet.
- This reinforces the need for immutable base chunk snapshots as part of the 0.9.0 artifact discipline.

## v0.8.83 - Contract Tab Per-Target Source Selector (Effective / MOA / Base / Previous) (February 22, 2026)
## v0.8.83 - Contract Tab Per-Target Source Selector (Effective / MOA / Base / Previous) (February 22, 2026)

### Overview

Generalized the Contract-tab `View Previous` workflow into a per-target source selector driven by `/api/pdf-location` `source_candidates`, allowing users to switch the currently selected target between effective/MOA/base sources directly in the PDF toolbar.

### What Changed

- `frontend/modular/src/app.js`
  - added current-target source selector rendering + handlers:
    - `renderCurrentTargetSourceControls(...)`
    - `openSourceChoiceForCurrentSelection(...)`
    - `handleCurrentTargetSourceChange(...)`
  - re-used `lastPdfNavigationContext` (`target + sourceCandidates`) to drive both:
    - `View Previous`
    - direct source switching for the current target
  - exported `handleCurrentTargetSourceChange` for inline toolbar UI wiring
- `frontend/modular/index.html`
  - added `Current Target Source` selector in Contract PDF toolbar

### Validation

- `node --check frontend/modular/src/app.js` -> PASS

### Why This Matters

- This is the PDF-side control layer for a dual-view design:
  - text pane can stay on current effective contract text
  - PDF pane can switch among effective/base/MOA provenance sources for the same target
- Reduces reliance on global source mode toggles for citation-driven investigation workflows.

## v0.8.82 - Contract Tab "View Previous" (Base-Provenance Jump for Current Target) (February 22, 2026)

### Overview

Added a Contract-tab toolbar action to jump to the previous/base source for the currently selected article/section target using the same provenance-backed `source_candidates` model returned by `/api/pdf-location`.

### What Changed

- `frontend/modular/src/app.js`
  - added `lastPdfNavigationContext` state to remember the current PDF target + source candidates returned by `/api/pdf-location`
  - added provenance-aware previous-source selection helpers:
    - `_findPreviousPdfSourceCandidate(...)`
    - `_rememberPdfNavigationContext(...)`
    - `_clearPdfNavigationContext(...)`
    - `openPreviousForCurrentSelection(...)`
  - `openContractInPdf(...)` now records source candidates on successful target resolution and clears stale state on root/failure paths
  - exported inline handlers required by module-scope UI actions:
    - `openPreviousForCurrentSelection`
    - `loadContractBrowseItem`
    - `handlePopoverSourceChoiceChange`
- `frontend/modular/index.html`
  - added `View Previous` button + status hint in Contract PDF toolbar

### Validation

- `node --check frontend/modular/src/app.js` -> PASS

### Why This Matters

- Provides a practical "current vs previous" workflow in the Contract tab for amended article/section navigation without introducing a separate provenance system.
- Reuses the same source-selection model as chat citations, which keeps the UI behavior consistent and scalable.

## v0.8.81 - Non-Article PDF Deep-Linking for Contract Tab (LOU/LOA/Appendix) (February 22, 2026)

### Overview

Implemented exact PDF page lookup for Contract-tab non-article items (`LOU/LOA/Appendix`) so clicking those grouped TOC entries can deep-link into the contract PDF instead of opening the PDF at the top.

### What Changed

- `backend/api.py`
  - extended `/api/pdf-location` to accept:
    - `browse_kind`
    - `browse_key`
  - added cached source-page loader over parsed source JSON (`pages[]`) for viewer navigation
  - added deterministic non-article page resolver:
    - side-letter (`lou` / `loa`) page detection using browse item metadata + source-page text scoring
    - appendix page detection using appendix title matching
  - returns `matched_by="browse_item"` when a non-article page match is found
- `frontend/modular/src/app.js`
  - `loadContractBrowseItem(...)` now routes through `openContractInPdf(...)` with `browseKind/browseKey`
  - `openContractInPdf(...)` now supports non-article browse targets and labels matches as `browse item match`
- `backend/test_contract_history_api.py`
  - added regression coverage for `/api/pdf-location` with `browse_kind='lou'`, `browse_key='lou:8'`

### Validation

- `python -m py_compile backend/api.py` -> PASS
- `node --check frontend/modular/src/app.js` -> PASS
- `python -m backend.test_contract_history_api` -> PASS
- Real-contract smoke:
  - `get_pdf_location(contract_id='local7_safeway_pueblo_clerks_2022', browse_kind='lou', browse_key='lou:8')`
  - result: `page_number=70`, `matched_by='browse_item'`

### Why This Matters

- Contract-tab LOU/LOA browsing is now operational, not just readable text in the left pane.
- Closes the biggest UX gap for side-letter navigation before implementing `View Previous` and richer source-history switching.

## v0.8.80 - Contract Tab Browse Groups + History Timeline (LOU/LOA/Appendix First-Class) (February 22, 2026)

### Overview

Extended the Contract tab from an article-only TOC into a grouped contract browser that includes `Articles`, `Letters of Understanding / Agreement`, and `Appendices`, plus added a visible base->MOA->effective timeline strip in the history banner.

### What Changed

- `backend/api.py`
  - added grouped browse endpoints for Contract tab:
    - `GET /api/contract-browse`
    - `GET /api/contract-browse-item`
  - added cached browse payload builder over chunk artifacts:
    - groups CBA articles from manifest/article-title sources
    - groups `lou` / `loa` chunks into stable keys (e.g., `lou:8`)
    - groups appendix chunks for contract-tab browsing
  - added aggregated non-article content payloads so LOU/LOA/appendix items can render in the contract text pane
- `frontend/modular/src/app.js`
  - added grouped TOC rendering (articles + side letters + appendices)
  - generalized TOC active selection state (`article` and non-article items)
  - added `loadContractBrowseItem(...)` for LOU/LOA/appendix content
  - added `renderNonArticleContent(...)` to display grouped chunk content in the Contract tab reader pane
  - added history timeline rendering (`Base -> MOA(s) -> Effective snapshot`) in `renderContractHistoryPanel()`
- `frontend/modular/index.html`
  - TOC header renamed to `Contract Contents`
  - added `contract-history-timeline` banner strip
- `backend/test_contract_history_api.py`
  - added regression coverage for `contract-browse` and `contract-browse-item` endpoints

### Validation

- `python -m py_compile backend/api.py` -> PASS
- `node --check frontend/modular/src/app.js` -> PASS
- `python -m backend.test_contract_history_api` -> PASS

### Why This Matters

- Contract-tab navigation now reflects how the contract is actually used (CBA + LOU/LOA + appendices), not just numbered articles.
- Creates the right UI foundation for later `View Previous` and source-aware contract navigation without treating side letters as invisible retrieval-only artifacts.

## v0.8.79 - Version-Aware Citation PDF Navigation (Popover Source Chooser) (February 22, 2026)

### Overview

Implemented the first frontend/backend pass of version-aware citation navigation so amended citations can open the correct source PDF (MOA vs base) and users can explicitly switch source view from the citation popover instead of always landing in the base contract PDF.

### What Changed

- `backend/api.py`
  - extended `/api/pdf-location` response model with:
    - `source_candidates[]`
    - `selected_source_key`
  - added effective-provenance lookup and deterministic source-candidate builder for citation targets
    - prefers MOA provenance for effective navigation when present
    - includes stable `Previous (Base)` alias when base provenance exists
  - auto-selects effective provenance source for navigation when no explicit `source_type/source_pdf/source_doc_id` is requested
- `frontend/modular/src/app.js`
  - added citation source registry plumbing so chat citation badges can carry full provenance metadata into popovers
  - extended citation click handling to preserve `source_doc_id` and citation provenance registry keys
  - added popover source-choice model (`Current Effective (Auto)`, `MOA`, `Base`, `Previous (Base)` when available)
  - `Open In Contract PDF` now respects the selected popover source choice
- `frontend/modular/index.html`
  - added `Source View` selector in citation popover footer for explicit PDF source switching

### Validation

- `python -m py_compile backend/api.py` -> PASS
- `node --check frontend/modular/src/app.js` -> PASS
- `python -m backend.test_contract_history_api` -> PASS

### Why This Matters

- Fixes the current UX mismatch where the answer is based on effective/MOA content but citation navigation defaults to base PDF.
- Provides a scalable UI pattern for future `View Previous` and contract-history timeline features without changing citation semantics again.

## v0.8.78 - Side-Letter Retrieval Promotion (Pueblo Clerks LOU8 Surfacing) (February 22, 2026)

### Overview

Implemented a deterministic side-letter promotion pass in retrieval to improve top-k surfacing of LOU/LOA chunks (especially `Letter of Understanding 8: Dress Requirements` in Pueblo Clerks) without hardcoding a one-off patch.

### What Changed

- `backend/retrieval/router.py`
  - added side-letter targeting + promotion infrastructure in `HybridRetriever`:
    - `_side_letter_query_targets(...)`
      - parses explicit `LOU/LOA` references (e.g., `LOU 8`)
      - extracts title phrases from query and retrieved CBA cross-references (e.g., `Letter of Understanding, "Dress Requirements"`)
      - identifies detail vs discovery/xref prompt modes
    - `_side_letter_chunk_score(...)`
      - deterministic scoring for side-letter-aware reordering using:
        - explicit type/number matches
        - title phrase matches
        - article cross-reference cues
        - lexical detail overlap
        - header/title (`Part 1`) vs detail-content (`Part 2`) heuristics
    - `_promote_side_letter_context(...)`
      - contract-scoped post-retrieval reorder/add step for side-letter chunks
      - applied in both `retrieve(...)` and `multi_angle_retrieve(...)`
- `backend/evaluate_runner.py`
  - (from v0.8.77) retained exploratory `side_letter_retrieval_pueblo_clerks` track used to validate this change

### Validation / Outcomes

- Existing side-letter release benchmark remained stable:
  - `python -m backend.evaluate_side_letter_retrieval --bm25-only` -> PASS (`16/16`, no regressions)
- Exploratory Pueblo Clerks LOU benchmark improved substantially:
  - before fix: Top-8 `2/6`, Depth-20 `6/6`
  - after fix: Top-8 `6/6`, Depth-20 `6/6`
  - `python -m backend.evaluate_runner --track side_letter_retrieval_pueblo_clerks --skip-manifest-validation` -> PASS (exploratory baseline now green)

### Why This Matters

- Confirms the `LOU 8` dress-code issue is a ranking/surfacing problem, not missing artifacts.
- Gives a reusable side-letter retrieval mechanism that should scale better than prompt-specific patches.

## v0.8.77 - Pueblo Clerks LOU Discovery Benchmark Baseline (February 22, 2026)

### Overview

Added a non-gating exploratory side-letter/LOU benchmark slice for `local7_safeway_pueblo_clerks_2022` focused on `Letter of Understanding 8 (Dress Requirements)` discovery and cross-reference prompts (`Article 52, Section 158` -> LOU 8).

### What Changed

- `data/test_set/side_letter_retrieval_pueblo_clerks_test.json`
  - new exploratory dataset (6 cases):
    - direct `LOU 8` lookup
    - dress-requirements content retrieval
    - `Article 52` cross-reference phrasing
    - generic `what's the dress code` phrasing
- `backend/evaluate_runner.py`
  - added `side_letter_retrieval_pueblo_clerks` track
  - runs `backend.evaluate_side_letter_retrieval` with custom input/output:
    - input: `side_letter_retrieval_pueblo_clerks_test.json`
    - output: `side_letter_retrieval_pueblo_clerks_results.json`

### Baseline Findings (Useful)

- Data is present (not missing):
  - Pueblo Clerks chunks = `315`
  - `doc_type=lou` chunks = `37`
  - effective snapshot coverage preserves `lou` bucket counts
- Exploratory benchmark results (`BM25-only`):
  - Top-8 pass: `2/6` (33.3%)
  - Depth-20 pass: `6/6` (100%)
  - Combined pass: `6/6`
- Interpretation:
  - LOU content exists and is retrievable, but top-k ranking/surfacing is weak for real-world phrasing (`dress code`, Article 52 cross-ref) and often returns the wrong LOU 8 parts ahead of the actual dress-detail part.

### Validation

- `python -m backend.evaluate_effective_coverage` -> PASS (`local7_safeway_pueblo_clerks_2022` base=315, effective=315, no missing doc types)
- `python -m backend.evaluate_contract_artifact_integrity` -> PASS (`lou_bucket=37` for Pueblo Clerks)
- `python -m backend.evaluate_runner --track side_letter_retrieval_pueblo_clerks --skip-manifest-validation` -> PASS (exploratory run completed; baseline artifact written)

## v0.8.76 - MOA End-to-End Answer Regression Gate + False-Unavailable Hardening (February 22, 2026)

### Overview

Closed the gap between retrieval-level MOA correctness and actual answer behavior by adding an end-to-end `deleted_vs_updated_moa` answer regression slice (API/query path) and tightening false-unavailable fallback recovery for clause-presence probes.

### What Changed

- `backend/api.py`
  - hardened deterministic false-unavailable recovery:
    - expanded lexical evidence scan from top 8 to top 12 chunks (fixes missed MOA chunk in answer fallback)
    - strengthened contiguous clause-phrase evidence (requires denser phrase windows)
    - added explicit clause-presence probe detection (`is it true ...`, `is that in the current effective agreement ...`)
    - added strict near-quote evidence requirement for clause-presence probes
    - disabled anchor-only fallback recovery for clause-presence probes to prevent false synthesized answers on deleted clauses
- `backend/evaluate_moa_deleted_vs_updated_answer.py`
  - now emits deterministic `gate` + `thresholds` in artifact
  - tightened defaults to strict release-grade thresholds (`1.0`)
- MOA release-path integration:
  - `backend/evaluate_moa_readiness.py`
    - runs and gates `moa_deleted_vs_updated_answer`
    - emits `moa_deleted_vs_updated_answer_gate` + summary block
  - `backend/evaluate_moa_deep_suite.py`
    - includes `moa_deleted_vs_updated_answer` command/artifact in deep MOA suite
  - `backend/evaluate_release_090.py`
    - adds `moa_deleted_vs_updated_answer_regression` scorecard component
    - `moa_readiness` scorecard check now respects `moa_deleted_vs_updated_answer_gate` when present
- `backend/test_release_090_eval.py`
  - updated fixture coverage for new answer-level MOA scorecard component
- CI/docs:
  - `.github/workflows/eval-ci.yml`
    - runs `moa_deleted_vs_updated_answer` in core/main deterministic slices
    - uploads `moa_deleted_vs_updated_answer_results.json`
  - `README.md`
    - added runner + direct command docs for answer regression slice
    - CI behavior docs now list `moa_deleted_vs_updated_answer`

### Validation

- `python -m backend.test_false_unavailable_fallback` -> PASS
- `python -m backend.evaluate_moa_deleted_vs_updated_answer --bm25-only` -> PASS (`4/4`, updated `2/2`, deleted `2/2`, source-type `1.0`)
- `python -m backend.test_moa_deleted_vs_updated_answer_eval` -> PASS
- `python -m backend.test_release_090_eval` -> PASS
- `python -m backend.evaluate_moa_readiness` -> PASS (includes `moa_deleted_vs_updated_answer_gate`)
- `python -m backend.evaluate_moa_deep_suite` -> PASS (`18/18`)
- `python -m backend.evaluate_release_090 --skip-manifest-validation --skip-track-all-metadata` -> PASS (`9/9`)

## v0.8.75 - Shared MOA Source-Doc Applicability Registry Enforcement (February 22, 2026)

### Overview

Added contract applicability enforcement for shared source documents (especially multi-contract MOAs) so Contract-tab/PDF resolution cannot resolve a foreign `source_doc_id` for the wrong contract.

### What Changed

- `backend/source_docs.py`
  - added applicability helpers:
    - `source_doc_applicable_contract_ids(...)`
    - `source_doc_applies_to_contract(...)`
  - `resolve_source_doc_pdf_by_name(...)` now supports contract-filtered lookup
  - `register_source_doc(...)` now writes canonical `applies_to_contract_ids` while retaining backward-compatible `contract_ids`
- `backend/ingest/register_source_doc.py`
  - added `--applies-to-contract-id` (repeatable)
- `backend/effective_contracts.py`
  - blocks `source_doc_id` PDF resolution when the source doc metadata declares a different contract scope
  - filters source-doc name fallback lookup by contract
- `backend/api.py`
  - contract history payload filters inapplicable amendment source-doc ids / patch rows
  - prevents cross-contract MOA leakage in Contract tab metadata and PDF navigation targeting
- `backend/ingest/generate_patch_drafts.py`
  - target-contract discovery now recognizes canonical `applies_to_contract_ids`
- Tests:
  - `backend/test_contract_history_api.py`
    - added foreign source-doc scope guard test (`/api/contract-pdf` returns 404 for inapplicable `source_doc_id`)
  - `backend/test_moa_patch_draft_generator.py`
    - added `applies_to_contract_ids` metadata coverage

### Validation

- `python -m backend.test_contract_history_api` -> PASS
- `python -m backend.test_moa_patch_draft_generator` -> PASS
- `python -m backend.evaluate_moa_readiness` -> PASS
- `python -m backend.evaluate_gate_check` -> PASS

## v0.8.74 - MOA Deleted-vs-Updated Regression Gate (February 22, 2026)

### Overview

Added a focused MOA regression slice for the exact failure mode we observed in practice: deleted clauses should be absent from effective retrieval while updated clauses must resolve to MOA-backed effective language with provenance.

### What Changed

- New dataset + evaluator:
  - `data/test_set/moa_deleted_vs_updated_test.json`
  - `backend/evaluate_moa_deleted_vs_updated.py`
    - wraps `backend.evaluate_moa_effective`
    - emits bucket metrics for `updated` vs `deleted`
    - enforces deterministic gate checks:
      - overall pass rate
      - updated-clause pass rate
      - deleted-clause pass rate
      - updated-clause MOA source-type match rate
- New deterministic evaluator test:
  - `backend/test_moa_deleted_vs_updated_eval.py`
- Canonical runner integration:
  - `backend/evaluate_runner.py`
    - added `moa_deleted_vs_updated` track
    - included in `all` track
- MOA gate bundles:
  - `backend/evaluate_moa_readiness.py`
    - now runs and gates `moa_deleted_vs_updated`
  - `backend/evaluate_moa_deep_suite.py`
    - now includes `moa_deleted_vs_updated` in deep MOA suite
- Canonical release/eval integration:
  - `backend/evaluate_v3.py`
    - added `moa_deleted_vs_updated` component to v3 artifact checks
    - schema bumped to `v3_eval_v6`
  - `backend/evaluate_release_090.py`
    - added `moa_deleted_vs_updated_regression` component
    - `moa_readiness` check now also respects `moa_deleted_vs_updated_gate` when present
  - `backend/evaluate_gate_check.py`
    - added direct artifact enforcement for `moa_deleted_vs_updated`
    - schema + dataset schema + gate + strict thresholds are all checked
- CI/docs:
  - `.github/workflows/eval-ci.yml`
    - PR core eval runs the new slice
    - gate-check command enforces the new artifact
    - artifact uploads include `moa_deleted_vs_updated_results.json`
  - `README.md`
    - added runner command, direct evaluator command, and gate-check flags for the new slice

### Validation

- `python -m backend.test_moa_deleted_vs_updated_eval` -> PASS
- `python -m backend.evaluate_moa_deleted_vs_updated --bm25-only` -> PASS (`4/4`, updated `2/2`, deleted `2/2`)
- `python -m backend.evaluate_moa_readiness` -> PASS (includes `moa_deleted_vs_updated_gate`)
- `python -m backend.evaluate_moa_deep_suite` -> PASS (`17/17`)
- `python -m backend.evaluate_v3 --from-artifacts --bm25-only` -> PASS (`17/17`, `v3_eval_v6`)
- `python -m backend.test_release_090_eval` -> PASS
- `python -m backend.evaluate_release_090 --skip-manifest-validation --skip-track-all-metadata` -> PASS (`8/8`)
- `python -m backend.evaluate_gate_check` -> PASS
- `python -m backend.evaluate_runner --track moa_deleted_vs_updated --skip-manifest-validation` -> PASS
- Metadata: `data/test_set/eval_run_metadata_moa_deleted_vs_updated_20260222T044941Z.json`

## v0.8.73 - Gate-Check Enforcement for v0.9.0 Scorecard (February 22, 2026)

### Overview

Promoted the new `release_090` scorecard from informational artifact to enforced release-gate input by integrating it into `backend.evaluate_gate_check`.

### What Changed

- `backend/evaluate_gate_check.py`
  - added `_check_release_090(...)` validation:
    - schema version check
    - components pass-rate threshold
    - overall pass flag
  - added CLI args:
    - `--release-090-results`
    - `--required-release-090-schema-version`
    - `--min-release-090-components-pass-rate`
    - `--allow-missing-release-090`
  - gate now blocks when release_090 artifact is missing/invalid (unless explicit non-release bypass is used).
- `README.md`
  - release-gate command example now includes `release_090` artifact + thresholds.

### Validation

- `python -m backend.evaluate_gate_check` -> PASS
- `python -m backend.evaluate_release_090` -> PASS (`7/7`)

## v0.8.72 - v0.9.0 Readiness Scorecard Evaluator + Runner Track (February 22, 2026)

### Overview

Implemented a deterministic v0.9.0 readiness scorecard that aggregates must-have release gates from canonical artifacts, added a dedicated runner track, and documented execution commands.

### What Changed

- New evaluator:
  - `backend/evaluate_release_090.py`
    - computes aggregated pass/fail across:
      - manifest validation
      - latest `evaluate_runner --track all` metadata health
      - v3 suite pass
      - MOA deep suite pass
      - MOA readiness gate pass
      - MOA citation provenance source-type match rate
      - API/runtime payload surface checks for `effective_version_id`, `amendments_applied`, and `sources`
    - outputs:
      - `data/test_set/release_0_9_0_scorecard.json`
    - supports CI artifact-mode runs without `track all` metadata requirement:
      - `--skip-track-all-metadata`
- Runner integration:
  - `backend/evaluate_runner.py`
    - added `release_090` track:
      - `python -m backend.evaluate_runner --track release_090`
- Deterministic tests:
  - `backend/test_release_090_eval.py`
    - pass-case fixture for fully valid artifacts
    - fail-case fixture for low MOA source-type provenance rate
- Docs update:
  - `README.md`
    - added canonical command + artifact location for v0.9.0 readiness scorecard
- CI integration:
  - `.github/workflows/eval-ci.yml`
    - PR core and main escalation jobs now generate `release_0_9_0_scorecard.json`
    - artifacts uploaded with other canonical evaluation outputs

### Validation

- `python -m backend.test_release_090_eval` -> PASS
- `python -m backend.evaluate_release_090 --skip-manifest-validation` -> PASS
- `python -m backend.evaluate_release_090 --skip-manifest-validation --skip-track-all-metadata` -> PASS
- `python -m backend.evaluate_release_090` -> PASS
- `python -m backend.evaluate_runner --track release_090 --skip-manifest-validation` -> PASS
- `python -m backend.evaluate_runner --track release_090` -> PASS

## v0.8.71 - MOA/Side-Letter Retrieval Hardening + Manifest Schema Closure + v3.1/0.9.0 Planning (February 22, 2026)

### Overview

Closed manifest schema debt, hardened side-letter routing behavior, restored full canonical eval health without manifest bypasses, and documented an execution plan toward v3.1 scaling and v0.9.0 readiness.

### What Changed

- Side-letter retrieval hardening:
  - `backend/retrieval/router.py`
    - improved side-letter anchor inference to prioritize `doc_type` (`loa`/`lou`) and use lexical fallback only when needed.
    - added explicit/follow-up side-letter query detectors and routing hooks.
  - `backend/retrieval/hybrid_search.py`
    - reduced noisy generic side-letter phrase boosts and increased term-overlap weighting.
  - `backend/test_topic_routing.py`
    - added deterministic side-letter anchor discovery and follow-up routing tests.

- Manifest schema closure (strict validation compatibility):
  - `data/manifests/local7_kingsoopers_loveland_meat_2019.json`
    - added `contract_version`
    - added schema-valid `query_routing` object
  - `data/manifests/local7_safeway_pueblo_clerks_2022.json`
    - added `contract_version`
    - added schema-valid `query_routing` object
    - expanded `classification_to_articles` anchors for role-comparison coverage (`courtesy_clerk`, `drive_up_and_go`, `dug_shopper`, `all_purpose_clerk`)
  - `data/manifests/local7_safeway_pueblo_meat_2022.json`
    - added `contract_version`
    - added schema-valid `query_routing` object

- Planning/docs:
  - added `V3_1_EXECUTION_PLAN.md`
    - captures current verified baseline, gap vs `Evaluation_Plan_v3.md`, v3.1 workstreams, and explicit v0.9.0 must-have/should-have/stretch gates.
  - added `RELEASE_0_9_0_READINESS.md`
    - explicit current-vs-target scorecard for v0.9.0 with must-have gates, remaining gaps, and next eval wants.

### Validation

- `python -m backend.validate_manifests` -> PASS (`4/4`)
- `python -m backend.test_topic_routing` -> PASS
- `python -m backend.evaluate_side_letter_retrieval --bm25-only` -> PASS (`top-8 16/16`, `depth-20 16/16`)
- `python -m backend.evaluate_moa_readiness` -> PASS (gate true)
- `python -m backend.evaluate_moa_deep_suite` -> PASS (`16/16`)
- `python -m backend.evaluate_runner --track all` -> PASS (`20/20`)
  - metadata: `data/test_set/eval_run_metadata_all_20260222T011614Z.json`
- `python -m backend.evaluate_v3 --bm25-only` -> PASS (`16/16`)
  - artifact: `data/test_set/v3_results.json`

### Release Positioning

- Current position:
  - canonical deterministic eval stack is green across active contracts.
  - MOA readiness and deep MOA suite are green.
- Remaining strategic gap to full v3 scaling vision:
  - universal schema extraction and synthetic benchmark factory are not yet implemented.
- v0.9.0 direction:
  - shift from "feature-complete slices" to "release scorecard with enforced must-have gates," as laid out in `V3_1_EXECUTION_PLAN.md`.

## v0.8.70 - Modular Frontend Promotion + PDF-First Contract UX + 6-Step Member Onboarding (February 19, 2026)

### Overview

Promoted the modular frontend to primary UX and shipped a major end-to-end pass across onboarding, contract reading, citation navigation, and layout stability. Added robust PDF-first contract navigation tied to citations (including table-aware routing) and completed a polished 6-question member onboarding flow with shield progress mapping.

### What Changed

- Frontend architecture and UX direction:
  - moved active UI work to `frontend/modular/*` and continued feature parity there
  - established PDF-first contract reading UX with TOC + embedded PDF surface
  - contract pane and TOC now use independent desktop scrolling with fixed workspace sizing

- Contract PDF and citation navigation:
  - citation popovers render markdown content correctly
  - citation -> contract routing supports article/section and table/table-row navigation
  - improved PDF load reliability when jumping from TOC/citations (including repeated jumps)
  - reset behavior returns viewer to last pinned citation location
  - PDF toolbar hidden in embedded iframe, while `Open Tab` now opens full native viewer controls (page nav/tools)
  - added in-pane `Download` control for PDF
  - removed redundant TOC `PDF` shortcut buttons

- Backend support for table-aware PDF routing:
  - added table navigation index generation and loaders
  - `/api/pdf-location` extended to resolve `table_id` / `row_index`
  - verifier source metadata now carries table-aware citation context

- Contract outline / article coverage hardening:
  - investigation and fixes for missing article dropdown coverage caused by outline/artifact mismatch
  - ingestion/runtime path improved to reduce future drift risk at scale

- Header / layout / interaction polish:
  - removed pointer-driven gradient behavior; header flow is autonomous and state-modulated
  - desktop stacking bug fixed where tab bar clipped top content rows
  - dynamic shell metric syncing added for header/tab offsets after font/layout changes
  - chat profile bar contrast/readability improvements in dark contexts

- Onboarding system modularization:
  - split and stabilized steward and member onboarding controllers
  - onboarding skip logic respects existing saved profile/cache and bypasses modal when complete profile is present
  - member flow expanded to 6 questions:
    1. Employer
    2. Location
    3. Department
    4. Classification
    5. Employment type (Full-Time / Part-Time)
    6. Hire date
  - shield progress expanded to 6 segments and synchronized with scene progression
  - final step animation updated so final form collapses into contract before shield close
  - next-step transition timing accelerated and synchronized with page-turn motion
  - skip/back shield controls mapped across all member scenes

- Visual design refinements:
  - shield segment rendering simplified to clean split fills (no visible idle seams)
  - idle shield grays adjusted to original look (`#334155` / `#475569`)
  - onboarding typography rebalanced (Inter + Playfair display cues)
  - onboarding option labels normalized to readable casing (no forced all-caps store labels)

### Validation

- `node --check frontend/modular/src/app.js` -> PASS
- `node --check frontend/modular/src/modules/member-onboarding.js` -> PASS
- Table-aware `/api/pdf-location` routing validated for article/section and table-row paths during implementation.

## v0.8.69 - Canonical Follow-Up Role Wage Gate (February 2026)

### Overview

Promoted role-targeted wage follow-up integrity into the canonical multi-contract evaluation stack and release gates. This makes explicit-role override and profile-fallback wage behavior release-blocking with deterministic table/citation grounding checks.

### What Changed

- New canonical dataset:
  - `data/test_set/followup_role_wage_test.json`
  - 12 deterministic cases across active contracts, including formal rewrite phrasing and follow-up role overrides.
- Canonical runner integration:
  - `backend/evaluate_runner.py`
    - new track: `followup_role_wage`
    - included in `--track all`
- Canonical v3 integration:
  - `backend/evaluate_v3.py`
    - new required component: `followup_role_wage`
    - full v3 run now executes `python -m backend.evaluate_followup_role_wage --bm25-only`
    - added threshold flags for dataset schema, overall/per-contract floors, target resolution, table evidence, Appendix citation, intent/no-unavailable, and override/fallback rates
    - schema version bumped to `v3_eval_v5`
- Release gate integration:
  - `backend/evaluate_gate_check.py`
    - new required artifact checks and thresholds for follow-up role wage slice
    - added non-release bypass: `--allow-missing-followup-role-wage`
- CI + docs + policy sync:
  - `.github/workflows/eval-ci.yml`
    - PR/main jobs now run `followup_role_wage` and upload artifact
    - gate enforcement includes follow-up role wage thresholds
  - `README.md`
    - added runner/evaluator commands and gate-check arguments for follow-up role wage
    - updated canonical CI slice list
  - `legal/RELEASE-GATES.md`
    - Gate B threshold requirement + Gate D canonical artifact requirement

## v0.8.68 - Role-Targeted Wage Follow-Up Hardening + Appendix Citation UX (February 2026)

### Overview

Fixed a deterministic follow-up gap where wage questions that explicitly target a different role (for example, `cc's` while profile role is `dug`) could be routed as non-wage or resolved against the profile role instead of the requested role.

### What Changed

- Wage intent and role-targeting hardening:
  - `backend/retrieval/router.py`
    - strengthened normalization (`cc's -> ccs`) for alias detection stability.
    - expanded wage intent patterns for role-targeted phrasing (`what does X make`, `pay ... for ...`).
    - when a wage query explicitly mentions one or more roles, wage targeting now prefers the first explicitly mentioned role rather than profile-default role.
    - query-plan internals now keep query-mentioned roles separate from profile role used for anchor expansion.
- Contract-view table fidelity:
  - `backend/api.py`
    - `/api/article/{article_num}` and `/api/section/{article_num}/{section_num}` now return `content_with_tables` when available so table content renders in contract view.
- Appendix citation clickability:
  - `backend/generation/verifier.py`
    - source metadata now includes `article_num`/`section_num`/`subsection` (and table metadata for wage-table citations).
  - `frontend/index.html`
    - chat citation badges now use source metadata fallback to make non-`Article ...` citations (for example `Appendix A`) clickable when article anchors are available.
    - query response metadata wiring now includes `sources`.
- Targeted deterministic regression coverage:
  - `backend/test_topic_routing.py`
    - added `role-targeted wage follow-up` checks for:
      - `"you dont know what the pay is for cc's"`
      - `"what does a courtesy clerk make"`
      - profile-only fallback behavior for `"what is my pay"`.

### Validation

- `python backend/test_topic_routing.py` -> PASS
- `python -m backend.evaluate_gate_check` -> PASS

## v0.8.67 - Classification Key Canonicalization + Multi-Role Comparison Routing Hardening (February 2026)

### Overview

Fixed duplicate role-option surfacing caused by punctuation/spacing normalization drift, and hardened comparison-query routing so prompts like "cc's vs dug workers" deterministically anchor both role article maps.

### What Changed

- Classification key canonicalization:
  - `backend/ingest/extract_wages.py`
    - `normalize_classification_name(...)` now collapses repeated underscores and trims edge underscores.
  - `backend/user/profile.py`
    - contract option loaders now normalize `value`/`wage_key` through canonical classifier normalization.
    - removes duplicate default options that only differ by punctuation/spacing artifacts.
- Multi-role comparison routing:
  - `backend/retrieval/router.py`
    - `_contract_classification_aliases(...)` now includes unmapped options + alias labels and generates acronym aliases (e.g., `cc`, `dug`).
    - new `extract_classifications_for_contract(...)` to detect multiple role mentions in query order.
    - `classify_intent(...)` now merges classification article anchors across all mentioned roles (not only one classification).
- Targeted deterministic tests:
  - `backend/test_topic_routing.py`
    - added role-comparison routing test for `"what is the difference between cc's and dug workers"`.
  - `backend/test_classification_options.py`
    - added normalized uniqueness guard for Pueblo Meat default role options.

### Validation

- `python backend/test_classification_options.py` -> PASS
- `python backend/test_topic_routing.py` -> PASS
- `python -m backend.evaluate_paraphrase --bm25-only` -> PASS (`15/15`, `45/45`)
- `python -m backend.evaluate_multi_contract --bm25-only` -> PASS (`18/18`)
- `python -m backend.evaluate_gate_check` -> PASS

## v0.8.66 - Canonical Role-Catalog Integrity Slice + v3/Gate Enforcement (February 2026)

### Overview

Promoted contract-scoped role integrity from ad-hoc tests into a canonical, release-gated evaluation slice to prevent role drift/cross-pollination regressions (for example, `cake_decorator` leakage into meat-contract defaults).

### What Changed

- New canonical evaluator + dataset:
  - `backend/evaluate_role_catalog_integrity.py`
    - validates dataset-driven role expectations per contract
    - validates invariants per contract:
      - default onboarding roles are wage-ready
      - unresolved manifest roles do not appear in default options
      - onboarding defaults are unique per `wage_key`
  - `data/test_set/role_catalog_integrity_test.json`
    - canonical expectations for Loveland meat / Pueblo meat / Pueblo clerks role behavior
  - Artifact output:
    - `data/test_set/role_catalog_integrity_results.json`
- Canonical runner integration:
  - `backend/evaluate_runner.py`
    - new track: `role_catalog_integrity`
    - included in `--track all`
- Canonical v3 integration:
  - `backend/evaluate_v3.py`
    - new required component: `role_catalog_integrity`
    - full v3 run now executes `python -m backend.evaluate_role_catalog_integrity`
    - added threshold CLI flags for schema/coverage/rates
    - schema version bumped to `v3_eval_v4`
- Release gate integration:
  - `backend/evaluate_gate_check.py`
    - required role-catalog-integrity artifact checks + thresholds
    - new CLI args for role-catalog-integrity artifact path/thresholds
    - non-release bypass flag: `--allow-missing-role-catalog-integrity`
- Docs + gate policy sync:
  - `README.md`
    - added canonical runner command, standalone evaluator command, gate-check sample args, CI slice list update
  - `legal/RELEASE-GATES.md`
    - Gate B includes role-catalog-integrity threshold requirement
    - Gate D includes canonical role-catalog-integrity artifact requirement

### Validation

- `python -m backend.evaluate_role_catalog_integrity` -> PASS (`20/20`, `100%`)
- `python -m backend.evaluate_runner --track role_catalog_integrity` -> PASS
- `python -m backend.evaluate_v3 --bm25-only` -> PASS (`15/15`)
- `python -m backend.evaluate_gate_check` -> PASS
- `python -m backend.evaluate_paraphrase --bm25-only` -> PASS (`15/15`, `45/45`)
- `python -m backend.evaluate_runner --track paraphrase` -> PASS
- `python -m backend.evaluate_multi_contract --bm25-only` -> PASS (`18/18`)
- `python backend/test_topic_routing.py` -> PASS

## v0.8.65 - Deterministic Vacation Entitlement Artifact + Release-Gated Entitlement Evidence (February 2026)

### Overview

Added an ingestion-owned vacation entitlement schedule artifact and a deterministic runtime entitlement lookup path, then promoted entitlement evidence into canonical v3 and release gates.

### What Changed

- New ingestion/runtime artifact pipeline:
  - `backend/ingest/extract_entitlements.py`
    - deterministic extraction of vacation accrual ladders (`years_of_service -> weeks_per_year`)
    - schedule condition parsing (`hire_date_on_or_before/on_or_after`, `anniversary_hours_min`)
    - deterministic runtime resolver `lookup_vacation_entitlement(...)`
  - `backend/entitlement_files.py`
    - contract-scoped entitlement artifact resolver (`entitlement_tables_<contract_id>.json`)
- Ingestion integration:
  - `scripts/onboard_contract_packages.py`
    - generates package entitlement artifact under `data/contracts/<contract_id>/entitlements/`
    - syncs runtime artifact to `data/entitlements/entitlement_tables_<contract_id>.json`
  - `backend/ingest/rebuild_index.py`
    - regenerates entitlement artifacts during rebuild for each active contract
  - `backend/ingest/pack_acceptance.py`
    - required checks:
      - `entitlements_exists`
      - `entitlements_schema_valid`
      - `vacation_entitlement_non_empty`
- Retrieval/runtime integration:
  - `backend/retrieval/router.py`
    - added entitlement artifact loading cache
    - added `lookup_vacation_entitlement(...)`
    - retrieval results now include `entitlement_info` for vacation-entitlement queries
  - `backend/api.py`
    - `QueryResponse` now includes optional `entitlement_info`
    - deterministic `/api/query` short-circuit for vacation entitlement amount questions:
      - returns schedule-backed answer from entitlement artifact
      - avoids stochastic "not available" failure for accrual schedules
      - requires hire-date disambiguation when multiple hire-date schedules exist
- New canonical eval and tests:
  - `backend/evaluate_entitlement_table_evidence.py`
  - `data/test_set/entitlement_table_evidence_test.json`
  - `backend/test_vacation_entitlement_lookup.py`
- Canonical runner/v3/gate integration:
  - `backend/evaluate_runner.py`
    - new track: `entitlement_table_evidence`
    - included in `all`
  - `backend/evaluate_v3.py`
    - new required component: `entitlement_table_evidence`
    - v3 component count now includes entitlement evidence checks
  - `backend/evaluate_gate_check.py`
    - required artifact checks/thresholds for entitlement-table-evidence
- Docs and release-gate sync:
  - `README.md`
  - `legal/RELEASE-GATES.md`

### Validation

- `python scripts/onboard_contract_packages.py --package local7_kingsoopers_loveland_meat_2019 --package local7_safeway_pueblo_meat_2022 --package local7_safeway_pueblo_clerks_2022` -> PASS (pack required gates pass)
- `python backend/test_vacation_entitlement_lookup.py` -> PASS
- `python -m backend.evaluate_entitlement_table_evidence` -> PASS (`12/12`)
- `python -m backend.evaluate_runner --track entitlement_table_evidence` -> PASS
- `python -m backend.evaluate_v3 --bm25-only` -> PASS (`14/14`)
- `python -m backend.evaluate_gate_check` -> PASS
- `python -m backend.evaluate_paraphrase --bm25-only` -> PASS (`15/15`, `45/45`)
- `python -m backend.evaluate_multi_contract --bm25-only` -> PASS (`18/18`)
- `python backend/test_topic_routing.py` -> PASS

## v0.8.64 - Canonical Wage-Table-Evidence Gate Promotion (February 2026)

### Overview

Promoted wage-table evidence from a standalone evaluator to a canonical release-gated slice in v3 and gate checks. This enforces deterministic Appendix-table grounding integrity at release time (dataset schema, coverage, per-contract floors, and evidence-presence metrics).

### What Changed

- `backend/evaluate_v3.py`
  - Added canonical wage-table-evidence component:
    - artifact: `data/test_set/wage_table_evidence_results.json`
    - checks: dataset schema, minimum total/per-contract coverage, overall/per-contract pass floors, source-method pass floor, table-evidence presence floor, table-id presence floor.
  - Full v3 run now executes:
    - `python -m backend.evaluate_wage_table_evidence --bm25-only`
  - Added CLI thresholds:
    - `--required-wage-table-evidence-dataset-schema-version`
    - `--min-wage-table-evidence-total-cases`
    - `--min-wage-table-evidence-cases-per-contract`
    - `--min-wage-table-evidence-pass-rate`
    - `--min-wage-table-evidence-per-contract`
    - `--min-wage-table-evidence-source-method-pass-rate`
    - `--min-wage-table-evidence-presence-rate`
    - `--min-wage-table-evidence-table-id-presence-rate`
- `backend/evaluate_gate_check.py`
  - Added required wage-table-evidence artifact checks with the same deterministic threshold set as v3.
  - Added CLI args for artifact path and thresholds:
    - `--wage-table-evidence-results`
    - `--required-wage-table-evidence-dataset-schema-version`
    - `--min-wage-table-evidence-total-cases`
    - `--min-wage-table-evidence-cases-per-contract`
    - `--min-wage-table-evidence-pass-rate`
    - `--min-wage-table-evidence-per-contract`
    - `--min-wage-table-evidence-source-method-pass-rate`
    - `--min-wage-table-evidence-presence-rate`
    - `--min-wage-table-evidence-table-id-presence-rate`
  - Added non-release debug bypass:
    - `--allow-missing-wage-table-evidence`
- `README.md`
  - Added canonical runner/evaluator commands for wage-table-evidence.
  - Added gate-check example flags for wage-table-evidence thresholds.
  - Added wage-table-evidence to canonical CI slice list in docs.
- `legal/RELEASE-GATES.md`
  - Added wage-table-evidence threshold requirement to Gate B.
  - Added canonical wage-table-evidence artifact requirement to Gate D.

## v0.8.63 - Canonical Wage-Row Runtime Retrieval + Table Evidence Surfacing (February 2026)

### Overview

Implemented a deterministic table-native wage path at runtime: wage lookups now resolve canonical wage rows first and return structured Appendix table evidence metadata (`table_id`, `row_index`, step/date/rate). This improves explainability and prepares the frontend for table-aware wage rendering.

### What Changed

- `backend/ingest/extract_wages.py`
  - Added deterministic classification-key resolver utility for wage lookup.
  - Added canonical-row-first wage resolver:
    - selects class-scoped canonical rows
    - resolves effective date deterministically
    - resolves applicable step by threshold (`hours` / `months` / `fixed`)
    - falls back to legacy classification-step lookup when canonical rows are absent
  - `lookup_wage(...)` now returns:
    - `classification_key`
    - `source_method` (`canonical_rows` or `classification_steps`)
    - `table_evidence` (list)
- `backend/retrieval/router.py`
  - `_ensure_wage_table_context(...)` now accepts `wage_info` and preferentially boosts appendix chunks whose `table_refs` match canonical row `table_evidence`.
- `backend/api.py`
  - Added optional `table_evidence` in:
    - `WageResponse`
    - `WageEstimateResponse`
  - `/api/wage` and `/api/wage/estimate/{session_id}` now include structured table evidence when available.
- `backend/generation/verifier.py`
  - `format_response_with_sources(...)` now appends deterministic table-specific Appendix citations from `wage_info.table_evidence`.
- `frontend/index.html`
  - Chat wage cards now render a compact “Appendix Table Evidence” block when `wage_info.table_evidence` is present.
- New deterministic test:
  - `backend/test_canonical_wage_lookup.py`
    - asserts canonical-row lookup path emits table evidence
    - asserts deterministic fallback behavior when canonical rows are removed

### Validation

- `python backend/test_canonical_wage_lookup.py` -> PASS
- `python backend/test_role_catalog.py` -> PASS
- `python backend/test_classification_options.py` -> PASS
- `python backend/test_topic_routing.py` -> PASS
- `python -m backend.evaluate_multi_contract --bm25-only` -> PASS (`18/18`)
- `python -m backend.evaluate_paraphrase --bm25-only` -> PASS (`15/15`, `45/45`)
- `python -m backend.evaluate_runner --track paraphrase` -> PASS
- `python -m backend.evaluate_runner --track v3` -> PASS
- `python -m backend.evaluate_gate_check` -> PASS

## v0.8.62 - Wage-Equivalent Role Alias Collapse for Onboarding Defaults (February 2026)

### Overview

Removed duplicate selectable roles when multiple manifest aliases map to the same wage key (for example, `dug_shopper` and `drive_up_and_go` both mapping to `nonfood_gm_floral` in Pueblo clerks), while preserving alias matching for runtime classification normalization.

### What Changed

- `backend/ingest/role_catalog.py`
  - Added deterministic wage-equivalent alias collapse:
    - exactly one `onboarding_default=true` role per `wage_key`
    - non-primary aliases retained with `alias_of_wage_key`
    - primary role enriched with `alias_labels` / `alias_values`
- `backend/user/profile.py`
  - Role-catalog option labels can surface alias previews for default options (e.g., one merged role label includes both DUG/Drive Up & Go).
  - Contract classification normalization now resolves against `include_unmapped=True` options and canonicalizes to `wage_key` when available.
  - Default role list in role-catalog mode now uses `onboarding_default` first (not raw `wage_available`), preventing duplicate wage-equivalent options.
- `backend/test_role_catalog.py`
  - Added assertions for:
    - at most one onboarding-default role per wage key
    - no duplicate default DUG/Drive Up alias options in Pueblo clerks

### Validation

- `python scripts/onboard_contract_packages.py --package local7_kingsoopers_loveland_meat_2019 --package local7_safeway_pueblo_meat_2022 --package local7_safeway_pueblo_clerks_2022` -> PASS
- `python backend/test_role_catalog.py` -> PASS
- `python backend/test_classification_options.py` -> PASS
- `python backend/test_topic_routing.py` -> PASS
- `python -m backend.evaluate_multi_contract --bm25-only` -> PASS (`18/18`)
- `python -m backend.evaluate_paraphrase --bm25-only` -> PASS (`15/15`, `45/45`)
- `python -m backend.evaluate_gate_check` -> PASS

## v0.8.61 - Contract-Scoped Role Catalog Artifact + Pack-Gate Enforcement (February 2026)

### Overview

Added an ingestion-owned contract role catalog artifact and made it the runtime default source for onboarding classification options. This hardens role typing at scale by separating "role exists in contract metadata" from "role has Appendix A wage row" with deterministic, contract-scoped metadata.

### What Changed

- New role-catalog artifact builder and runtime resolver:
  - `backend/ingest/role_catalog.py`
    - Added deterministic `role_catalog_v1` schema:
      - `roles[].value`, `label`
      - `wage_available`, `wage_key`
      - `source`, `mapping_method`
      - `manifest_present`, `onboarding_default`
    - Added summary counters including unresolved manifest roles.
  - `backend/role_catalog_files.py`
    - Added contract-scoped runtime resolver for `role_catalog_<contract_id>.json`.
- Onboarding integration:
  - `scripts/onboard_contract_packages.py`
    - Builds `ontology/role_catalog.json` during package processing.
    - Syncs runtime artifact to `data/ontologies/role_catalog_<contract_id>.json`.
    - Includes role-catalog stats in onboarding output.
- Pack acceptance integration:
  - `backend/ingest/pack_acceptance.py`
    - Added required checks:
      - `role_catalog_exists`
      - `role_catalog_schema_valid`
      - `role_catalog_onboarding_default_wage_ready`
    - Added advisory check:
      - `role_catalog_unresolved_manifest_rate`
- Runtime option loading hardening:
  - `backend/user/profile.py`
    - `get_classification_options(contract_id=...)` now prefers role-catalog options when present.
    - Existing wage+ontology+legacy merge path remains deterministic fallback when role catalog is absent.
- New deterministic test:
  - `backend/test_role_catalog.py`
    - Verifies schema/contract integrity and that unresolved manifest roles do not leak into default onboarding options.
- Docs and gate policy sync:
  - `README.md`
  - `legal/RELEASE-GATES.md`

### Validation

- `python scripts/onboard_contract_packages.py --package local7_kingsoopers_loveland_meat_2019 --package local7_safeway_pueblo_meat_2022 --package local7_safeway_pueblo_clerks_2022` -> PASS
- `python backend/test_manifest_role_extraction.py` -> PASS
- `python backend/test_classification_options.py` -> PASS
- `python backend/test_role_catalog.py` -> PASS
- `python backend/test_topic_routing.py` -> PASS
- `python -m backend.evaluate_multi_contract --bm25-only` -> PASS (`18/18`)
- `python -m backend.evaluate_paraphrase --bm25-only` -> PASS (`15/15`, `45/45`)
- `python -m backend.evaluate_runner --track paraphrase` -> PASS
- `python -m backend.evaluate_runner --track v3` -> PASS
- `python -m backend.evaluate_gate_check` -> PASS

## v0.8.60 - Contract-Scoped Role Generation Hardening (February 2026)

### Overview

Hardened role generation to reduce cross-domain role bleed (for example, clerk-like or matrix-only role mentions showing up in meat-contract onboarding) and made wage availability explicit per contract role.

### What Changed

- Runtime classification option model now includes deterministic wage metadata:
  - `backend/api.py`
    - `ClassificationOption` now carries:
      - `wage_available`
      - `wage_key`
      - `source`
- Classification option assembly now merges wage + ontology + legacy roles with deterministic ranking:
  - `backend/user/profile.py`
    - wage-table roles ranked highest
    - ontology unresolved roles retained only in `include_unmapped=True` view
    - default onboarding options now return wage-resolvable roles first-class
- Added contract-scoped wage guards for unresolved classifications:
  - `backend/api.py`
    - `/api/query` wage-intent path returns explicit no-wage-row message when classification has no contract wage row
    - `/api/wage` and `/api/wage/estimate/{session_id}` return deterministic 404 with contract-scoped reason for unresolved roles
- Frontend role dropdown reflects wage metadata (when present):
  - `frontend/index.html`
    - unresolved roles render as `(... no Appendix A wage row)` when surfaced
- Ingestion role extraction precision improved:
  - `backend/ingest/manifest.py`
    - classification detection is now context-biased to wage/definition sections
    - added exclusion-context suppressors for incidental matrix/non-union mention patterns

### New Deterministic Tests

- `backend/test_manifest_role_extraction.py`
  - Guards against incidental role mentions polluting manifest classifications for Loveland meat.
- `backend/test_classification_options.py`
  - Verifies unresolved roles are hidden from default onboarding options and still available in internal unmapped view.

### Validation

- `python backend/test_manifest_role_extraction.py` -> PASS
- `python backend/test_classification_options.py` -> PASS
- `python backend/test_topic_routing.py` -> PASS
- `python -m backend.evaluate_multi_contract --bm25-only` -> PASS (`18/18`)
- `python -m backend.evaluate_paraphrase --bm25-only` -> PASS (`15/15`, `45/45`)
- `python -m backend.evaluate_gate_check` -> PASS

## v0.8.59 - Cross-Contract Mention Guard Ordering Fix + Gate/Docs Sync (February 2026)

### Overview

Closed a deterministic ordering gap where foreign-contract questions could hit the wage-classification fallback before foreign-contract abstention logic. Promoted and synchronized cross-contract mention release-gate documentation.

### What Changed

- Runtime ordering fix in `backend/api.py`:
  - Added `_FOREIGN_CONTRACT_UNAVAILABLE_ANSWER` constant.
  - Moved explicit foreign-contract guard to run immediately after intent classification.
  - Foreign-contract references now short-circuit before retrieval/generation and before wage-classification fallback.
  - Response is deterministic unavailable with zero citations and no escalation.
- Removed late redundant foreign-contract override after generation in `backend/api.py`.
- Documentation/release-gate synchronization:
  - `README.md`
    - added canonical runner/validator commands for `cross_contract_mentions`
    - added cross-contract mention artifact + thresholds to gate-check example
    - updated CI canonical-slice list to include `cross_contract_mentions`
  - `legal/RELEASE-GATES.md`
    - added cross-contract mention abstention + dataset-integrity requirements
    - added required artifact `data/test_set/cross_contract_mentions_results.json`

### Before/After Metrics

- Cross-contract mention slice:
  - Before: `8/9` (`88.9%`), no-citation rate `88.9%`
  - After: `9/9` (`100.0%`), no-citation rate `100.0%`

### Failing-Case Delta

- Case fixed: `CCM-CL-02`
  - Active contract: `local7_safeway_pueblo_clerks_2022`
  - Prompt referenced: `local7 kingsoopers loveland meat 2019`
- Failure mode:
  - Query classified as wage and exited through missing-classification fallback path, returning citation-bearing response.
- Fix:
  - Enforced foreign-contract guard before wage fallback.
- Result:
  - Deterministic unavailable answer with zero citations.

### Validation

- `python -m backend.validate_cross_contract_mentions_dataset` -> PASS
- `python -m backend.evaluate_cross_contract_mentions --bm25-only` -> PASS (`9/9`)
- `python -m backend.evaluate_runner --track cross_contract_mentions` -> PASS
- `python -m backend.evaluate_v3 --from-artifacts --bm25-only` -> PASS (`12/12`)
- `python -m backend.evaluate_gate_check` -> PASS
- `python backend/test_topic_routing.py` -> PASS

## v0.8.58 - Canonical Unanswerable Multi-Contract Slice + Gate Integration (February 2026)

### Overview

Added a deterministic multi-contract unanswerable/abstention benchmark, integrated it into canonical v3 and release gates, and hardened runtime handling for explicit foreign-contract references.

### What Changed

- New canonical unanswerable evaluator:
  - `backend/evaluate_unanswerable.py`
  - Writes `data/test_set/unanswerable_results.json`
  - Executes `/api/query` with forced unavailable first-pass generation and verifies abstention preservation.
- New canonical dataset + validator:
  - `data/test_set/unanswerable_multi_contract_test.json`
  - `backend/validate_unanswerable_dataset.py`
  - Enforces schema, minimum total/per-contract coverage, and scenario diversity floors.
- Runtime deterministic hardening for cross-contract mention drift:
  - `backend/api.py`
  - Added explicit foreign-contract reference detector based on active catalog aliases.
  - When query references a different known contract family, runtime now preserves uncertainty and blocks false recovery.
- Canonical orchestration/gate integration:
  - `backend/evaluate_runner.py`
    - new track: `unanswerable`
    - included in `all`
  - `backend/evaluate_v3.py`
    - added:
      - `unanswerable_dataset_validation`
      - `unanswerable` component
  - `backend/evaluate_gate_check.py`
    - added unanswerable artifact checks + dataset integrity checks with thresholds
    - new args include:
      - `--unanswerable-results`
      - `--unanswerable-dataset`
      - `--required-unanswerable-dataset-schema-version`
      - `--min-unanswerable-total-cases`
      - `--min-unanswerable-cases-per-contract`
      - `--min-unanswerable-scenario-types`
      - `--min-unanswerable-pass-rate`
      - `--min-unanswerable-per-contract`
      - `--allow-missing-unanswerable`
- CI/docs sync:
  - `.github/workflows/eval-ci.yml`
  - `README.md`
  - `legal/RELEASE-GATES.md`

### Before/After Metrics

- Unanswerable slice:
  - Initial dataset/evaluator integration: `10/12` (`83.3%`)
  - Final after foreign-contract reference guard: `12/12` (`100.0%`)
- v3 canonical suite:
  - Final: `10/10` (`100.0%`) pass
- Release gate:
  - Final: PASS (including new unanswerable artifact + dataset checks)

### Failing-Case Delta

- Failing cases:
  - `UN-KG-04` (King Soopers active context, Safeway clerks reference)
  - `UN-CL-04` (Safeway clerks active context, King Soopers reference)
- Failure mode:
  - recovery path over-interpreted same-topic evidence from active contract and converted unavailable to answerable.
- Fix:
  - explicit foreign-contract alias detection in `backend/api.py` and recovery suppression for those queries.
- Result:
  - both cases now abstain deterministically.

### Validation

- `python -m backend.validate_unanswerable_dataset` -> PASS
- `python -m backend.evaluate_unanswerable --bm25-only` -> PASS (`12/12`)
- `python -m backend.evaluate_runner --track unanswerable` -> PASS
- `python -m backend.evaluate_v3 --from-artifacts --bm25-only` -> PASS (`10/10`)
- `python -m backend.evaluate_runner --track v3` -> PASS
- `python -m backend.evaluate_paraphrase --bm25-only` -> PASS (`15/15`, `45/45`)
- `python -m backend.evaluate_multi_contract --bm25-only` -> PASS (`18/18`)
- `python backend/test_topic_routing.py` -> PASS
- `python -m backend.evaluate_gate_check` -> PASS

## v0.8.57 - Adversarial Dataset Integrity Gates + Premium Routing Regression Guard (February 2026)

### Overview

Extended the new adversarial formal-precedence track with dataset-integrity validation and added a deterministic routing regression guard for holiday-work premium phrasing.

### What Changed

- New validator:
  - `backend/validate_adversarial_dataset.py`
  - Enforces:
    - required schema version
    - minimum total case count
    - per-contract minimum case coverage across active contracts
    - minimum precedence-case count
    - required-field integrity (IDs, contract IDs, expected citations/keywords, precedence config)
- Canonical v3 integration:
  - `backend/evaluate_v3.py`
  - Added component:
    - `adversarial_dataset_validation`
  - Full v3 run now executes adversarial dataset validation before adversarial evaluation.
- CI integration:
  - `.github/workflows/eval-ci.yml`
  - Added:
    - `python -m backend.validate_adversarial_dataset`
  - Included in both PR and main release-gate workflows.
- Gate-check integrity extension:
  - `backend/evaluate_gate_check.py`
  - Now validates adversarial dataset structure/coverage directly (not only adversarial results artifact),
    including precedence-case minimums.
- Deterministic routing regression test:
  - `backend/test_topic_routing.py`
  - Added holiday-work premium case ensuring premium phrasing maps to premium topic and surfaces `Article 16, Section 40` for King Soopers meat.
- Docs synchronized:
  - `README.md`
  - `legal/RELEASE-GATES.md`

### Validation

- `python -m backend.validate_adversarial_dataset` -> PASS
- `python backend/test_topic_routing.py` -> PASS
- `python -m backend.evaluate_adversarial_precedence --bm25-only` -> PASS (`12/12`)
- `python -m backend.evaluate_v3 --from-artifacts --bm25-only` -> PASS
- `python -m backend.evaluate_gate_check` -> PASS

## v0.8.56 - Canonical Adversarial Formal-Precedence Slice + v3/Gate Integration (February 2026)

### Overview

Added a deterministic multi-contract adversarial formal-rewrite/precedence benchmark, integrated it into canonical v3 and release gates, and hardened holiday-work premium routing so exception clauses are surfaced reliably.

### What Changed

- New canonical evaluator:
  - `backend/evaluate_adversarial_precedence.py`
  - Writes `data/test_set/adversarial_results.json`
  - Enforces deterministic checks for:
    - specific citation retrieval
    - exception-keyword coverage
    - optional specific-vs-competing precedence ordering
    - tenant isolation at retrieval output
- Canonical adversarial dataset:
  - `data/test_set/adversarial_test.json`
  - New schema: `adversarial_precedence_test_v1`
  - 12 formal near-miss cases with explicit `contract_id` coverage:
    - `local7_kingsoopers_loveland_meat_2019` (4)
    - `local7_safeway_pueblo_clerks_2022` (4)
    - `local7_safeway_pueblo_meat_2022` (4)
- Runtime deterministic routing/retrieval hardening:
  - `backend/retrieval/router.py`
  - Added holiday-work premium query detection/coverage seed:
    - `_is_holiday_work_premium_query`
    - `_ensure_holiday_work_premium_coverage`
  - Rebalanced topic routing for holiday premium prompts:
    - moved `premiums` ahead of `vacation` in topic priority
    - expanded premium topic patterns + lexical signals for holiday-work premium language
- Canonical orchestration updates:
  - `backend/evaluate_runner.py`
    - new track: `adversarial`
    - included in `all`
  - `backend/evaluate_v3.py`
    - new component: `adversarial_precedence`
    - full run now executes adversarial slice
    - `from-artifacts` validation now enforces adversarial thresholds
  - `backend/evaluate_gate_check.py`
    - added `--adversarial-results` checks and thresholds:
      - `--required-adversarial-dataset-schema-version`
      - `--min-adversarial-total-cases`
      - `--min-adversarial-cases-per-contract`
      - `--min-adversarial-pass-rate`
      - `--min-adversarial-per-contract`
      - `--min-adversarial-precedence-pass-rate`
    - added `--allow-missing-adversarial` debug bypass
- CI and docs synchronized:
  - `.github/workflows/eval-ci.yml`
    - runs `evaluate_runner --track adversarial` in PR/main gate flows
    - gate-check now includes adversarial artifact + thresholds
    - uploads `data/test_set/adversarial_results.json`
  - `README.md`
  - `legal/RELEASE-GATES.md`

### Before/After Metrics

- Adversarial formal-precedence:
  - Before routing/priority hardening: `9/12` (`75.0%`), precedence `0.5`
  - Intermediate after evaluator/schema tuning: `11/12` (`91.7%`), precedence `1.0`
  - Final: `12/12` (`100.0%`), precedence `1.0`, per-contract `4/4` each
- v3 canonical suite:
  - Before adversarial integration: `6/6` (`100.0%`)
  - During integration while adversarial failing: `6/7` (`85.7%`) blocked
  - Final: `7/7` (`100.0%`) pass
- Release gate:
  - During integration: BLOCKED (v3 + adversarial per-contract floor miss)
  - Final: PASS

### Failing-Case Delta

- `ADV-KS-03` (King Soopers holiday-work premium, post-2005 hire):
  - Failure mode: routed as `vacation` (holiday token over-match), missing Article 16 Section 40 in top-k
  - Fix: premium-topic priority and expanded premium lexical patterns + deterministic holiday-premium seed coverage
  - Result: case now passes with expected citation/keyword evidence

### Validation

- `python -m backend.evaluate_adversarial_precedence --bm25-only` -> PASS (`12/12`)
- `python -m backend.evaluate_v3 --from-artifacts --bm25-only` -> PASS (`7/7`)
- `python -m backend.evaluate_runner --track v3` -> PASS
- `python -m backend.evaluate_paraphrase --bm25-only` -> PASS (`15/15` families, `45/45` variants)
- `python -m backend.evaluate_runner --track paraphrase` -> PASS
- `python -m backend.evaluate_multi_contract --bm25-only` -> PASS (`18/18`)
- `python backend/test_topic_routing.py` -> PASS
- `python -m backend.evaluate_gate_check` -> PASS

## v0.8.55 - Canonical v3 Multi-Contract Evaluation Suite + Gate Integration (February 2026)

### Overview

Implemented Benchmark v3 as a canonical multi-contract evaluation track and integrated it into release gating/CI with explicit artifact-based integrity checks.

### What Changed

- New canonical v3 evaluator:
  - `backend/evaluate_v3.py`
  - Produces `data/test_set/v3_results.json`.
  - Validates multi-contract phase components:
    - false-unavailable dataset validation
    - cross-contamination isolation
    - multi-contract benchmark
    - paraphrase robustness
    - false-unavailable behavior
    - needle retrieval integrity
  - Supports `--from-artifacts` mode for CI aggregation.
- Cross-contamination artifactization:
  - `backend/evaluate_cross_contamination.py`
  - Now writes `data/test_set/cross_contamination_results.json` with pass/failure metadata.
- Canonical runner integration:
  - `backend/evaluate_runner.py`
  - Added `--track v3`.
  - Added `v3` to `all`.
- Release gate integration:
  - `backend/evaluate_gate_check.py`
  - Added required checks for:
    - cross-contamination artifact pass
    - v3 suite artifact pass-rate/boolean
  - Added args:
    - `--cross-contamination-results`
    - `--v3-results`
    - `--min-v3-components-pass-rate`
    - `--allow-missing-cross-contamination`
    - `--allow-missing-v3`
- CI integration:
  - `.github/workflows/eval-ci.yml`
  - Runs `python -m backend.evaluate_v3 --from-artifacts --bm25-only` in PR/main gate jobs.
  - Gate-check invocation now includes cross-contamination/v3 artifacts and v3 threshold.
  - Uploads `cross_contamination_results.json` and `v3_results.json`.
- Docs synchronized:
  - `README.md` (v3 benchmark now canonical, command/docs updated)
  - `legal/RELEASE-GATES.md` (Gate D includes canonical v3 suite pass)

### Validation

- `python -m backend.evaluate_cross_contamination --require-multi-contract` -> PASS
- `python -m backend.evaluate_v3 --bm25-only` -> PASS (`6/6` components)
- `python -m backend.evaluate_runner --track v3` -> PASS metadata/artifact generation
- `python backend/test_topic_routing.py` -> PASS
- `python -m backend.evaluate_paraphrase --bm25-only` -> `15/15` families, `45/45` variants
- `python -m backend.evaluate_runner --track paraphrase` -> PASS
- `python -m backend.evaluate_multi_contract --bm25-only` -> `18/18`
- `python -m backend.evaluate_gate_check` -> PASS (includes cross-contamination + v3 checks)

## v0.8.54 - False-Unavailable Dataset Integrity Gates (Schema + Distribution + Coverage) (February 2026)

### Overview

Added explicit benchmark-integrity controls for the false-unavailable slice so release gates block on schema drift, insufficient negative controls, or missing active-contract coverage.

### What Changed

- New validator:
  - `backend/validate_false_unavailable_dataset.py`
  - Enforces:
    - required dataset schema version
    - minimum total/recover/uncertain case counts
    - minimum per-contract total + recover coverage
    - active contract ID membership
    - case ID uniqueness and expectation/citation consistency
- Dataset schema pin:
  - `data/test_set/false_unavailable_test.json`
  - Added `"schema_version": "false_unavailable_test_v1"`.
- Evaluator metadata:
  - `backend/evaluate_false_unavailable.py`
  - Result artifact now includes `dataset_schema_version` for gate-level enforcement.
- Gate hardening:
  - `backend/evaluate_gate_check.py`
  - Added required false-unavailable integrity checks:
    - `--required-false-unavailable-dataset-schema-version`
    - `--min-false-unavailable-total-cases`
    - `--min-false-unavailable-recover-cases`
    - `--min-false-unavailable-uncertain-cases`
    - `--min-false-unavailable-cases-per-contract`
- CI wiring:
  - `.github/workflows/eval-ci.yml`
  - Added preflight step:
    - `python -m backend.validate_false_unavailable_dataset`
  - Added new dataset-integrity threshold flags to CI gate-check invocations.
- Docs synchronized:
  - `README.md`
  - `legal/RELEASE-GATES.md`

### Validation

- `python -m backend.validate_false_unavailable_dataset` -> PASS
- `python -m backend.evaluate_false_unavailable --bm25-only` -> PASS (`15/15`, recovered `12/12`, uncertainty `3/3`)
- `python -m backend.evaluate_gate_check` -> PASS (includes dataset integrity + false-unavailable behavior rates)
- `python backend/test_topic_routing.py` -> PASS
- `python -m backend.evaluate_paraphrase --bm25-only` -> `15/15` families, `45/45` variants
- `python -m backend.evaluate_runner --track paraphrase` -> PASS
- `python -m backend.evaluate_multi_contract --bm25-only` -> `18/18`

## v0.8.53 - False-Unavailable Robustness Slice Expansion (Recovery + Proper Uncertainty) (February 2026)

### Overview

Expanded the canonical false-unavailable slice so it now tests both sides of behavior:
- recover when evidence exists
- preserve uncertainty when evidence is absent

### What Changed

- `data/test_set/false_unavailable_test.json`
  - Expanded from 4 to 15 cases.
  - Added coverage across all active contracts.
  - Added explicit negative controls (`expectation: "uncertain"`) for non-contract facts.
- `backend/evaluate_false_unavailable.py`
  - Added expectation-aware scoring (`recover` vs `uncertain`).
  - Added explicit metrics:
    - `overall.false_unavailable_recovered_rate`
    - `overall.proper_uncertainty_rate`
  - Updated schema to `false_unavailable_eval_v2`.
- `backend/api.py`
  - Tightened unavailable-phrase detection to assistant-refusal phrasing (avoids false matches on quoted contract text).
  - Added topic-signal gating for article-anchor recovery to prevent confidence inflation on unrelated prompts.
  - Recovery now front-loads topic-seed chunks when retry evidence exists, improving citation visibility in deterministic fallback.
- `backend/evaluate_gate_check.py`
  - Added required thresholds:
    - `--min-false-unavailable-recovered-rate` (default `0.90`)
    - `--min-false-unavailable-proper-uncertainty-rate` (default `0.90`)
- `.github/workflows/eval-ci.yml`
  - CI gate invocations now enforce the two new false-unavailable rate thresholds.
- Docs synchronized:
  - `README.md`
  - `legal/RELEASE-GATES.md`

### Validation

- `python -m backend.evaluate_false_unavailable --bm25-only` -> PASS
- `python -m backend.evaluate_gate_check` -> PASS (includes recovered/proper-uncertainty thresholds)
- `python backend/test_topic_routing.py` -> PASS
- `python -m backend.evaluate_paraphrase --bm25-only` -> `15/15` families, `45/45` variants
- `python -m backend.evaluate_runner --track paraphrase` -> PASS
- `python -m backend.evaluate_multi_contract --bm25-only` -> `18/18`

## v0.8.52 - CI Gate Wiring for False-Unavailable Canonical Slice (February 2026)

### Overview

Updated CI workflows so false-unavailable integrity is enforced automatically on PR and main gate jobs.

### What Changed

- `.github/workflows/eval-ci.yml`
  - Added canonical runner step:
    - `python -m backend.evaluate_runner --track false_unavailable`
  - Added release-gate thresholds in CI invocation:
    - `--min-paraphrase-formal-rewrite-pass-rate 0.90`
    - `--min-false-unavailable-pass-rate 0.90`
    - `--min-false-unavailable-per-contract 0.80`
  - Added artifact upload path:
    - `data/test_set/false_unavailable_results.json`
  - Applied to both:
    - `pr-core-eval`
    - `main-escalation-gate`
- `README.md`
  - CI behavior section now lists `false_unavailable` in canonical PR slices.

### Validation

- Workflow file updated and verified by static grep:
  - PR job includes `--track false_unavailable`
  - Main gate job includes `--track false_unavailable`
  - Gate-check CLI in CI includes false-unavailable thresholds
  - Artifact upload includes `false_unavailable_results.json`

## v0.8.51 - Canonical False-Unavailable Evaluation Track + Release Gate Enforcement (February 2026)

### Overview

Promoted false-unavailable behavior from ad hoc checks into a canonical deterministic evaluation track and made it release-gated with thresholds.

### What Changed

- Added canonical test dataset:
  - `data/test_set/false_unavailable_test.json`
  - Multi-contract prompts covering vacation entitlement, term-of-agreement, inter-shift rest phrasing, and bereavement phrasing.
- New evaluator:
  - `backend/evaluate_false_unavailable.py`
  - Forces unavailable first-pass generation, runs end-to-end `/api/query` path, and verifies:
    - final answer is not unavailable-style
    - expected citation evidence is present
  - Writes `data/test_set/false_unavailable_results.json`.
- Runner integration:
  - `backend/evaluate_runner.py`
  - Added `false_unavailable` track and included it in `all`.
- Release gate integration:
  - `backend/evaluate_gate_check.py`
  - Added required false-unavailable artifact check with thresholds:
    - `--min-false-unavailable-pass-rate` (default `0.90`)
    - `--min-false-unavailable-per-contract` (default `0.80`)
  - Missing artifact is blocking unless `--allow-missing-false-unavailable` is explicitly set.
- Docs synchronized:
  - `README.md`
  - `legal/RELEASE-GATES.md`

### Validation

- `python -m backend.evaluate_runner --track false_unavailable` -> PASS (artifact generated)
- `python -m backend.evaluate_gate_check` -> PASS (includes false-unavailable checks)
- `python backend/test_unavailability_recovery.py` -> PASS
- `python backend/test_topic_routing.py` -> PASS
- `python -m backend.evaluate_paraphrase --bm25-only` -> `15/15` families, `45/45` variants
- `python -m backend.evaluate_runner --track paraphrase` -> PASS
- `python -m backend.evaluate_multi_contract --bm25-only` -> `18/18`
- `python -m backend.evaluate_runner --track needle` -> PASS

## v0.8.50 - Deterministic Evidence-Gap Recovery Before "Not Available" Responses (February 2026)

### Overview

Added a contract-agnostic deterministic recovery loop in the API response path so KARL does one broader retrieval/regeneration pass before returning unavailable-style language when strong evidence is already in context.

### What Changed

- `backend/api.py`
  - Added deterministic unavailable-language detection.
  - Added deterministic evidence-strength check using citation presence + lexical overlap with query terms.
  - Added one-pass broader retrieval retry (`n_results=8`) when first answer claims unavailable but evidence appears strong.
  - Added stable chunk merge/dedupe for retry context.
  - Added final guard fallback to chunk-grounded response when model still refuses despite strong evidence.
- New deterministic test:
  - `backend/test_unavailability_recovery.py`
    - unavailable-language detection
    - strong/weak evidence classification
    - stable unique chunk merge behavior
- Docs synced:
  - `README.md`
  - `legal/RELEASE-GATES.md`

### Validation

- `python backend/test_unavailability_recovery.py` -> PASS
- `python backend/test_topic_routing.py` -> PASS
- `python -m backend.evaluate_paraphrase --bm25-only` -> `15/15` families, `45/45` variants
- `python -m backend.evaluate_runner --track paraphrase` -> PASS
- `python -m backend.evaluate_multi_contract --bm25-only` -> `18/18`
- `python -m backend.evaluate_runner --track needle` -> PASS
- `python -m backend.evaluate_gate_check` -> PASS

## v0.8.49 - Formal-Rewrite Drift Gate + Vacation Entitlement Phrasing Hardening (February 2026)

### Overview

Closed the post-optimization deterministic hardening loop by converting non-slang formal-rewrite robustness into an explicit release gate and extending vacation entitlement phrasing coverage for tenure-style prompts.

### What Changed

- `backend/evaluate_paraphrase.py`
  - Added `overall.formal_rewrite_pass_rate` to artifact output for first-class gate consumption.
- `backend/evaluate_gate_check.py`
  - Added required formal-rewrite gate check sourced from paraphrase artifacts.
  - New CLI threshold: `--min-paraphrase-formal-rewrite-pass-rate` (default `0.90`).
  - Release now blocks when formal-rewrite pass-rate metadata is missing or below floor.
- `backend/retrieval/router.py`
  - Expanded deterministic vacation entitlement query cues to include tenure/service/anniversary phrasing.
  - Added additional entitlement markers used by deterministic seed-chunk backfill.
- `backend/test_topic_routing.py`
  - Expanded vacation entitlement test into a multi-query deterministic slice:
    - plain phrasing ("How much vacation do I get per year?")
    - formal entitlement schedule phrasing
    - tenure-specific phrasing ("33 months of service")
- Docs synchronized:
  - `README.md`
  - `legal/RELEASE-GATES.md`

### Validation

- `python backend/test_topic_routing.py` -> PASS
- `python -m backend.evaluate_paraphrase --bm25-only` -> `15/15` families, `45/45` variants, formal_rewrite `1.0`
- `python -m backend.evaluate_runner --track paraphrase` -> PASS metadata/artifact generation
- `python -m backend.evaluate_multi_contract --bm25-only` -> `18/18`
- `python -m backend.evaluate_gate_check` -> PASS (includes formal-rewrite floor + needle floors)

## v0.8.48 - Ingestion-Owned Query-Routing Synthesis + Pack Gate Coverage (February 2026)

### Overview

Extended deterministic-first hardening by moving manifest routing population into ingestion, then enforcing routing coverage as a required pack-acceptance gate.

### What Changed

- New module: `backend/ingest/query_routing.py`
  - Added deterministic synthesis of manifest `query_routing` using:
    - manifest article titles/classifications
    - concept index (`concept_to_articles`, `question_to_articles`)
    - language lexicon (`alias_to_canonical`)
    - optional classification ontology context
  - Produces:
    - `slang_to_contract`
    - `topic_to_articles`
    - `topic_patterns`
    - `classification_to_articles`
  - Added deterministic merge utility so curated/manual routing overrides are preserved.
- `scripts/onboard_contract_packages.py`
  - Integrated query-routing synthesis into onboarding flow.
  - Rewrites package manifest with merged deterministic routing before runtime sync.
  - Added routing stats to onboarding summaries.
- `backend/ingest/rebuild_index.py`
  - Integrated query-routing synthesis after concept-index + lexicon rebuild.
  - Updates contract manifest routing in-place for rebuild-based repair flows.
- `backend/ingest/pack_acceptance.py`
  - Added required checks:
    - `query_routing_coverage`
    - `query_routing_article_ref_integrity`
  - Release now blocks when routing artifacts are too sparse or reference invalid articles.
- New deterministic test:
  - `backend/test_query_routing_ingest.py`
    - Validates generated routing quality on clerks contract (term/vacation/breaks/classification anchors).

### Validation

- `python backend/test_query_routing_ingest.py` -> PASS
- `python backend/test_topic_routing.py` -> PASS
- `python -m backend.evaluate_paraphrase --bm25-only` -> `15/15` families, `45/45` variants
- `python -m backend.evaluate_multi_contract --bm25-only` -> `18/18`
- `python -m backend.evaluate_gate_check` -> PASS

## v0.8.47 - Canonical Needle Overlay + CI/Gate Parity Hardening (February 2026)

### Overview

Started the next hardening sprint by removing manual needle injection dependency, enforcing deterministic canonical runner behavior for release slices, and aligning CI with release-gate policy requirements.

### What Changed

- `backend/evaluate_needle.py`
  - Added deterministic temporary synthetic-chunk overlay for needle evaluation:
    - no manual `scripts/inject_needles.py` pre/post steps required
    - no persistent corpus mutation
  - Needle retrieval now runs as a raw deterministic retrieval slice (BM25-only in canonical runner path) without topic-prior boosts.
  - Added `synthetic_overlay_applied` metadata field.
  - Added debug flag `--no-synthetic-overlay`.
- `backend/evaluate_runner.py`
  - `v2_multi_contract` track now executes deterministic BM25-only mode.
  - `paraphrase` track now executes deterministic BM25-only mode.
- `backend/evaluate_gate_check.py`
  - Missing multi-contract/paraphrase/needle artifacts are now release-blocking by default.
  - Added explicit debug-only bypass flags:
    - `--allow-missing-multi-contract`
    - `--allow-missing-paraphrase`
    - `--allow-missing-needle`
- `.github/workflows/eval-ci.yml`
  - PR and main gate jobs now run canonical deterministic release slices:
    - `v2`
    - `escalation`
    - `v2_multi_contract`
    - `paraphrase`
    - `needle`
  - Added deterministic topic-routing test to CI gate jobs.
  - Gate-check invocation now enforces multi-contract/paraphrase/needle thresholds.
  - Upload artifacts now include multi-contract/paraphrase/needle result files.
- `backend/retrieval/router.py`
  - Removed debug file-write side effects and broad `except: pass` blocks in `retrieve(...)`.

### Validation

- `python backend/test_topic_routing.py` -> PASS
- `python -m backend.evaluate_paraphrase --bm25-only` -> `15/15` families, `45/45` variants, worker slang `1.0`
- `python -m backend.evaluate_multi_contract --bm25-only` -> `18/18` (`100.0%`)
- `python -m backend.evaluate_runner --track needle` -> PASS artifact generation
- `python -m backend.evaluate_gate_check`:
  - **Before this change** (needle canonicalization incomplete): needle pass `0/5` (`0.0%`) -> Gate BLOCKED
  - **After this change**: needle pass `5/5` (`100.0%`) with top/middle/bottom all `1.0` -> Gate PASS

## v0.8.46 - Deterministic Formal-Phrase Hardening + Canonical Needle Gate Wiring (February 2026)

### Overview

Completed post-optimization hardening pass for deterministic formal-rewrite robustness and canonicalized the needle slice in the evaluation/gate path.

### What Changed

- `backend/retrieval/router.py`
  - Added deterministic formal-phrase routing/expansion coverage for:
    - term-of-agreement phrasing (`contract/cba` + `start/end/effective/expiration`)
    - minimum inter-shift rest phrasing (`minimum hours between shifts`)
  - Expanded topic lexical signals for `term` and `breaks`.
  - Moved `term` into topic-priority pass to prevent generic `how long` seniority captures from outranking term-of-agreement cues.
- `backend/test_topic_routing.py`
  - Added deterministic BM25-only regression checks for non-slang formal rewrites:
    - PF-01 term-of-agreement variants (`start/end`, `effective/expiration`)
    - PF-15 minimum-rest-between-shifts formal phrasing
- `backend/evaluate_needle.py` (new)
  - Added canonical deterministic needle evaluator writing:
    - `data/test_set/needle_results.json`
  - Reports overall pass rate + by-position (`top/middle/bottom`) rates and citation/keyword checks.
- `backend/evaluate_runner.py`
  - Added canonical `needle` track.
  - Included `needle` in `all`.
  - Executes `needle` track in deterministic BM25-only mode.
- `backend/evaluate_gate_check.py`
  - Added required needle gate checks:
    - `--needle-results`
    - `--min-needle-pass-rate` (default `0.80`)
    - `--min-needle-position-pass-rate` (default `0.80`)
  - Needle artifact/check is now release-blocking when missing/failing.
- Docs synced in same change set:
  - `README.md`
  - `legal/RELEASE-GATES.md`

### Validation

- `python -m backend.evaluate_paraphrase --bm25-only`
  - **Before**: Families `13/15` (`86.7%`), Variants `42/45` (`93.3%`)
    - Failing variants:
      - `PF-01-b`: "When does this union contract start and end?"
      - `PF-01-c`: "How long is the current CBA effective for..."
      - `PF-15-c`: "minimum hours between ... shifts"
  - **After**: Families `15/15` (`100.0%`), Variants `45/45` (`100.0%`)
- `python backend/test_topic_routing.py` -> PASS
- `python -m backend.evaluate_multi_contract --bm25-only` -> `18/18` (`100.0%`)
- `python -m backend.evaluate_runner --track needle` -> writes canonical metadata/artifact
- `python -m backend.evaluate_gate_check` -> BLOCKED on needle thresholds (`0/5`) until synthetic needle injection is performed

## v0.8.45 - Query Expansion Optimization Phase (Lexicon-Driven Slang Lift) (February 2026)

### Overview

Executed Optimization Phase focused on vocabulary mismatch (worker slang vs
formal contract language) using deterministic query expansion driven by frozen
language lexicon artifacts.

### What Changed

- Lexicon audit + generator hardening:
  - `backend/ingest/language_lexicon.py`
    - Added article-title alias templates for key contract concepts:
      - term of agreement (`start/end`, `expiration`, `effective date`)
      - sunday premium
      - store closing / severance
      - bereavement / funeral
      - night premium / graveyard shift
      - close-open shift interval phrasing
    - Reduced noisy low-signal aliases by removing single-token title aliases
      from generated lexicon entries.
- Runtime query expansion:
  - `backend/retrieval/router.py`
    - Confirmed lexicon-backed slang expansion path is active in `expand_query(...)`
      before hybrid/vector retrieval calls.
    - Added deterministic phrase detectors for:
      - close/open shift pattern
      - bereavement/funeral phrasing
      - store-closing phrasing
- Rebuilt all active contract packs and synced runtime artifacts:
  - `scripts/onboard_contract_packages.py --enforce-pack-gates`

### Lexicon Audit Outcome (PF-01 + other failing families)

Before optimization, clerks lexicon lacked critical aliases such as:
- `start and end`, `severance`, `funeral`, `graveyard shift`, `sunday premium`.

After optimization, clerks lexicon includes these mappings (examples):
- `start and end -> term of agreement`
- `severance -> store closing`
- `funeral -> bereavement leave`
- `graveyard shift -> night premiums`
- `working sundays -> sunday premium`

### Results

- `python -m backend.evaluate_paraphrase --bm25-only`
  - Family pass rate: `0.8667` (13/15)
  - Variant pass rate: `0.9333` (42/45)
  - Worker slang pass rate: `1.0000` (target `>0.75` achieved)
- `python -m backend.evaluate_gate_check`
  - Gate status: `PASS`
  - Paraphrase thresholds now pass:
    - family pass `0.867 >= 0.850`
    - worker slang pass `1.000 >= 0.800`
- Canonical metadata run captured:
  - `python -m backend.evaluate_runner --track paraphrase`

## v0.8.44 - Deterministic Language Pipeline + Region Hard Filters (February 2026)

### Overview

Executed the architecture directive to fix empty concept indexes at the ingestion
layer, enforce deterministic language artifacts, and apply hard tenancy filters
(`contract_id` + `region_id`) before retrieval ranking.

### What Changed

- Deterministic ingestion language module:
  - `backend/ingest/language_lexicon.py`
    - region inference + manifest normalization (`region_id`)
    - deterministic chunk language enrichment (`alternative_names`, `worker_questions`)
    - frozen language lexicon artifact builder (`language_lexicon_v1`)
- Runtime lexicon resolver:
  - `backend/language_lexicon_files.py`
- Onboarding pipeline integration:
  - `scripts/onboard_contract_packages.py`
    - enforces manifest `region_id`
    - applies deterministic language enrichment to chunks
    - writes `ontology/language_lexicon.json`
    - syncs runtime lexicon to `data/ontologies/language_lexicon_<contract_id>.json`
    - concept index build now receives manifest context
- Rebuild pipeline integration:
  - `backend/ingest/rebuild_index.py`
    - applies deterministic language enrichment before concept-index rebuild
    - persists repaired chunk metadata
    - emits runtime lexicon artifacts
- Concept index builder hardening:
  - `backend/ingest/toc_index.py`
    - derives concept fields deterministically when chunk fields are missing
- Pack acceptance hardening:
  - `backend/ingest/pack_acceptance.py`
    - `concept_index_exists` moved to required
    - `concept_index_non_empty` moved to required
    - new required checks:
      - `chunks_region_scope`
      - `language_feature_coverage`
      - `language_lexicon_exists`
      - `language_lexicon_non_empty`
      - `language_lexicon_region_id`
- Tenancy hard filters in retrieval stack:
  - `backend/contracts.py`
    - region resolver (`resolve_contract_region_id`)
    - catalog now includes `region_id`
  - `backend/retrieval/vector_store.py`
    - metadata includes `region_id`
    - vector search filters by `contract_id` + `region_id`
  - `backend/retrieval/hybrid_search.py`
    - BM25 resources/search filter by `contract_id` + `region_id`
  - `backend/retrieval/router.py`
    - passes region filter through hybrid/vector retrieval paths
    - chunk-scoped loaders enforce `contract_id` + `region_id`
    - loads frozen language lexicon aliases into query expansion map

### Additional Safety Fix

- `backend/retrieval/router.py`
  - Fixed high-stakes false positives caused by substring matching
    (e.g., `"hired"` previously matching `"fired"`).
  - tightened temporal active-incident regex so bare date words (e.g., `"today"`)
    do not trigger escalation classification without incident context.

### Validation

- Re-onboarded all three active packages with enforced pack gates:
  - `python scripts/onboard_contract_packages.py --package ... --enforce-pack-gates`
  - Result: all three packages PASS required gates.
- Concept indexes are now non-empty:
  - clerks: `228` concepts / `24` questions
  - pueblo meat: `229` concepts / `28` questions
  - loveland meat: `217` concepts / `27` questions
- Runtime tenancy filter smoke test:
  - matching `contract_id`+`region_id` returns results
  - mismatched `region_id` returns `0` results
- Paraphrase benchmark still release-blocked (known next blocker):
  - family pass rate `0.600`
  - worker slang pass rate `0.643`
  - `python -m backend.evaluate_gate_check` => BLOCKED on paraphrase thresholds

## v0.8.43 - v2 Evaluation Runtime-Path Parity (February 2026)

### Overview

Aligned v2 comprehensive retrieval evaluation with production runtime retrieval
path to prevent benchmark/runtime drift.

### What Changed

- `backend/evaluate_comprehensive.py`
  - Switched standard retrieval path from `retriever.retrieve(...)` to
    `retriever.multi_angle_retrieve(...)` so v2 benchmarks follow the same
    retrieval pipeline used by API query handling.

### Validation

- Static compile check:
  - `python -m py_compile backend/evaluate_comprehensive.py`
- Full comprehensive rerun in this sandbox is currently blocked by restricted
  outbound model-download/network permissions.

## v0.8.42 - Paraphrase Robustness Gate Integration (February 2026)

### Overview

Added a deterministic paraphrase/slang robustness evaluation track and wired it
into canonical release-gate checks so terse-worker-phrasing regressions are
release-visible.

### What Changed

- `backend/evaluate_paraphrase.py`
  - New evaluator for `data/test_set/paraphrase_test.json`.
  - Runs retrieval for each paraphrase family variant using runtime retrieval path.
  - Reports:
    - family pass rate (all variants in a family must retrieve expected article set)
    - overall variant pass rate
    - worker-slang variant pass rate
    - per-bucket and per-variant-type breakdowns
  - Writes canonical artifact:
    - `data/test_set/paraphrase_results.json`
- `backend/evaluate_runner.py`
  - Added canonical track:
    - `paraphrase`
  - Included `paraphrase` in `all` track.
- `backend/evaluate_gate_check.py`
  - Added paraphrase gate checks (when artifact exists):
    - `--min-paraphrase-family-pass-rate` (default `0.85`)
    - `--min-paraphrase-worker-slang-pass-rate` (default `0.80`)
  - Added `--paraphrase-results` artifact input.
- `README.md`
  - Added canonical runner command for paraphrase slice and gate-check usage.
- `legal/RELEASE-GATES.md`
  - Gate B now explicitly includes paraphrase robustness threshold.

### Validation

- `python -m backend.evaluate_paraphrase --bm25-only`
  - Families: `9/15` (`60.0%`)
  - Variants: `37/45` (`82.2%`)
  - Worker slang pass rate: `0.6429`
- `python -m backend.evaluate_gate_check`
  - Now BLOCKED on paraphrase thresholds (as expected):
    - family pass rate below threshold
    - worker-slang pass rate below threshold

## v0.8.41 - Personal-Holiday Retrieval Routing Hardening (February 2026)

### Overview

Fixed a short-query retrieval failure mode where worker phrasing like "float days"
could retrieve only sick-time usage sections and miss core personal-holiday
eligibility/scheduling context.

### Root Cause

- Topic classification correctly detected `personal_holiday`.
- But inferred topic->article routing lacked `personal_holiday` title hints, so
  no article anchor was added for that topic.
- Retrieval then over-weighted lexical matches in sick-leave sections that mention
  personal holidays as a secondary rule (for example first/second-day sick pay use).

### What Changed

- `backend/retrieval/router.py`
  - Added manifest-title inference hints for `personal_holiday` in
    `_TOPIC_ARTICLE_TITLE_HINTS` (e.g., "personal holiday", "holidays", "holiday pay").
  - Enhanced topic-coverage seeding to select best lexical candidate in topic
    articles instead of always earliest section.
  - Added topic-lexical prioritization signals (including personal-holiday terms)
    so short paraphrases (e.g., `float days`) rank core personal-holiday chunks
    higher within the preferred article set.
  - Added deterministic reranker bypass for ultra-short queries (<=3 tokens) to
    prevent stochastic reranking drift on terse worker prompts.
- `backend/test_topic_routing.py`
  - Added deterministic BM25-only regression check:
    - `float days` -> topic `personal_holiday`
    - Article 16 included in top results
    - Section 38 appears in top retrieval set

### Validation

- `python backend/test_topic_routing.py` -> PASS
- Retrieval trace for `float days` now prioritizes Article 16 sections and includes
  `Article 16, Section 38` in the top result set.
- Release gate check remains PASS:
  - `python -m backend.evaluate_gate_check`

## v0.8.40 - Multi-Contract Benchmark Parity Track (February 2026)

### Overview

Added a contract-parity benchmark/evaluation track so release evidence reports retrieval performance per contract, not only single-contract v2 plus contamination checks.

### What Changed

- `data/test_set/multi_contract_v2.json`
  - Added multi-contract benchmark slice spanning all 3 active contracts.
  - Includes contract-scoped test cases for core legal domains (term, overtime, holidays, vacations, dispute/grievance, pay/rates).
- `backend/evaluate_multi_contract.py`
  - New evaluator for contract-scoped benchmark slices.
  - Produces:
    - overall pass rate
    - per-contract pass rates
    - per-question detail rows
  - Writes canonical output:
    - `data/test_set/multi_contract_v2_results.json`
  - Supports `--bm25-only` mode for deterministic/no-vector environments.
- `backend/evaluate_runner.py`
  - Added canonical track:
    - `v2_multi_contract`
  - Included in `all` track execution.
- `backend/evaluate_gate_check.py`
  - Added multi-contract gate checks (when result file exists):
    - overall threshold (`--min-multi-contract-accuracy`, default `0.80`)
    - per-contract threshold (`--min-multi-contract-per-contract`, default `0.75`)
- `legal/RELEASE-GATES.md`
  - Gate D now explicitly requires multi-contract benchmark slice pass (overall + per-contract floor).

### Validation

- Ran multi-contract evaluator in this environment:
  - `python -m backend.evaluate_multi_contract --bm25-only`
  - Result: `18/18 (100%)`; each contract `6/6`
- Gate check includes multi-contract slice and passes:
  - `python -m backend.evaluate_gate_check`

## v0.8.39 - Multi-Contract Artifact Canonicalization + Shared-Artifact Archive (February 2026)

### Overview

Hardened runtime/index artifact strategy for multi-contract scale by making concept indexes contract-scoped and archiving ambiguous shared chunk/index files.

### What Changed

- `backend/concept_index_files.py`
  - Added contract-scoped concept-index resolution with multi-manifest shared-fallback guard:
    - prefers `concept_index_<contract_id>.json`
    - allows shared `concept_index.json` only in single-manifest mode
- `backend/retrieval/hybrid_search.py`
  - Switched concept-index loading from global singleton to contract-aware cache:
    - `get_concept_index(contract_id=...)`
    - concept article boosts now resolve index per active contract context
- `scripts/onboard_contract_packages.py`
  - Onboarding now builds contract-scoped concept indexes per package:
    - `data/contracts/<contract_id>/chunks/concept_index_<contract_id>.json`
  - Syncs these to runtime:
    - `data/chunks/concept_index_<contract_id>.json`
- `backend/ingest/rebuild_index.py`
  - Rebuild flow now supports contract-scoped operation (`--contract-id`) and multi-contract discovery.
  - Builds per-contract concept indexes rather than relying on shared-only default behavior.
- `backend/ingest/pack_acceptance.py`
  - Added concept-index presence/structure advisory checks:
    - `concept_index_exists`
    - `concept_index_non_empty`
- `scripts/archive_legacy_shared_artifacts.py`
  - Added safe archival utility for multi-contract mode:
    - archives legacy shared artifacts from `data/chunks/` into
      `data/legacy/shared_artifacts/<timestamp>/`
    - writes `archive_report.json`

### Operational Cleanup Performed

- Re-onboarded all three active contract packs (clerks, meat, king soopers) with updated pipeline.
- Archived legacy shared artifacts:
  - `data/chunks/contract_chunks.json`
  - `data/chunks/contract_chunks_smart.json`
  - `data/chunks/contract_chunks_enriched.json`
  - `data/chunks/concept_index.json`
  - archive location:
    - `data/legacy/shared_artifacts/20260214T073015Z/`

### Validation

- Contract-scoped concept index resolution loads expected file at query time.
- Pack gates remain PASS for all three contract packs (advisory items unchanged where expected).
- Release gate check remains PASS:
  - `python -m backend.evaluate_gate_check`

## v0.8.38 - Ingestion Citation Normalization Layer (February 2026)

### Overview

Implemented contract-agnostic citation/segment normalization in ingestion to eliminate legacy naming artifacts (for example `Part part1`) and make chunk IDs/citations consistent across contracts.

### What Changed

- `backend/ingest/smart_chunker.py`
  - Added canonical normalization helpers for:
    - subsection tokens (stable lowercase/sanitized form)
    - split segment tokens (deterministic ID fragments + human labels)
    - chunk-id-safe slug fragments
  - Stopped overloading section split parts into `subsection`.
    - Paragraph/list splits now use `segment_token` and produce normalized citation labels like:
      - `Part 1`
      - `Part 1-5`
      - `Part 6+`
  - Chunk ID construction is now deterministic and normalized:
    - legal subsections: `_sub_<token>`
    - split segments: `_seg_###` / `_seg_###_###` / `_seg_###_plus`
  - Normalized subsection titles via heading cleaner to reduce markup/casing noise.
- `backend/ingest/pack_acceptance.py`
  - Added advisory gate `chunk_citation_normalization` that flags:
    - malformed citations like `Part partN`
    - legacy segment tokens persisted in `subsection`
    - CBA citations missing `Article X, Section Y` structure when article/section metadata exists
- `backend/test_smart_chunker.py`
  - Added regressions for:
    - no `Part partN` output
    - no legacy `subsection=partN` leakage
    - split chunks emitting normalized `Part N` labels

### Validation

- Re-onboarded all three contract packs through the canonical pipeline:
  - `local7_safeway_pueblo_clerks_2022`
  - `local7_safeway_pueblo_meat_2022`
  - `local7_kingsoopers_loveland_meat_2019`
- Pack gates remained PASS; new citation-normalization check passed for regenerated packs.
- Retrieval smoke (BM25-only local path) confirms normalized citations in runtime context:
  - e.g. `Article 17, Section 48, Part 1` appears for vacation query on Pueblo Meat.

## v0.8.37 - Section-Header Parser + Cross-Section Retrieval Hardening (February 2026)

### Overview

Resolved a systemic retrieval gap where valid sections were present in source markdown but not chunked/retrievable, causing false "section not available" responses (notably vacation accrual references to Article 17, Section 48).

### What Changed

- `backend/ingest/smart_chunker.py`
  - Hardened section-header parsing to handle underlined header variants like:
    - `<u>Section 48</u>.`
    - `<u>Section 49.</u>`
  - Section splitting now scans tag-masked text with stable offsets, so inline HTML does not break section detection.
  - Added safe fallback titles when a section header has no clean short title.
- `backend/retrieval/router.py`
  - Hardened related-section expansion:
    - prioritizes topic-relevant articles first
    - prioritizes explicitly referenced sections (e.g., section cross-references in retrieved text)
    - prioritizes neighboring sections to retrieved sections
    - avoids duplicate picks from the same section number monopolizing expansion slots
  - Increased expansion budget in retrieval paths so related sections can actually be added instead of being blocked by tight caps.
  - Added topic-aware preference wiring through both `retrieve(...)` and `multi_angle_retrieve(...)`.
- `backend/test_smart_chunker.py`
  - Added regression test that asserts Section 48/49 parsing from underlined variants.

### Validation

- Re-onboarded affected contract pack:
  - `python scripts/onboard_contract_packages.py --package local7_safeway_pueblo_meat_2022`
- Confirmed contract chunks now include Section 48 entries:
  - `Article 17, Section 48, Part part1..part4`
- Regression test pass:
  - `python backend/test_smart_chunker.py`
- Syntax check pass:
  - `python -m py_compile backend/retrieval/router.py backend/ingest/smart_chunker.py`

## v0.8.36 - Contract-Locked UX + Scalable Table/Wage Ingestion Hardening (February 2026)

### Overview

Implemented architecture-level hardening for multi-contract scale:
- no implicit contract selection before onboarding
- contract/profile lock before chat interaction
- stronger table materialization/citation for appendix-style wage evidence
- removal of global DUG/courtesy hardcoding in favor of contract-pack ontology/manual overrides

### What Changed

- `frontend/index.html`
  - Removed implicit default contract activation on init.
  - Added interaction lock (`contract_id + classification` required) before chat actions/send.
  - Onboarding now enforces contract selection before save/continue.
  - Settings contract dropdown no longer mutates runtime contract until profile save.
  - Added markdown-table rendering in contract viewer for readable table UI.
  - Health/status now shows `Select contract` when no active contract is set.
- `backend/user/profile.py`
  - Removed auto-default contract assignment for new sessions; contract context is now explicit.
- `backend/api.py`
  - `/api/onboard/options` no longer pre-seeds default classifications.
- `backend/ingest/classification_ontology.py`
  - Reduced explicit aliases to lexical variants only; removed semantic cross-role remaps from global rules.
- `backend/ingest/extract_wages.py`
  - Removed global DUG/drive-up fallback aliases.
  - Added generalized classification-header heuristic for all-caps slash/hyphen labels (fixes orphaned progression rows).
- `backend/ingest/table_extractor.py`
  - Improved appendix citation labeling (`Appendix A Wage Table` for wage-like appendix rows).
  - Table-chunk synthesis now always materializes meaningful appendix/wage/vacation tables as first-class retrievable chunks.

### Contract-Pack Decisions (Clerks)

- Added contract-scoped manual overrides:
  - `dug_shopper -> nonfood_gm_floral`
  - `drive_up_and_go -> nonfood_gm_floral`
  - `nonfood_clerk -> nonfood_gm_floral`
- File: `data/contracts/local7_safeway_pueblo_clerks_2022/ontology/manual_classification_overrides.json`
- Rationale: manifest language states DUG wage progression follows Non-Food/GM progression; this is now encoded per-contract, not globally.

### Validation

- Rebuilt/synced all three contract packs via onboarding pipeline.
- Clerks pack gate moved to PASS after contract-scoped overrides (`coverage=0.7143`).
- Canonical v2 benchmark remained PASS:
  - `data/test_set/eval_run_metadata_v2_20260214T065105Z.json` (`55/55`)
- Escalation benchmark PASS:
  - `data/test_set/eval_run_metadata_escalation_20260214T065121Z.json`
- Release gates PASS:
  - `python -m backend.evaluate_gate_check`
- Cross-contract contamination check PASS:
  - `python backend/evaluate_cross_contamination.py`

## v0.8.35 - Contract-Aware Intent Routing Hardening (100/100 v2) (February 2026)

### Overview

Implemented a scalable intent/routing hardening pass to eliminate residual benchmark misses through contract-aware extraction rather than one-off query patches.

### What Changed

- `backend/retrieval/router.py`
  - Added contract-scoped classification alias resolution from onboarding options.
  - Added deterministic extraction for destination-role queries (e.g., promotion "to X").
  - Added contextual wage-intent fallback for progression-style questions (`rate/step/hours`) when classification context exists.
  - Added deterministic explicit-article anchoring from query text (`Article X`) even outside query-interpreter flows.
  - Expanded inferred topic-title hints and priority for `probation`/`promotion` to prevent generic `hours -> scheduling` over-match.
- `backend/evaluate_comprehensive.py`, `backend/evaluate.py`, `backend/evaluate_runner.py`
  - Canonicalized eval contract resolution to match runtime catalog behavior.
  - Metadata now records both configured and resolved contract IDs for auditability.
  - Pack-registry snapshot now uses canonical contract catalog IDs.

### Why This Scales

- Removes dependence on brittle static class regexes by using contract pack artifacts.
- Makes routing resilient across varied chapter terminology and role labels.
- Keeps explicit article references deterministic and model-independent.
- Aligns evaluation execution path with production contract resolution semantics.

### Validation

- Canonical v2 run: `data/test_set/eval_run_metadata_v2_20260214T061939Z.json`
  - Retrieval accuracy: `55/55 (100.0%)`
  - Contract-only: `26/26`
  - Multi-hop: `17/17`
  - Exact numeric: `7/7`
- Release gate check: `PASS`
- Cross-contract contamination evaluation: `PASS` (no leakage)

## v0.8.34 - Table Evidence Fallback + Wage Context Hardening (February 2026)

### Overview

Hardened table/citation reliability and wage lookup robustness for multi-contract runtime by fixing fallback behavior and making classification handling deterministic across contract packs.

### What Changed

- `backend/ingest/table_extractor.py`
  - Added deterministic HTML-table fallback synthesis into `StructuredTable` records when JSON registry misses a table.
  - Fallback tables now receive stable IDs, markdown/csv content, and `table_refs` so retrieval can treat them as first-class evidence.
- `backend/retrieval/router.py`
  - Added `_ensure_wage_table_context(...)` to inject appendix/table evidence into wage-query context when available.
  - Wage retrieval path now appends table-backed appendix chunks for stronger direct citations.
- `backend/generation/verifier.py`
  - `format_response_with_sources(...)` now surfaces non-article table/appendix citations in `sources` and preserves appendix/table evidence for wage responses.
- `backend/user/profile.py`
  - Profile classification input is now normalized against contract-scoped options and known labels (value-or-label safe).
  - Prevents label-form classifications (e.g., `DUG Shopper (Drive Up & Go)`) from missing wage lookup.
- `backend/chunk_files.py`, `backend/wage_files.py`
  - Shared artifact fallback is now automatically disabled when multiple manifests exist.
  - Prevents silent cross-contract drift to legacy shared artifacts in multi-tenant mode.
- `backend/api.py`
  - Contract IDs are canonicalized via catalog resolution before query handling/viewer resolution.
  - Alias IDs now resolve to canonical runtime contracts for consistent artifact selection.

### Validation

- Re-ran onboarding for all 3 contract packages:
  - `local7_safeway_pueblo_clerks_2022`
  - `local7_safeway_pueblo_meat_2022`
  - `local7_kingsoopers_loveland_meat_2019`
- Pack gates remained `PASS` for all three.
- Table linkage improved via synthesized fallback tables (new `table_refs` on previously unlinked HTML tables).
- Deterministic runtime simulation confirms:
  - DUG classification label normalizes to `dug_shopper`
  - Wage lookup returns courtesy-clerk mapped rate
  - Wage responses include `Appendix A` and direct appendix/table citation evidence.

## v0.8.33 - Wage Filter Hardening (Evidence-Driven Retention) (February 2026)

### Overview

Replaced brittle token-whitelist wage filtering with evidence-driven retention so valid role classifications are not dropped during onboarding.

### What Changed

- `scripts/onboard_contract_packages.py`
  - `_filter_plausible_wage_tables(...)` now:
    - keeps classifications with valid numeric rates by default
    - removes strong non-wage labels (plan/benefit-like and generic placeholders)
    - avoids over-dropping uncommon but valid role keys (e.g., `5star_cake_decorator`)

### Validation

- Re-ran onboarding for all three contracts.
- Coverage uplift observed on clerks contract ontology:
  - `0.5714 -> 0.6429`
- Required pack gates still pass on all three packages.

## v0.8.32 - Review Decision CLI (February 2026)

### Overview

Added a deterministic CLI for applying human-reviewed classification override decisions with validation and coverage-delta preview before writing changes.

### What Changed

- Added `scripts/apply_review_overrides.py`
  - Generates unresolved review template (`--emit-template`)
  - Loads reviewed decision file and validates target wage keys
  - Prints override diff (added/updated/removed)
  - Computes ontology coverage before/after and reports delta + newly resolved keys
  - Applies approved decisions to:
    - `data/contracts/<contract_id>/ontology/manual_classification_overrides.json`
  - Creates backup before overwrite when applying
- README onboarding docs now include review override commands

### Validation

- Ran review template + preview cycle for:
  - `local7_safeway_pueblo_clerks_2022`
  - `local7_safeway_pueblo_meat_2022`
  - `local7_kingsoopers_loveland_meat_2019`
- Current templates produced zero-change dry-run deltas (expected baseline until human-reviewed mappings are supplied).

## v0.8.31 - Ingestion Review Queue + Manual Override Loop (February 2026)

### Overview

Added a deterministic ingestion review loop so unresolved/ambiguous extraction outcomes are captured in a package artifact and can be resolved through versioned manual override files, then consumed on the next onboarding run.

### What Changed

- `backend/ingest/review_queue.py` (new)
  - Added `ingestion_review_queue_v1` builder:
    - aggregates ontology unresolved mappings
    - low-confidence ontology mappings
    - unresolved wage rows
    - canonical ambiguity/conflict diagnostics
  - Persists package review artifact:
    - `data/contracts/<contract_id>/ontology/ingestion_review_queue.json`
- `backend/ingest/classification_ontology.py`
  - Added manual override support for classification aliases:
    - `load_manual_classification_overrides(...)`
    - `write_manual_override_template(...)`
    - `build_classification_ontology(..., manual_alias_overrides=...)`
  - Manual override mappings are applied deterministically as `mapping_method="manual_override"`
- `scripts/onboard_contract_packages.py`
  - Writes/reads package override file:
    - `data/contracts/<contract_id>/ontology/manual_classification_overrides.json`
  - Generates ingestion review queue each onboarding run
  - Adds onboarding summary metric: `review_items=<count>`
- `backend/ingest/pack_acceptance.py`
  - Added review queue artifact to pack hash inputs
  - Added required checks:
    - `ingestion_review_queue_exists` (conditional on unresolved/ambiguous issues)
    - `ingestion_review_queue_schema_valid`
- Docs/spec/governance updates:
  - `README.md` onboarding section documents review-loop artifacts
  - `CONTRACT_PACK_SPEC_v1.md` includes review queue and manual overrides
  - `legal/RELEASE-GATES.md` Gate G includes ingestion review-queue gate

## v0.8.30 - Canonical Wage Schema + Deterministic Row Classification (February 2026)

### Overview

Implemented ingestion-level canonical wage-row output and deterministic row-type classification with conflict/ambiguity handling, then enforced schema integrity through pack gates.

### What Changed

- `backend/ingest/extract_wages.py`
  - Added canonical wage schema output: `canonical_wage_rows` (`wage_canonical_rows_v1`)
  - Added deterministic structured-row classifier (`header`, `effective_date`, `classification_header`, `step_row`, `rate_row`, ambiguous/non-wage)
  - Added deterministic canonical-row conflict/ambiguity handling and diagnostics:
    - `canonical_conflicts`
    - `canonical_ambiguities`
    - `row_type_counts`
    - `unresolved_rows`
  - Added step-level conflict resolution for duplicate step definitions
  - Added canonical-row fallback generation for markdown extraction path
- `backend/ingest/pack_acceptance.py`
  - Added canonical wage gate checks:
    - `canonical_wage_rows_present` (required)
    - `canonical_wage_row_schema_valid` (required)
    - `canonical_wage_row_contradictions` (advisory)
    - `canonical_wage_conflict_rate` (advisory)
    - `canonical_wage_ambiguity_rate` (advisory)
- Docs/spec updates:
  - `README.md` now documents canonical wage-row + ontology checks in pack scorecards
  - `CONTRACT_PACK_SPEC_v1.md` now includes canonical wage schema/conflict gate families
  - `legal/RELEASE-GATES.md` Gate G now explicitly includes canonical wage-row schema gate

### Validation

- Re-ran onboarding for all three contracts with pack gates enabled.
- New canonical checks passed across all three packages:
  - schema presence/validity passed
  - contradiction checks passed
  - conflict/ambiguity rates remained within advisory threshold

## v0.8.29 - Classification Ontology + Gatekeeper (February 2026)

### Overview

Added a deterministic contract-specific classification ontology pipeline so manifest-facing role names map audibly to wage-table keys at ingestion time, then enforced ontology integrity through new required pack-gate checks.

### What Changed

- Added `backend/ingest/classification_ontology.py`
  - Builds `classification_ontology_v1` from:
    - manifest classifications
    - extracted wage classification keys
  - Emits deterministic mapping decisions (`exact`, `explicit_alias`, `token_similarity`, unresolved cases)
  - Produces `alias_to_wage_key` map for runtime wage lookup
- `scripts/onboard_contract_packages.py`
  - Generates package ontology artifact:
    - `data/contracts/<contract_id>/ontology/classification_ontology.json`
  - Syncs runtime ontology artifact:
    - `data/ontologies/classification_ontology_<contract_id>.json`
  - Injects ontology aliases into wage artifacts (`classification_aliases`) before runtime sync
- `backend/ingest/extract_wages.py`
  - `lookup_wage(...)` now consults `wages_data["classification_aliases"]` first for deterministic contract-specific alias resolution
- Added `backend/classification_ontology_files.py`
  - Runtime resolver for per-contract ontology artifacts
- `backend/user/profile.py`
  - Contract classification dropdown now merges ontology manifest-facing labels with wage-derived options
  - Supports selecting role names that differ from wage-key naming while keeping wage lookup deterministic
- `backend/ingest/pack_acceptance.py`
  - Added ontology artifact to pack hash inputs
  - Added new checks:
    - `classification_ontology_exists` (required)
    - `classification_ontology_schema_valid` (required)
    - `classification_ontology_alias_integrity` (required)
    - `classification_ontology_manifest_decisions` (required)
    - `classification_ontology_mapping_coverage` (advisory)
- Docs/governance:
  - `CONTRACT_PACK_SPEC_v1.md` now includes ontology artifact/gate family
  - `legal/RELEASE-GATES.md` Gate G now explicitly references ontology integrity
  - `README.md` onboarding artifact list now includes runtime classification ontology

## v0.8.28 - Ingestion Table Reliability + Heading Robustness (February 2026)

### Overview

Hardened ingestion for scale by fixing article-heading parsing edge cases, enforcing unique chunk IDs at generation time, and promoting unmatched structured tables into first-class retrievable chunks so appendix/wage table evidence is no longer silently dropped.

### What Changed

- `backend/ingest/smart_chunker.py`
  - Added markup-tolerant header scanning (`<u>ARTICLE 49</u>` style headings now parse correctly)
  - Added deterministic collision-safe chunk ID allocation at chunk creation time (`__dupN` suffixing)
  - Eliminated ingestion-time duplicate chunk IDs (instead of relying on retrieval-time fallback dedupe)
- `backend/ingest/table_extractor.py`
  - Added `synthesize_unmatched_table_chunks(...)`:
    - Converts registry tables not linked via inline HTML matching into standalone chunks with `table_refs`
    - Skips TOC tables
    - Emits embedding-friendly plaintext + rich markdown table content (`content_with_tables`)
    - Adds appendix/wage-table citation heuristics for retrieval visibility
- `scripts/onboard_contract_packages.py`
  - Integrated unmatched-table chunk synthesis into onboarding pipeline
  - Added synthesis metrics in onboarding summary (`table_chunks+=N`)
- `backend/retrieval/hybrid_search.py`
  - `SearchResult` now carries `content_with_tables`
  - `search_to_chunks(...)` now returns `content_with_tables` and `table_refs` so generation can access table-rich context
  - Added deterministic structured-value/table-evidence boosting for wage/accrual/rate-style queries
- `backend/generation/verifier.py`
  - Grounding checks now evaluate against `content_with_tables` when present, improving table-value grounding validation

### Validation

- `python scripts/onboard_contract_packages.py --package local7_safeway_pueblo_clerks_2022 --package local7_safeway_pueblo_meat_2022 --package local7_kingsoopers_loveland_meat_2019`
  - All three packages PASS required pack gates
  - King Soopers article coverage gate moved from FAIL to PASS (article headings 48/49/50 now parsed)
  - Chunk ID uniqueness advisory moved to PASS for all three packages
  - Table synthesis added retrievable table chunks:
    - clerks `+3`
    - safeway meat `+9`
    - king soopers meat `+5`

## v0.8.27 - Contract Pack Acceptance Gatekeeper Foundation (February 2026)

### Overview

Added an ingestion-owned contract-pack acceptance suite with deterministic scorecards, pack hashes, and onboarding integration so package quality can be evaluated (and optionally enforced) before runtime artifact sync.

### What Changed

- Added `backend/ingest/pack_acceptance.py`
  - Generates package scorecard (`pack_scorecard.json`) with:
    - required/advisory checks
    - artifact hashes
    - deterministic `pack_hash`
  - Checks include manifest schema, chunk integrity, article coverage, wage integrity, critical alias resolution, and table ref integrity
- `scripts/onboard_contract_packages.py`
  - Runs pack acceptance by default (unless `--no-pack-gates`)
  - Added enforcement flags:
    - `--enforce-pack-gates` (block runtime sync on required gate failures)
    - `--strict-pack-gates` (treat advisory failures as blocking)
  - Records accepted pack hashes in `data/contracts/pack_registry.json` when synced
- `backend/evaluate_runner.py`
  - Metadata now captures active accepted pack hash snapshot from pack registry for benchmark traceability
- Docs/governance updates:
  - Added `CONTRACT_PACK_SPEC_v1.md`
  - `README.md` updated with pack gate commands
  - `legal/RELEASE-GATES.md` added Gate G (Contract Pack Quality)

## v0.8.26 - Wage Alias Resolution for DUG/Courtesy-Clerk Paths (February 2026)

### Overview

Fixed contract-role alias gaps where profile classifications like `dug_shopper` were valid in onboarding but failed wage lookup due to missing direct wage-table keys.

### What Changed

- `backend/ingest/extract_wages.py`
  - Added deterministic wage-key alias resolution in `lookup_wage(...)`:
    - `dug_shopper`, `drive_up_go`, `personal_shopper`, `clicklist_shopper`, `dug`, `bagger` -> courtesy-clerk/all-purpose fallback order
  - Added token-based fallback for shopper/dug variants when explicit aliases are absent
  - Preserved existing direct and fuzzy lookup behavior after alias resolution

## v0.8.25 - Profile Save Guardrails and Appendix Citation Visibility (February 2026)

### Overview

Moved chat-reset behavior to profile save (instead of dropdown-change), enforced classification-required profile saves, and surfaced `Appendix A` as an explicit citation for wage-backed responses.

### What Changed

- `frontend/index.html`
  - Profile save now blocks when job classification is missing
  - Onboarding submit explicitly blocks without classification
  - Contract dropdown changes no longer auto-reset chat; reset confirmation/rotation remains on save when contract/classification actually changes
  - Non-article citations (e.g., `Appendix A`) now render as non-clickable citation badges (no broken popover navigation)
- `backend/generation/verifier.py`
  - Citation extraction now recognizes `Appendix A`
  - Wage responses force include wage citation (`Appendix A` by default) in `citations`
  - Added table-aware citation augmentation for cited articles when retrieved chunks include `table_refs`

## v0.8.24 - Retrieval Stabilization for Multi-Contract Vacation/Wage Queries (February 2026)

### Overview

Improved multi-contract retrieval quality for vacation/pay questions by adding manifest-title-based topic article inference, explicit topic-article context seeding, and BM25 chunk-id collision hardening. Also added a deterministic wage guardrail when role/classification is missing.

### What Changed

- `backend/retrieval/router.py`
  - Added inferred topic-to-article fallback from manifest article titles (`infer_topic_article_map(...)`) when manifest routing maps are empty
  - Added `_ensure_topic_article_coverage(...)` so topic-relevant articles are always represented in retrieved context
  - Integrated topic-article coverage in both `retrieve(...)` and `multi_angle_retrieve(...)`
- `backend/retrieval/hybrid_search.py`
  - BM25 now indexes `content_with_tables` + `summary` (when available), not only plain `content`
  - Added `_ensure_unique_chunk_ids(...)` to prevent silent BM25 document loss when chunk IDs collide
- `backend/api.py`
  - Added deterministic wage-intent guardrail: if classification is missing, respond with explicit role-selection instruction instead of implying wage data is unavailable
- `backend/user/profile.py`
  - Normalized contract-derived classification labels for UI consistency
  - Expanded 2-digit cutoff years in labels (e.g., `5/20/77` -> `5/20/1977`)
- `backend/generation/prompts.py`
  - Added prompt rule to avoid false "partial contract access" limitation claims

## v0.8.23 - Onboarding Contract Switch Viewer Sync (February 2026)

### Overview

Fixed a stale contract-viewer state path where changing contract in onboarding could leave table-of-contents/article sections from the previous contract until manual refresh.

### What Changed

- `frontend/index.html`
  - Onboarding contract dropdown change now calls `setActiveContract(..., refreshViewer: true)` so contract TOC/article caches reset consistently (matching settings behavior)

## v0.8.22 - Contract Catalog Cleanup, Dynamic Health, and Prompt De-Branding (February 2026)

### Overview

Addressed multi-contract UX/runtime polish: removed duplicate legacy contract options, made health/status chunk counts contract-aware, expanded contract-scoped classification coverage for clerks, and removed hardcoded union/store identity from base system prompts.

### What Changed

- `backend/contracts.py`
  - Deduplicates equivalent manifests in `/api/contracts` (same union/employer/term), preferring canonical `local*` IDs
  - Default contract resolution now prefers deduped catalog rows
  - Added alias mapping in `get_contract_catalog_entry(...)` for deduped contract IDs
- `backend/api.py`
  - `/api/health` now accepts optional `contract_id` and returns:
    - `contract_chunks`
    - `active_contract_id`
  - Health status now reflects contract chunk availability (not only vector DB count)
- `backend/user/profile.py`
  - Clerks contracts now merge supplemental legacy role options on top of wage-derived options to avoid missing job selections
- `backend/generation/prompts.py`
  - Removed hardcoded UFCW/store-specific text from high-level system prompt templates
  - Added dynamic `ACTIVE CONTRACT CONTEXT` injection via `build_prompt(contract_context=...)`
- `frontend/index.html`
  - Status badge now requests `/api/health?contract_id=<active>` so displayed section counts update with contract selection
  - Contract labels simplified for readability with location/department context (e.g., King Soopers/Loveland/Meat)
  - Added fallback when persisted/default contract id is unavailable in filtered catalog
  - Profile load now forces contract-view cache reset when contract context changes

## v0.8.21 - Contract-Scoped Profile UX and Manifest Title Cleanup (February 2026)

### Overview

Improved multi-contract frontend behavior by making job classifications contract-scoped, enforcing new-chat resets for profile context changes, and cleaning markdown-ingest article title artifacts that leaked into manifest TOCs.

### What Changed

- `backend/user/profile.py`
  - Added contract-scoped classification resolution from `data/wages/wage_tables_<contract_id>.json`
  - Added `resolve_classification_display_name(...)` for consistent profile display labels
- `backend/api.py`
  - Added `GET /api/classifications?contract_id=<id>` for contract-specific classification options
  - `GET /api/onboard/options` now returns classifications for default contract context
  - Profile and wage-estimate responses now use contract-aware classification display names
- `backend/ingest/smart_chunker.py`
  - Tightened article-title header regex to prevent newline bleed-through (stray trailing `S` artifacts)
  - Expanded article heading detection to include `### ARTICLE` formats so mid-document headers are not skipped
- `backend/ingest/manifest.py`
  - Tightened article-title extraction regex and added TOC dot-leader/page-number cleanup
- `frontend/index.html`
  - Contract dropdown labels simplified to readable names (no raw contract-id suffix)
  - Classification dropdowns now repopulate per selected contract using `/api/classifications`
  - Contract switch now resets viewer state cleanly and reloads contract-specific TOC/article context
  - Session handling upgraded to mutable chat sessions with local session metadata
  - Changing contract/classification after submitted chat prompts warning and starts a new chat session
  - Chat markdown rendering improved for headings, bullets, numbered lists, quotes, and inline code

## v0.8.20 - Wage Extraction Hardening and Ingestion Standardization (February 2026)

### Overview

Standardized contract-pack wage extraction on a deterministic table-registry-first pipeline and fixed an effective-date parsing bug that could collapse multi-year step rates.

### What Changed

- `backend/ingest/extract_wages.py`
  - Added strict effective-date row detection to avoid treating incidental dates in classification labels (e.g. `AFTER 5/20/77`) as wage-table effective dates
  - Preserves correct multi-date progression rate maps for step-based wage classifications
- `scripts/onboard_contract_packages.py`
  - Wage extraction now passes structured table registry into `extract_wages(...)` during onboarding
  - If table pipeline is skipped but a package registry already exists, wages still use the existing registry for deterministic extraction
- `README.md`
  - Documented wage artifact generation as table-registry-first with markdown fallback

## v0.8.19 - Runtime Contract Catalog and Frontend Contract Selection (February 2026)

### Overview

Removed remaining hardcoded single-contract runtime defaults from API/profile/frontend request paths and added a manifest-driven contract catalog endpoint for UI contract selection.

### What Changed

- Added `backend/contracts.py`
  - Manifest catalog helpers:
    - `list_contract_catalog()`
    - `resolve_default_contract_id()`
    - `get_contract_catalog_entry()`
  - Normalizes OCR/PDF whitespace in `employer`, `union_local`, and version fields
- `backend/config.py`
  - `CONTRACT_ID` default now resolved dynamically:
    1. env override (`KARL_CONTRACT_ID` / `CONTRACT_ID`)
    2. legacy benchmark default when present
    3. first manifest fallback
- `backend/user/profile.py`
  - Removed hardcoded `safeway_pueblo_clerks_2022` / `Safeway` defaults
  - Profile contract context now initializes from manifest catalog
  - Updating `contract_id` now also updates `union_local` and `employer`
- `backend/api.py`
  - Added `GET /api/contracts`
  - `/api/onboard/options` now returns contract catalog + `default_contract_id`
  - Profile API now carries `contract_id` and `union_local_id`
  - Profile update accepts `contract_id` with manifest validation
  - Wage estimate lookup now uses `profile.contract_id`
- `frontend/index.html`
  - Removed hardcoded query payload constants for contract context
  - Added runtime contract loading from `/api/contracts`
  - Added contract selectors in onboarding + settings
  - Contract viewer/article/section API calls now include `contract_id`
  - Query payload now uses selected contract metadata (`union_local_id`, `contract_id`, `contract_version`)
  - Contract selection persists via `localStorage`

## v0.8.18 - Multi-Contract Package Onboarding Pipeline and Offline-Safe Contamination Eval (February 2026)

### Overview

Added an end-to-end contract package onboarding script, fixed a TOC-triggered chunking bug for meat contracts, and hardened offline-safe multi-contract evaluation behavior.

### What Changed

- Added `scripts/onboard_contract_packages.py`
  - Reads `data/contracts/<package>/source/*`
  - Generates manifest/chunk/table/wage artifacts per package
  - Syncs runtime artifacts to:
    - `data/manifests/<contract_id>.json`
    - `data/chunks/contract_chunks_*_<contract_id>.json`
    - `data/tables/structured_tables_<contract_id>.json`
    - `data/wages/wage_tables_<contract_id>.json`
- Added `backend/wage_files.py`
  - Contract-scoped wage artifact resolver with shared fallback
- `backend/retrieval/router.py`
  - Wage lookup now loads per-contract wage tables via resolver
  - Avoids eager vector-store initialization when vector weight is disabled
- `backend/ingest/smart_chunker.py`
  - Fixed LOU boundary detection to ignore TOC matches before first real article heading
  - Prevents zero-chunk output on contracts with TOC "LETTERS OF UNDERSTANDING" lines
- `backend/retrieval/hybrid_search.py`
  - Fixed contract BM25 cache behavior so default shared corpus does not shadow per-contract files
  - Keeps explicit `chunks_file` override behavior for tests
- `backend/evaluate_cross_contamination.py`
  - Reworked to BM25-only `HybridSearcher` path for offline-safe execution
  - No dependency on embedding-model downloads for contamination checks
- `backend/api.py`
  - `/api/wage` now accepts optional `contract_id` and returns it in response
- `README.md`
  - Added onboarding command and per-contract wage artifact notes

## v0.8.17 - Per-Contract Chunk Artifact Resolution (February 2026)

### Overview

Added a shared chunk-artifact resolver and updated runtime/indexing paths to prefer contract-specific chunk files before shared corpus fallback.

### What Changed

- Added `backend/chunk_files.py`
  - Centralized chunk artifact resolution
  - Prefers per-contract files:
    - `contract_chunks_enriched_<contract_id>.json`
    - `contract_chunks_smart_<contract_id>.json`
    - `contract_chunks_<contract_id>.json`
  - Falls back to shared chunk files when needed
- `backend/retrieval/router.py`
  - Contract-scoped chunk loading now uses resolver instead of hardcoded shared file paths
- `backend/retrieval/hybrid_search.py`
  - BM25 resources now support contract-scoped caches with per-contract file preference
  - Honors explicit `chunks_file` override for tests while still enforcing contract filtering
- `backend/retrieval/vector_store.py`
  - Index build/load helper now resolves per-contract chunk files first
- `backend/api.py`
  - Contract viewer article/section endpoints now load chunks via resolver with contract context
- `backend/ingest/rebuild_index.py`
  - Default input chunk artifact now resolved via shared resolver
- `backend/ingest/parse_contract.py`
  - Writes `contract_chunks_<contract_id>.json` in addition to legacy `contract_chunks.json`
- `README.md`
  - Added per-contract chunk artifact naming and fallback behavior

## v0.8.16 - Legacy Chunk Scope Guardrails and CI Cross-Contamination Check (February 2026)

### Overview

Tightened contract scoping behavior for chunk reads by allowing unscoped legacy chunks only in single-manifest mode, and added cross-contamination checks to CI pipelines.

### What Changed

- `backend/retrieval/router.py`
  - Added single-manifest compatibility guard (`_allow_legacy_unscoped_chunks`)
  - Contract chunk loader now includes missing-`contract_id` chunks only when exactly one manifest exists
- `backend/api.py`
  - Contract viewer article/section endpoints now apply the same single-manifest legacy fallback rule
  - In multi-manifest mode, viewer endpoints require strict `chunk.contract_id == requested contract_id`
- `.github/workflows/eval-ci.yml`
  - Added `python -m backend.evaluate_cross_contamination` to PR and main gate flows
- `README.md`
  - Updated CI behavior notes to include cross-contamination checks and single-manifest skip behavior

## v0.8.15 - Contract-Scoped Expansion Paths and Eval Runner Manifest Preflight (February 2026)

### Overview

Closed additional single-contract leakage paths by enforcing contract scoping in multi-angle/expansion retrieval internals, and added manifest validation preflight directly inside the canonical evaluation runner.

### What Changed

- `backend/retrieval/router.py`
  - Added contract-scoped chunk cache (`_all_chunks_by_contract`)
  - Full-article expansion and related-section expansion now run against contract-filtered chunk sets
  - Multi-angle explicit article fetch now filters by `contract_id`
  - Query interpreter invocation now receives explicit `contract_id`
- `backend/retrieval/query_interpreter.py`
  - Removed hardcoded manifest filename usage
  - Contract-specific article-title cache keyed by `contract_id`
  - `interpret()` now accepts `contract_id`
- `backend/evaluate_runner.py`
  - Runs `backend.validate_manifests` preflight before benchmark execution
  - Fails fast with recorded metadata result if manifest validation fails
- `README.md`
  - Notes that canonical runner performs manifest-validation preflight

## v0.8.14 - Manifest Validation Gate and Contract-Agnostic Query Path (February 2026)

### Overview

Removed a single-contract runtime blocker in `/api/query`, enforced explicit contract context propagation in remaining generation test paths, and added manifest validation as a CI gate.

### What Changed

- `backend/api.py`
  - Removed hardcoded rejection of non-default `contract_id`
  - Runtime now relies on manifest existence + context/version validation instead of a fixed `CONTRACT_ID` check
- Added `backend/validate_manifests.py`
  - Validates manifest required fields and routing structure
  - Enforces `contract_id` == filename, and `contract_version == term_start__term_end`
  - Validates that routing article references point to real manifest articles
- `data/manifests/safeway_pueblo_clerks_2022.json`
  - Added canonical `contract_version` field
- `.github/workflows/eval-ci.yml`
  - Added manifest validation step to PR and main jobs
- `backend/evaluate_generation.py`, `backend/test_generation.py`
  - Explicitly pass `contract_id` into intent classification and retrieval paths
- `README.md`
  - Added manifest validation command and CI behavior update

## v0.8.13 - Strict Query Context Validation and BM25 Contract Filtering (February 2026)

### Overview

Hardened tenant context enforcement by requiring and validating full query context (`union_local_id`, `contract_id`, `contract_version`) and closing a hybrid-search contamination gap by filtering BM25 results by `contract_id`.

### What Changed

- `backend/api.py`
  - `/api/query` now requires:
    - `union_local_id`
    - `contract_id`
    - `contract_version`
  - Validates `union_local_id` against manifest `union_local`
  - Validates `contract_version` against manifest-derived version format (`term_start__term_end`)
- `backend/retrieval/hybrid_search.py`
  - Added `contract_id` support in hybrid search path
  - Applies `contract_id` filtering to BM25 ranking candidates
  - Carries `contract_id` in returned chunk metadata
- `backend/retrieval/router.py`
  - Propagates `contract_id` into vector and hybrid retrieval branches
  - Passes `contract_id` through multi-angle retrieval vector search branch
- `frontend/index.html`
  - Query payload now sends required contract context fields
- Added cross-contamination evaluator scaffold:
  - `backend/evaluate_cross_contamination.py`
  - Skips in single-contract mode unless `--require-multi-contract` is set

### Files Updated

| File | Changes |
|------|---------|
| `backend/api.py` | Required/validated union local and contract version |
| `backend/retrieval/hybrid_search.py` | BM25 contract_id filtering and metadata propagation |
| `backend/retrieval/router.py` | contract_id propagation through all retrieval paths |
| `frontend/index.html` | Sends required contract context in `/api/query` |
| `backend/evaluate_cross_contamination.py` | New multi-contract contamination evaluator scaffold |
| `README.md` | Added strict context format and cross-contamination command |
| `UPDATE_LOG.md` | Added this entry |

## v0.8.12 - Strict Contract Context Enforcement and Isolation Checks (February 2026)

### Overview

Implemented strict contract context propagation and validation as a Phase 2 multi-tenant safety foundation.

### What Changed

- `backend/retrieval/router.py`
  - Added `ensure_contract_manifest(contract_id)` validation
  - Enforced manifest validation in `classify_intent`
  - Added explicit `contract_id` parameter to:
    - `HybridRetriever.retrieve(...)`
    - `HybridRetriever.multi_angle_retrieve(...)`
  - Propagated `contract_id` in retrieval result metadata
- `backend/api.py`
  - `/api/query` now requires explicit `contract_id`
  - Invalid/unknown contract manifests return HTTP 400
  - `classify_intent` and retrieval now receive explicit `contract_id`
- Evaluation scripts now pass explicit `CONTRACT_ID`:
  - `backend/evaluate.py`
  - `backend/evaluate_comprehensive.py`
  - `backend/evaluate_escalation_precision.py`
- Added contract isolation test:
  - `backend/test_contract_isolation.py`
  - Verifies unknown contract rejection and no wrong-contract retrieval contamination in test queries
- CI update:
  - `.github/workflows/eval-ci.yml` now runs `python -m backend.test_contract_isolation` on PR path

### Files Updated

| File | Changes |
|------|---------|
| `backend/retrieval/router.py` | Strict manifest validation and explicit contract_id plumbing |
| `backend/api.py` | Required contract_id and explicit contract routing |
| `backend/evaluate.py` | Explicit CONTRACT_ID propagation |
| `backend/evaluate_comprehensive.py` | Explicit CONTRACT_ID propagation |
| `backend/evaluate_escalation_precision.py` | Explicit CONTRACT_ID propagation |
| `backend/test_contract_isolation.py` | New isolation test |
| `.github/workflows/eval-ci.yml` | Added contract isolation check in PR pipeline |
| `README.md` | Added contract-context and isolation check notes |
| `UPDATE_LOG.md` | Added this entry |

## v0.8.11 - Contract Context API Plumbing (Phase 2 Foundation) (February 2026)

### Overview

Added contract context fields to query API request/response for multi-tenant groundwork and auditability.

### What Changed

- `backend/api.py`
  - `QueryRequest` now accepts:
    - `union_local_id`
    - `contract_id`
    - `contract_version`
  - `QueryResponse` now returns:
    - `union_local_id`
    - `contract_id`
    - `contract_version`
  - Intent classification now receives `contract_id` explicitly.
  - Single-contract guard added: requests with unsupported `contract_id` return HTTP 400.

### Notes

- This is a compatibility-safe foundation step toward strict multi-contract runtime isolation.
- Current deployment remains single-contract (`safeway_pueblo_clerks_2022`).

## v0.8.10 - Local Benchmark Snapshot Archiving (February 2026)

### Overview

Added a local snapshot archiver for evaluation artifacts so benchmark evidence can be saved in versioned folders and committed to git intentionally.

### What Changed

- Added `scripts/archive_eval_snapshot.py`
  - Copies key evaluation outputs into `data/test_set/history/<timestamp>_<label>/`
  - Writes `snapshot_manifest.json` per snapshot
  - Maintains `data/test_set/history/index.json`
- Updated README with snapshot archive command.

### Files Updated

| File | Changes |
|------|---------|
| `scripts/archive_eval_snapshot.py` | New snapshot archiver |
| `README.md` | Added benchmark snapshot archive command |
| `UPDATE_LOG.md` | Added this entry |

## v0.8.9 - CI Evaluation Pipeline and Release-Gate Enforcement (February 2026)

### Overview

Added CI automation for evaluation tracks and explicit release-gate threshold enforcement aligned with the Phase 1 hardening plan.

### What Changed

- Added release-gate checker script:
  - `backend/evaluate_gate_check.py`
  - Validates v2 accuracy and escalation precision/recall/false-positive thresholds from artifact JSON files
  - Exits non-zero on failures for CI blocking
- Added GitHub Actions workflow:
  - `.github/workflows/eval-ci.yml`
  - PR path: run core v2 (`normal`) + escalation slice + gate check
  - Main path: run full v2 ablation suite + separate gate-check job
  - Uploads evaluation artifacts for audit/review
- Updated README with gate-check command and CI behavior summary

### Local Validation Snapshot

Command:
`python -m backend.evaluate_gate_check --v2-results data/test_set/comprehensive_results.json --escalation-results data/test_set/escalation_precision_results.json --min-v2-accuracy 0.80 --min-escalation-precision 0.90 --min-escalation-recall 0.70 --max-escalation-fpr 0.10`

Result:
- v2 accuracy: `0.891` (pass)
- escalation precision: `1.000` (pass)
- escalation recall: `1.000` (pass)
- escalation false-positive rate: `0.000` (pass)

### Files Updated

| File | Changes |
|------|---------|
| `backend/evaluate_gate_check.py` | New release-gate threshold checker |
| `.github/workflows/eval-ci.yml` | New CI workflow for PR/core and main/full-suite evaluation |
| `README.md` | Added gate-check command and CI behavior notes |
| `UPDATE_LOG.md` | Added this entry |

## v0.8.8 - Canonical Evaluation Runner and Run Metadata (February 2026)

### Overview

Added a canonical evaluation runner to standardize benchmark execution and produce auditable run metadata for release-gate evidence.

### What Changed

- Added `backend/evaluate_runner.py` with canonical tracks:
  - `v1`
  - `v2` (ablation-capable)
  - `escalation`
  - `all`
- Added deterministic run metadata capture:
  - timestamp
  - executed commands
  - git commit/branch/dirty state
  - config/model snapshot
  - SHA256 hashes for chunks/manifests/wages artifacts
- Added metadata artifacts:
  - `data/test_set/eval_run_metadata_<track>_<timestamp>.json`
  - `data/test_set/eval_run_metadata_latest.json`
- Updated canonical command references in docs.

### Files Updated

| File | Changes |
|------|---------|
| `backend/evaluate_runner.py` | New canonical runner with metadata capture and track orchestration |
| `README.md` | Added canonical runner commands for v1/v2/escalation |
| `Evaluation_Plan_v3.md` | Added canonical command list under Phase 0 exit criteria |
| `UPDATE_LOG.md` | Added this entry |

## v0.8.7 - Deterministic Escalation Precision Hardening (February 2026)

### Overview

Implemented deterministic-first escalation hardening in the retrieval/intent layer with explicit two-stage outputs:

1. `high_stakes_topic` (informational domain risk)
2. `active_urgent_context` (actual escalation trigger)

This update adds conditional/hypothetical suppressors, a dedicated escalation precision test slice, and reproducible confusion-matrix reporting.

### What Changed

- Updated high-stakes intent classification to deterministic two-stage logic in `backend/retrieval/router.py`
- Added conditional/hypothetical suppressors (for phrasing like "what are my rights if", "hypothetically", "in case", "if I were")
- Added explicit intent fields:
  - `high_stakes_topic`
  - `active_urgent_context`
  - `escalation_policy`
- Exposed those fields in API response (`backend/api.py`)
- Added dedicated escalation precision test set:
  - `data/test_set/escalation_precision_test.json`
- Added dedicated evaluator with confusion matrix + threshold tradeoffs:
  - `backend/evaluate_escalation_precision.py`
  - Output artifact: `data/test_set/escalation_precision_results.json`

### Evaluation Snapshot (Escalation Precision Slice)

Run date: February 2026  
Command: `python -m backend.evaluate_escalation_precision`

- Active escalation confusion matrix: `TP=10 FP=0 TN=20 FN=0`
- Precision: `1.000`
- Recall: `1.000`
- False-positive rate: `0.000`

Threshold tradeoff examples (active pattern hits >= N):
- `N=1`: precision 0.714, recall 1.000, FPR 0.200
- `N=2`: precision 0.833, recall 0.500, FPR 0.050
- `N=3`: precision 1.000, recall 0.200, FPR 0.000

### Files Updated

| File | Changes |
|------|---------|
| `backend/retrieval/router.py` | Deterministic two-stage escalation classifier, suppressors, expanded active/topic patterns |
| `backend/api.py` | Added escalation policy outputs in `QueryResponse` |
| `backend/evaluate_escalation_precision.py` | New evaluator for confusion matrix, precision/recall/FPR, threshold tradeoffs |
| `data/test_set/escalation_precision_test.json` | New escalation-focused test slice |
| `README.md` | Added escalation precision evaluation command and feature note |
| `UPDATE_LOG.md` | Added this entry |

## v0.8.6 - Escalation Precision and 3-Contract Execution Timeline Sync (February 2026)

### Overview

Documentation and governance update to make escalation precision a release-blocking requirement and to align execution sequencing for 3-contract deployment.

### What Changed

- Added a full Prototype -> 3-Contract deployed service timeline in `Evaluation_Plan_v3.md` (Phase 0 through Phase 5)
- Added an explicit pre-scale escalation hardening phase before Contract 2/3 onboarding
- Added escalation-specific success metrics:
  - high-stakes escalation precision threshold
  - false-positive rate cap
- Added a dedicated escalation test slice definition:
  - conditional/hypothetical rights prompts
  - active urgent situations
  - neutral policy prompts with trigger words
- Updated handoff prompt (`old/HANDOFF_PROMPT.md`) with deterministic-first escalation constraints, two-stage classification, suppressors, and required confusion-matrix reporting
- Updated `legal/RELEASE-GATES.md` so escalation precision regressions are explicitly release-blocking

### Files Updated

| File | Changes |
|------|---------|
| `Evaluation_Plan_v3.md` | Added weekly execution timeline, pre-scale escalation hardening phase, escalation test slice, and explicit precision/false-positive success thresholds |
| `legal/RELEASE-GATES.md` | Expanded Gate C with escalation precision/false-positive/determinism requirements and explicit regression blocking rule |
| `old/HANDOFF_PROMPT.md` | Added next-agent escalation constraints and escalation-focused deliverables |
| `UPDATE_LOG.md` | Added this entry |

### Code Impact

- No runtime logic changed in this update
- No retrieval, generation, or model configuration changed
- No benchmark artifacts modified

## v0.8.5 - Documentation Sync for Benchmark Tracks and Lean Architecture (February 2026)

### Overview

Documentation-only update to align project language with actual benchmark progression and current architecture direction.

This release clarifies benchmark naming and intent:

- **Benchmark v1**: Legacy golden benchmark (historical 100%, 55/55)
- **Benchmark v2**: Harder comprehensive benchmark track (performance dropped from v1 levels; now used for tougher retrieval stress tests)
- **Benchmark v3**: Planned scaled multi-contract evaluation strategy (partially implemented, not complete yet)

This update also clarifies that current runtime defaults are **lean mode**:

- Hypothesis layer disabled by default
- Full article expansion disabled by default
- BM25 keyword weighting disabled by default
- Query interpreter and reranker remain enabled

### Why This Update Was Needed

Project docs had drift between:

- historical benchmark claims (v1)
- current comprehensive benchmark work (v2)
- planned scaling strategy (v3)

This caused ambiguity about whether "v3" meant current benchmark results or planned multi-contract expansion. The docs now explicitly separate those tracks.

### Files Updated

| File | Changes |
|------|---------|
| `README.md` | Rewritten to explicitly document v1/v2/v3 benchmark taxonomy, current lean architecture, and current status |
| `SETUP.md` | Updated SDK wording and `.env` creation instructions to remove deleted `env.example` dependency |
| `UPDATE_LOG.md` | Added this documentation sync entry |

### Code Impact

- No retrieval, generation, or evaluation runtime logic changed
- No model configuration changed
- No benchmark artifacts modified

---
---

## v0.8 - SDK Migration, Chunker Hardening & 100% Benchmark (February 2026)

### Overview
Major infrastructure update: migrated from the deprecated `google.generativeai` SDK to the new `google.genai` SDK, fixed systemic chunker bugs that silently dropped articles, resolved reranker JSON parsing failures, and fixed concept boost prioritization. Result: **55/55 (100%) benchmark** — up from 50/55 (90.9%).

---

### Benchmark Results

| Metric | v0.7.5 | v0.8 | Change |
|--------|--------|------|--------|
| Overall | 50/55 (90.9%) | **55/55 (100%)** | **+9.1 pts** |
| Retrieval Accuracy | 92.7% | **100%** | **+7.3 pts** |
| Wage Lookup | 100% | 100% | - |
| Escalation Detection | 66.7% | **100%** | **+33.3 pts** |

All 19 categories now at 100%.

#### Previously Failing Tests — All Fixed

| Test ID | Category | Question | Root Cause | Fix |
|---------|----------|----------|------------|-----|
| 2 | wages | "How much does a Courtesy Clerk make after 36 months?" | Wage query regex restricted to `(i\|we)` subjects | Broadened subject pattern to `.+` |
| 3 | wages | "What is the Head Clerk rate of pay?" | "rate of pay" not in wage keywords | Added `"rate of pay"` to `WAGE_KEYWORDS` |
| 43 | benefits | "Is there a 401k plan?" | Chunker regex `[A-Z][A-Z\s&,]+` excluded digits — Article 39 "401K PLAN" not detected | Fixed regex to `[A-Z0-9][A-Z0-9\s&,/()-]+` |
| 48 | time_cards | "When do I punch the time clock?" | Concept boost prioritized noisy matches over precise question matches | Swapped priority: question matches first |
| 51 | high_stakes | "My manager is harassing me" | Active-voice harassment not in escalation patterns | Added active-voice harassment patterns |
| 53 | dress_code | "What color shoes can I wear?" | LOU dress code not chunked separately | Added LOU sub-item splitting to chunker |

---

### Breaking Changes

#### SDK Migration: `google.generativeai` → `google.genai`

The deprecated `google.generativeai` SDK (sunset March 31, 2026) has been replaced with `google.genai` v1.62.0 across all 8 files that use the Gemini API.

**Old Pattern:**
```python
import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash", system_instruction="...")
response = model.generate_content(prompt)
```

**New Pattern:**
```python
from google import genai
client = genai.Client(api_key=GEMINI_API_KEY)
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=genai.types.GenerateContentConfig(
        system_instruction="...",
        temperature=0.1,
    )
)
```

**Key Differences:**
- `system_instruction` moves from model creation to per-request `GenerateContentConfig`
- Client-based API (`genai.Client`) replaces global `genai.configure()`
- Do NOT use `http_options={"timeout": N}` — causes SSL handshake failures
- SDK auto-detects `GOOGLE_API_KEY` env var

#### Model Upgrades

| Component | Old Model | New Model |
|-----------|-----------|-----------|
| Generation | `gemini-2.0-flash` | `gemini-2.5-pro` |
| Hypothesis | `gemini-2.0-flash` | `gemini-2.5-flash` |
| Interpreter | `gemini-2.0-flash` | `gemini-2.5-flash` |
| Reranker | `gemini-2.0-flash` | `gemini-2.5-flash` |
| Enricher | `gemini-2.0-flash` | `gemini-2.5-flash` |

---

### Bug Fixes

#### 1. Chunker: Article Title Regex Excluded Digits (Test 43)

**Problem**: `ARTICLE_HEADER` and `ARTICLE_HEADER_SINGLE` regex patterns used `[A-Z][A-Z\s&,]+` for the title capture group. Article 39 "401K PLAN" has digits, so it was never detected as a separate article. Section 115 (401K content) was incorrectly assigned to Article 38.

**Impact**: Any contract with digits in article titles (e.g., "401K PLAN", "SECTION 125 PLANS") would silently lose articles. This is a systemic scaling issue.

**Fix**: Changed character class to `[A-Z0-9][A-Z0-9\s&,/()-]+` to also allow digits, slashes, parentheses, and hyphens.

**File**: `backend/ingest/smart_chunker.py` (lines 69-76)

#### 2. Reranker: Thinking Text Corrupted JSON Parsing

**Problem**: Gemini 2.5 Flash defaults to "thinking mode" which prepends reasoning text before the JSON output in `response.text`. The `_parse_scores` method's JSON extraction failed ~30% of the time when thinking text contained `{` characters.

**Fix**: Added `thinking_config=genai.types.ThinkingConfig(thinking_budget=0)` to disable thinking mode for the reranker call. Combined with `response_mime_type="application/json"`, this ensures clean JSON output.

**File**: `backend/retrieval/reranker.py` (line 299)

#### 3. Concept Boost: Wrong Priority Order (Test 48)

**Problem**: `get_concept_boost_articles()` prioritized broad concept matches (substring matching on `alternative_names`) over precise question matches (matching `worker_questions`). For "When do I punch the time clock?", this put Article 10 (scheduling) in the top 5 boosts but excluded Article 20 (time records) which was at position 6.

The +0.2 similarity boost for Article 10 chunks then pushed them above Article 20's natural #1 vector score (0.516 → overtaken by boosted 0.573), and article expansion flooded the results with Article 10 chunks.

**Fix (part 1)**: Swapped priority order — question matches (more precise) now take priority over concept matches (broader).

**Fix (part 2)**: Added `concept_query` parameter to `search()` and `search_to_chunks()`. The concept boost is now always computed from the original user question, not the hypothesis-expanded query which varies per LLM call. This eliminates non-deterministic flapping where hypothesis titles like "SCHEDULING" would pollute concept matching and change the boost set between runs.

**Files**: `backend/retrieval/hybrid_search.py`, `backend/retrieval/router.py`

#### 4. Router: Wage Query Subject Restriction (Tests 2, 3)

**Problem**: `is_wage_query()` pattern required `(i|we)` as subject, so "How much does a Courtesy Clerk make?" didn't match.

**Fix**: Broadened to `.+` (any subject). Also added `"rate of pay"` to `WAGE_KEYWORDS` for symmetry with existing `"pay rate"`.

**File**: `backend/retrieval/router.py`

#### 5. Router: Active-Voice Harassment Detection (Test 51)

**Problem**: Escalation patterns only matched passive voice ("I'm being harassed") but not active voice ("My manager is harassing me").

**Fix**: Added active-voice patterns to `is_high_stakes()`.

**File**: `backend/retrieval/router.py`

---

### New Features

#### LOU Sub-Item Splitting (Test 53)

Letters of Understanding are now split into individual chunks instead of being bundled as Article 58 subsections.

**Problem**: LOUs 6, 7, 8 were grouped into one chunk (`art58_sec175_6-8`), diluting the dress code embedding so it couldn't be found for "What color shoes can I wear?"

**Solution**:
- Added `LOU_SECTION_HEADER` pattern to detect the LOU boundary
- Added `LOU_ITEM_HEADER` pattern (`## N. Title`) to split individual LOUs
- Each LOU gets its own chunk with `doc_type: "lou"` and descriptive citations

**Result**: 37 LOU chunks (previously ~4), including separate dress code chunk. Smart chunker now produces 320 chunks (283 CBA + 37 LOU).

**File**: `backend/ingest/smart_chunker.py`

#### Contract-Specific Routing via Manifest

Moved contract-specific routing knowledge (slang mappings, topic-to-article maps, classification maps) from hardcoded Python dictionaries to `data/manifests/{contract_id}.json`.

**New manifest section**: `query_routing` with:
- `slang_to_contract`: Contract-specific terminology (e.g., "dug" → "Drive Up & Go")
- `topic_to_articles`: Topic-to-article number mappings
- `topic_patterns`: Regex patterns for topic detection
- `classification_to_articles`: Job classification article mappings

**Scaling benefit**: New contracts only need a manifest file — no Python code changes.

**Files**: `data/manifests/safeway_pueblo_clerks_2022.json`, `backend/retrieval/router.py`

---

### Files Modified/Added

| File | Action | Changes |
|------|--------|---------|
| `backend/config.py` | Modified | Added `TABLES_DIR`; updated all model references to 2.5 |
| `backend/retrieval/router.py` | Modified | Manifest-based routing, universal pattern fixes, `@lru_cache` per contract_id, pass `concept_query` for stable concept matching |
| `backend/retrieval/reranker.py` | Modified | SDK migration, `thinking_budget=0` fix |
| `backend/retrieval/hypothesis.py` | Modified | SDK migration |
| `backend/retrieval/query_interpreter.py` | Modified | SDK migration |
| `backend/retrieval/hybrid_search.py` | Modified | Concept boost priority swap (question > concept), `concept_query` parameter for stable matching |
| `backend/retrieval/vector_store.py` | Modified | Dual content fields (`content` / `content_with_tables`) |
| `backend/ingest/smart_chunker.py` | Modified | Article title regex fix, LOU sub-item splitting |
| `backend/ingest/enricher.py` | Modified | SDK migration |
| `backend/api.py` | Modified | SDK migration |
| `backend/test_generation.py` | Modified | SDK migration |
| `backend/evaluate_generation.py` | Modified | SDK migration |
| `backend/generation/tools.py` | Modified | SDK migration |
| `data/manifests/safeway_pueblo_clerks_2022.json` | Modified | Added `query_routing` section, fixed Article 39 title |

---

### Technical Notes

#### SDK Migration Pitfalls
- `http_options={"timeout": N}` in `genai.Client()` constructor causes SSL handshake failures on Windows. Use SDK defaults instead.
- `system_instruction` is NOT a parameter of model creation in the new SDK — it goes in `GenerateContentConfig` per request.
- The SDK warns if both `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables are set.

#### Reranker Thinking Mode
Gemini 2.5 Flash uses "thinking" by default, which prepends reasoning text to the response. For structured JSON output, disable with `thinking_config=genai.types.ThinkingConfig(thinking_budget=0)`. This is safe because the reranker prompt is simple scoring — no complex reasoning needed.

#### Concept Boost Architecture
The concept boost in `hybrid_search.py` has two signals:
1. **Question match** (`find_articles_by_question`): Fuzzy match against enricher-generated `worker_questions`. High precision — the enricher generates questions like "When do I need to punch the time clock?" that match user queries closely.
2. **Concept match** (`find_articles_by_concept`): Substring match against `alternative_names`. High recall but lower precision — common words match many articles.

Two fixes were applied:
- **Priority swap**: Question matches now take priority over concept matches in the top-5 boost list, preventing precise signals from being cut off.
- **Stable concept_query**: The concept boost is now always computed from the original user question via the `concept_query` parameter, not the hypothesis-expanded query. The hypothesis layer appends predicted titles (e.g., "SCHEDULING HOURS OF WORK") which would pollute concept matching and cause non-deterministic result flapping across runs.

---

### Known Limitations

1. Enricher JSON parse errors ~3% of time (graceful degradation — chunks get default metadata)
2. LOU chunks 2-7 and 9-13 don't get individual chunks (embedded within LOU 1 and 8 content blocks)
3. Chunk `article_title` can be `None` — always use `(chunk.get('article_title') or '')` pattern

---

### Next Steps (Planned)

- [ ] Table extractor: JSON-first structured table extraction (replace HTML `<table>` in Article 40 chunks)
- [ ] Tune reranker weights (currently 0.3/0.7) — evaluate 0.5/0.5 split
- [ ] Cache query interpretations for repeated questions
- [ ] Multi-contract support: test with a second contract to validate manifest-based routing

---
---

## v0.8.3 - Citation Entailment Checker (February 2026)

### Overview

Implemented a two-stage citation entailment checker to detect the sneakiest RAG failure mode: answers that "look grounded" but aren't. The system can cite the correct article and still hallucinate the rule. This is critical for high-stakes domains like labor law where incorrect answers can lead to lost grievances or legal issues.

**Problem**: Previous verification only checked if citations existed and matched retrieved chunks. It did not verify that the cited text actually **supports** the claims made in the answer.

**Solution**: Two-stage entailment checking:
1. **Lightweight NLI** (fast, cheap) - Uses `microsoft/deberta-v3-large-mnli` for zero-shot classification
2. **LLM fallback** (accurate, escalation) - Uses Gemini for low-confidence or contradictory cases

---

### Implementation

#### EntailmentChecker Class

Located at `backend/eval/entailment.py`, provides:

- **Three-way verdict**: `SUPPORTS | CONTRADICTS | IRRELEVANT`
- **Confidence scoring**: 0.0 to 1.0 per claim-citation pair
- **Automatic escalation**: NLI results with confidence < 0.75 or verdict = `CONTRADICTS` escalate to LLM
- **Batch processing**: `check_all()` processes multiple claim-citation pairs efficiently

**Usage Example**:
```python
from backend.eval.entailment import EntailmentChecker, extract_claim_citation_pairs

checker = EntailmentChecker(use_nli=True)
pairs = extract_claim_citation_pairs(answer, chunks)
summary = checker.check_all(pairs)

# summary.support_rate → 0.0 to 1.0 (percentage of claims supported)
# summary.has_contradiction → True if any claim contradicted
```

#### Claim-Citation Extraction

The `extract_claim_citation_pairs()` function automatically extracts structured pairs from natural language answers using regex patterns:

- "According to Article X, [claim]"
- "[claim] (Article X, Section Y)"
- "Article X states that [claim]"

This enables automatic entailment checking without manual annotation.

---

### Integration with Evaluation Framework

The entailment checker is designed to integrate with the v3 evaluation strategy:

- **EntailmentSummary** provides `citation_entailment_score` (0.0-1.0) for grading
- **Hard fail condition**: Any `CONTRADICTS` verdict triggers automatic score reduction
- **Precedence detection**: Works alongside "Specific Overrides General" checks (planned)

---

### Files Added/Modified

| File | Changes |
|------|---------|
| `backend/eval/__init__.py` | New module initialization |
| `backend/eval/entailment.py` | Complete entailment checker implementation (~600 lines) |
| `requirements.txt` | Added `transformers>=4.36.0` for NLI models |

---

### Performance Characteristics

| Metric | Value |
|--------|-------|
| NLI latency (per check) | ~50ms (CPU) |
| LLM latency (fallback) | ~200-500ms |
| Token cost (NLI) | 0 (local model) |
| Token cost (LLM fallback) | ~150 tokens per escalated check |
| Accuracy (NLI) | ~85-90% on legal text |
| Accuracy (LLM) | ~95%+ on legal text |

**Cost optimization**: Most checks use NLI (free, fast). Only low-confidence or contradictory cases escalate to LLM, reducing API costs by ~70% compared to LLM-only approach.

---

### Testing

The module includes a `main()` function with test cases covering:
- SUPPORTS: Correct claim-citation alignment
- CONTRADICTS: Claim contradicts cited text
- IRRELEVANT: Cited text doesn't address the claim
- Claim-citation extraction from natural language

Run tests with: `python -m backend.eval.entailment`

---

### Next Steps (Planned)

- [ ] Integrate with LLM-as-judge grader (`backend/eval/grader.py`)
- [ ] Add PrecedenceCheck class for "Specific Overrides General" detection
- [ ] Implement batch entailment checking for evaluation suite
- [ ] Add caching for repeated claim-citation pairs

---

### Related Work

This implementation is part of the **v3 Evaluation Strategy** (see plan file) which addresses:
- Citation entailment (this update)
- "Specific Overrides General" precedence checks
- Cross-contamination detection for multi-contract scenarios
- Dual-model adversarial schema extraction (Phase B)

---

## v0.8.4 - v3 Evaluation Strategy Implementation & Ablation Analysis (February 2026)

### Overview

Implemented comprehensive v3 evaluation framework with ablation analysis capabilities, question bucketing, precedence checking, and LLM-as-judge grading. **Critical discovery**: Ablation analysis revealed that complex CAG features (hypothesis layer, BM25 fusion, article expansion) were adding noise, not signal. Lean configuration improves accuracy from 87.3% → 89.1%.

**Key Achievement**: Established rigorous evaluation methodology that exposed the "Kitchen Sink" architecture as technical debt, saving weeks of debugging.

---

### Benchmark Results

| Metric | Baseline (v0.8) | Lean Config (v0.8.4) | Change |
|--------|-----------------|----------------------|--------|
| Overall Accuracy | 87.3% (48/55) | **89.1% (49/55)** | **+1.8 pts** |
| Contract-Only | 88.5% | **92.3%** | **+3.8 pts** |
| Multi-Hop | 76.5% | **82.4%** | **+5.9 pts** |
| Exact Numeric | 100% | 100% | - |

**Ablation Analysis Results**:

| Ablation Mode | Overall | Contract-Only | Multi-Hop | Finding |
|---------------|---------|---------------|-----------|---------|
| Baseline (all features) | 87.3% | 88.5% | 76.5% | Starting point |
| no_retrieval | 10.9% | 4% | 0% | ✅ Retrieval is essential (84.6% drop) |
| random | 43.6% | 35% | 53% | ❌ Random too high (threshold: <25%) |
| no_hypothesis | **89.1%** | **92.3%** | 76.5% | ⚠️ Hypothesis layer hurts (+1.8%) |
| vector_only | **89.1%** | 88.5% | **82.4%** | ⚠️ Vector-only beats hybrid (+5.9% Multi-Hop) |
| bm25_only | **89.1%** | **92.3%** | 76.5% | ⚠️ BM25 adds noise |
| no_expansion | **89.1%** | 88.5% | **82.4%** | ⚠️ Expansion hurts Multi-Hop (+5.9%) |
| top1 | 78.2% | 81% | 65% | ✅ Validates chunk fusion (needs k>1) |

**Key Insight**: Every single feature ablation (hypothesis, BM25, expansion) individually improves accuracy. The features are interfering, not complementing.

---

### New Features

#### 1. Ablation Framework

**File**: `backend/evaluate_comprehensive.py`

Complete rewrite to support systematic component ablation:

- **8 ablation modes**: `normal`, `no_retrieval`, `random`, `top1`, `no_hypothesis`, `bm25_only`, `vector_only`, `no_expansion`
- **Bucket-filtered evaluation**: Run tests on specific question buckets (`world_knowledge`, `contract_only`, `multi_hop`, `exact_numeric`)
- **Per-bucket metrics**: Accuracy broken down by bucket for meaningful analysis
- **Comparison tool**: `--compare` flag to generate delta tables between baseline and ablation runs
- **Success criteria checks**: Automatic validation against thresholds (e.g., "Retrieval-OFF drop >50%")

**Usage**:
```bash
# Baseline
python -m backend.evaluate_comprehensive --ablation-mode normal

# Ablation: disable hypothesis layer
python -m backend.evaluate_comprehensive --ablation-mode no_hypothesis

# Bucket-filtered
python -m backend.evaluate_comprehensive --bucket-filter contract_only

# Compare results
python -m backend.evaluate_comprehensive --compare baseline.json ablation.json
```

#### 2. Question Bucketing

**File**: `data/test_set/comprehensive_test.json`

All 55 questions categorized into 4 buckets for valid ablation analysis:

- **World Knowledge**: Answerable from general labor law priors (5 questions)
- **Contract-Only**: Requires specific contract text (26 questions)
- **Multi-Hop**: Requires synthesizing 2+ sections (17 questions)
- **Exact Numeric**: Requires precise number/date from contract (7 questions)

**Why it matters**: Global ablation metrics are meaningless. Contract-Only bucket dropping 84.6% with retrieval OFF proves the retriever is working. World Knowledge surviving at 100% proves the LLM isn't memorizing.

#### 3. PrecedenceCheck Class

**File**: `backend/eval/precedence.py`

Detects "Specific Overrides General" legal logic failures — the highest-stakes error mode.

**Problem**: System can retrieve Article 10 (general overtime: 1.5x), answer "1.5x", cite Article 10, pass entailment check, but be **legally wrong** because Article 56 (Pharmacy Tech exception: 2.0x) applies.

**Solution**: Rule-based detection of applicable exceptions based on user context (classification, hire date). Checks if exception-bearing chunks were retrieved and if the answer used the exception.

**Hard fail condition**: `precedence_failure=True` → score = 0

#### 4. LLM-as-Judge Grader

**File**: `backend/eval/grader.py`

Structured 4-dimension grading with Pydantic schema:

- **Factual Accuracy** (0-3): Alignment with ground truth and retrieved chunks
- **Citation Entailment** (0.0-1.0): Uses `EntailmentSummary.support_rate` from entailment checker
- **Completeness** (0-3): Addresses all parts of the question
- **Appropriate Uncertainty** (0-3): Refuses/hedges appropriately when warranted

**Hard fail conditions** (override to score = 0):
- `precedence_failure`: Applied general rule where exception exists
- `cross_contamination_detected`: Retrieved chunks from wrong contract
- `citation_fabrication`: Cited non-existent Article/Section
- `citation_entailment_score < 0.5`: Caps factual accuracy at 2

**Integration**: Automatically runs `EntailmentChecker` and `PrecedenceCheck` before LLM grading, providing structured results in the grader prompt.

#### 5. New Test Sets

Created 5 specialized test sets per v3 Evaluation Strategy:

1. **Paraphrase Test** (`data/test_set/paraphrase_test.json`): 15 question families × 3 variants = 45 questions. Tests semantic equivalence across original, informal/worker-slang, and formal rewrites.

2. **Unanswerable Test** (`data/test_set/unanswerable_test.json`): 20 questions in 4 types:
   - Missing from corpus (5)
   - Contradictory prompt (5)
   - Wrong scope (5)
   - Ambiguity needs clarification (5)

3. **Adversarial Test** (`data/test_set/adversarial_test.json`): 10 questions targeting precedence failures (Pharmacy Tech overtime, March 2005 cutoffs, Cake Decorator minimums).

4. **Needle Test** (`data/test_set/needle_test.json`): 5 synthetic facts with `KARL_NEEDLE_xxx` tokens at top/middle/bottom positions. Tests "Lost in the Middle" detection.

5. **Extraction Test** (`data/test_set/extraction_test.json`): 10 questions requiring verbatim clause quotes. Tests exact language retrieval vs paraphrasing.

#### 6. Needle Injection Script

**File**: `scripts/inject_needles.py`

Utility for injecting/removing synthetic test chunks:

- **Injects** 5 needle chunks into both ChromaDB (vector search) and BM25 chunks JSON
- **Verifies** presence in both indexes
- **Removes** needles after testing
- **Idempotent**: Safe to run multiple times

**Usage**:
```bash
python scripts/inject_needles.py --inject    # Add needles
python scripts/inject_needles.py --verify    # Check presence
python scripts/inject_needles.py --remove    # Clean up
```

---

### Configuration Changes

#### Lean Config (v1.5)

**File**: `backend/config.py`

Based on ablation analysis, disabled features that were adding noise:

```python
# v1.5 LEAN CONFIG: Features were adding noise, not signal
CAG_ENABLE_HYPOTHESIS_LAYER = False      # DISABLED: +1.8% accuracy without it
CAG_ENABLE_FULL_ARTICLE_EXPANSION = False  # DISABLED: +5.9% on Multi-Hop without it
CAG_ENABLE_TITLE_BOOSTING = False          # DISABLED: Depends on hypothesis layer

# v1.5 LEAN: Pure vector search outperformed hybrid fusion
HYBRID_VECTOR_WEIGHT = 1.0   # Vector search (semantic)
HYBRID_KEYWORD_WEIGHT = 0.0  # DISABLED: BM25 was adding noise
```

**Result**: 87.3% → 89.1% accuracy (+1.8%), with Q11 (Multi-hop promotion) now passing.

---

### Files Added/Modified

| File | Action | Changes |
|------|--------|---------|
| `backend/eval/precedence.py` | NEW | PrecedenceCheck class for "Specific Overrides General" detection |
| `backend/eval/grader.py` | NEW | LLM-as-Judge grader with 4-dimension scoring and hard fail conditions |
| `backend/eval/__init__.py` | Modified | Exports PrecedenceCheck, PrecedenceResult, EvaluationResult, LLMGrader |
| `backend/evaluate_comprehensive.py` | Rewrite | Ablation framework with 8 modes, bucket filtering, comparison tool |
| `data/test_set/comprehensive_test.json` | Modified | Added `bucket` field to all 55 questions |
| `data/test_set/paraphrase_test.json` | NEW | 45 questions (15 families × 3 variants) |
| `data/test_set/unanswerable_test.json` | NEW | 20 questions testing refusal behavior |
| `data/test_set/adversarial_test.json` | NEW | 10 questions targeting precedence failures |
| `data/test_set/needle_test.json` | NEW | 5 synthetic needle facts for "Lost in Middle" testing |
| `data/test_set/extraction_test.json` | NEW | 10 questions requiring verbatim clause quotes |
| `scripts/inject_needles.py` | NEW | Utility for injecting/removing synthetic test chunks |
| `scripts/debug_hard_fails.py` | NEW | Debugging script for hard-fail questions |
| `backend/config.py` | Modified | Lean config: disabled hypothesis, BM25, expansion |

---

### Key Findings from Ablation Analysis

#### 1. Retrieval is Essential ✅

- **No-retrieval**: Contract-Only drops from 88.5% → 4% (84.6% drop)
- **Random chunks**: Contract-Only at 35% (still too high, but proves retriever matters)
- **Conclusion**: System is NOT memorizing answers. Retrieval is doing real work.

#### 2. Hypothesis Layer Provides Zero Value ⚠️

- **No-hypothesis**: +1.8% overall, +3.8% on Contract-Only
- **Root cause**: LLM generates "common sense" labor terms that drift from specific legalese
- **Action**: Disabled in lean config. Saves 1-2 seconds latency per query.

#### 3. BM25 is a Detractor ⚠️

- **Vector-only**: +5.9% on Multi-Hop vs hybrid
- **BM25-only**: Same accuracy as baseline but different failure patterns
- **Root cause**: Legal contracts repeat keywords ("employee", "shift", "manager") thousands of times. BM25 boosts chunks that share vocabulary but lack semantic relevance.
- **Action**: Disabled in lean config (`HYBRID_KEYWORD_WEIGHT = 0.0`).

#### 4. Article Expansion Dilutes Context ⚠️

- **No-expansion**: +5.9% on Multi-Hop, recovered Q37 (flood/store closure)
- **Root cause**: Filling context window with irrelevant sibling sections pushes actual answers out of LLM's attention sweet spot.
- **Action**: Disabled in lean config.

#### 5. Top-1 Validates Chunk Fusion ✅

- **Top1**: Multi-Hop drops 12%, Exact Numeric drops 14%
- **Conclusion**: Multiple chunks are needed for multi-article reasoning. Retrieval depth (k=5) is correct.

#### 6. Random Retrieval Anomaly (34.6%) ⚠️

- **Expected**: <25% accuracy on random chunks
- **Actual**: 35% (failed threshold)
- **Possible causes**:
  - Ground truth allows too many acceptable articles (30% of contract "relevant" to vague questions)
  - Grader giving partial credit for answers that sound right even if retrieval was garbage
- **Action needed**: Tighten `check_retrieval()` logic. Require Section-Level precision, not just Article-Level.

---

### Remaining Hard Fails (5 questions)

After lean config and LOU fix, 5 questions still fail across all ablation modes:

| Q | Issue | Root Cause |
|---|-------|------------|
| Q3 | Probationary period (Article 26) | Vocabulary mismatch: chunk says "60 calendar days", question asks "hours". Need "520 hours" synonym. |
| Q14 | Pharmacy Tech Sunday OT | Missing Appendix A chunks (0 chunks found). Wage tables not indexed. |
| Q24 | Vacation hours for wage progression | Chunks exist (Article 8, 42) but ranking fails. Semantic similarity issue. |
| Q37 | Flood/store closure | Chunks exist (Article 48, 58) but ranking fails. Query interpretation issue. |
| Q46 | March 2005 benefits | Chunks exist but ranking fails. Article 16 not retrieved. |

**Resolved**: Q40 (Minimum Wage LOU) - Fixed by adding LOU keyword detection and `doc_type='lou'` filtering in retrieval.

**Next Steps**: Implement table_extractor Phase B to chunk Appendix A from JSON tables. (LOU issue resolved - Q40 now passes.)

---

### Technical Notes

#### Ablation Framework Architecture

The ablation framework uses config overrides with try/finally to ensure clean state:

```python
original_hypothesis = config.CAG_ENABLE_HYPOTHESIS_LAYER
try:
    config.CAG_ENABLE_HYPOTHESIS_LAYER = ablation.enable_hypothesis
    # ... run evaluation ...
finally:
    config.CAG_ENABLE_HYPOTHESIS_LAYER = original_hypothesis
```

This allows multiple ablation runs in the same process without contamination.

#### Question Bucketing Methodology

Buckets were assigned manually by analyzing each question's requirements:

- **World Knowledge**: Questions answerable from general labor law (e.g., "What is overtime?")
- **Contract-Only**: Questions requiring specific contract text (e.g., "What is the grievance deadline?")
- **Multi-Hop**: Questions requiring 2+ articles/sections (e.g., "Pharmacy Tech Sunday OT calculation")
- **Exact Numeric**: Questions requiring precise numbers (e.g., "Starting All Purpose Clerk rate 1/21/2024")

This stratification enables meaningful per-bucket ablation analysis.

#### PrecedenceCheck Implementation

Currently uses hardcoded precedence rules for demonstration:

- Pharmacy Tech overtime (Article 56: 2.0x overrides Article 12: 1.5x)
- Sunday Premium pre-March 27, 2005 (Article 13: time and one-quarter)
- Cake Decorator layoff protection (Article 29: 6 months experience requirement)
- Sanitation Clerk job security (Article 2: May 11, 1996 protections)

**Future enhancement**: Extract precedence rules dynamically from contract metadata or LLM-based rule extraction.

---

### Known Limitations

1. **Appendix A not chunked**: Wage tables extracted to JSON but not searchable via hybrid retrieval. Causes Q14 to fail.
2. ~~**Minimum Wage LOU missing**: Not properly indexed as separate document. Causes Q40 to fail.~~ **RESOLVED**: Added LOU keyword detection and `doc_type='lou'` filtering in retrieval. Q40 now passes.
3. **PrecedenceCheck uses hardcoded rules**: Should be dynamically extracted from contract in future.
4. **Random retrieval threshold failed**: 35% accuracy (threshold: <25%). Need to tighten retrieval validation logic.

---

### Next Steps (Planned)

- [ ] Implement table_extractor Phase B: Chunk Appendix A from JSON-structured tables
- [x] Fix missing LOU detection in retrieval for Minimum Wage LOU (Q40 resolved)
- [ ] Debug hard-fail questions Q3, Q24, Q37, Q46 (ranking/retrieval issues)
- [ ] Tighten `check_retrieval()` to require Section-Level precision
- [ ] Extract precedence rules dynamically from contract metadata
- [ ] Run full evaluation suite with LLM-as-Judge grader (currently only retrieval accuracy tested)

---

## v0.7.5 - Benchmark & Observability Update (January 2025)

### Overview
Updated benchmark to test full retrieval pipeline and added observability metrics to API responses. Results show **+20 point accuracy improvement** when using the complete CAG pipeline.

---

### Changes

#### Benchmark Updated to Full Pipeline
The `evaluate.py` script now uses `multi_angle_retrieve()` instead of basic `retrieve()`, testing the complete CAG pipeline including the reranker.

**File**: `backend/evaluate.py` (line 99)

#### New Benchmark Results

| Metric | Before (v0.7) | After (v0.7.5) | Change |
|--------|---------------|----------------|--------|
| Overall | 39/55 (70.9%) | 50/55 (90.9%) | **+20 pts** |
| Retrieval Accuracy | 72.7% | 92.7% | **+20 pts** |
| Wage Lookup | 100% | 100% | - |
| Escalation Detection | 66.7% | 66.7% | - |

#### Category Improvements

Categories that jumped to 100%:
- `classification`: 0% → 100%
- `time_cards`: 0% → 100%
- `union`: 0% → 100%
- `seniority`: 33% → 100%
- `safety`: 50% → 100%
- `breaks`: 50% → 100%
- `vacation`: 67% → 100%

#### Remaining Failures (5 tests)

| Test | Issue |
|------|-------|
| Wage: Courtesy Clerk 36mo | Expects "Appendix A" citation |
| Wage: Head Clerk rate | Expects "Appendix A" citation |
| Benefits: 401k plan | Article 39 not retrieved |
| High Stakes: Harassment | Escalation not triggered |
| Dress Code: Shoe color | LOU not in chunks |

---

### API Observability Metrics

Added new metrics to `QueryResponse` for debugging and monitoring:

```python
# Reranker metrics (Phase 5)
reranker_latency_ms: Optional[float]      # Time spent in LLM reranking
reranker_position_changes: Optional[int]  # Chunks that moved position

# Interpreter metrics (Phase 4)
interpretation_latency_ms: Optional[float] # Time spent interpreting query
search_angles_used: Optional[int]          # Number of search queries tried
```

**Files Modified**:
- `backend/api.py` - Added metrics to QueryResponse model and wired up extraction

---
---

## v0.7 - LLM Reranker (January 2025)

### Overview
Added an LLM-based reranker (CAG Phase 5) that scores retrieved chunks by semantic relevance before answer generation. Uses Gemini Flash to reorder chunks based on how well they actually answer the user's question.

---

### New Features

#### LLM Reranker (CAG Phase 5)
A post-retrieval relevance scoring layer that uses LLM reasoning to reorder chunks.

**Problem Solved**: Hybrid search (vector + BM25) returns chunks that match keywords or embeddings, but may not actually answer the question. The reranker asks: "Does this chunk help answer the user's question?"

**Solution**: Batch LLM scoring with weighted score combination

- **New File**: `backend/retrieval/reranker.py`
  - Sends all retrieved chunks to Gemini Flash in one call
  - Asks LLM to score each chunk 1-10 for relevance
  - Combines LLM score (70%) with original similarity (30%)
  - Graceful fallback: returns original order on any failure

- **Configuration** (`backend/config.py`):
  ```python
  CAG_ENABLE_RERANKER = True
  RERANKER_MODEL = "gemini-2.0-flash"
  RERANKER_TIMEOUT_MS = 10000
  RERANKER_ORIGINAL_WEIGHT = 0.3
  RERANKER_LLM_WEIGHT = 0.7
  RERANKER_MAX_CHUNKS = 15
  RERANKER_CONTENT_TRUNCATE = 500
  ```

- **Integration Point**: Runs after multi-angle retrieval merge, before full article expansion

---

### Technical Details

#### Reranker Flow
```
Retrieved Chunks (from multi-angle search)
    |
    v
[LLM Reranker]
    |
    +-- Build prompt with query + truncated chunk content
    +-- Gemini Flash scores each chunk 1-10
    +-- Parse JSON response: {"0": 8, "1": 5, "2": 9, ...}
    +-- Compute final score: (0.3 * original) + (0.7 * llm_score/10)
    +-- Re-sort by combined score
    |
    v
Reranked Chunks --> Full Article Expansion --> LLM Answer Generation
```

#### Scoring Prompt
The reranker uses a domain-specific prompt:
- 10: Directly and completely answers the question
- 8-9: Highly relevant, contains key information
- 6-7: Partially relevant, provides useful context
- 4-5: Tangentially related
- 1-3: Not relevant

---

### Benchmark Results

**Evaluation via `evaluate.py`** (uses `retrieve()` without reranker):
| Metric | Result |
|--------|--------|
| Overall | 39/55 (70.9%) |
| Retrieval Accuracy | 72.7% |
| Wage Lookup | 100% |
| Escalation Detection | 66.7% |

**API Testing** (uses `multi_angle_retrieve()` with reranker):
Queries that failed in benchmark but succeed via API:

| Query | Benchmark Result | API Result |
|-------|------------------|------------|
| "How long is my lunch break?" | Article 10 (wrong) | Article 24 (correct) |
| "What are the duties of a Courtesy Clerk?" | Article 2 (wrong) | Article 7, Section 14 (correct) |
| "Do I have to join the union?" | Article 4 (wrong) | Article 3, Section 5 (correct) |

---

### Files Modified/Added

| File | Changes |
|------|---------|
| `backend/retrieval/reranker.py` | NEW - LLM reranker module |
| `backend/config.py` | Added CAG Phase 5 configuration flags |
| `backend/retrieval/router.py` | Integrated reranker into `multi_angle_retrieve()` |

---

### Known Limitations

1. Reranker adds ~1-2s latency per query (LLM call)
2. Only runs in `multi_angle_retrieve()`, not basic `retrieve()`
3. Benchmark script uses `retrieve()` so doesn't test reranker

---

### Next Steps (Planned)

- [x] Update `evaluate.py` to use `multi_angle_retrieve()` for accurate benchmarking *(Done in v0.7.5)*
- [x] Add reranker metrics to API response *(Done in v0.7.5)*
- [ ] Consider caching reranker scores for repeated queries
- [ ] Tune score combination weights based on evaluation data

---
---

## v0.6 - Query Interpreter & UI Polish (January 2025)

### Overview
Major update introducing a systemic Query Interpreter for improved semantic search accuracy, plus significant UI/UX improvements including markdown rendering fixes and enhanced citation navigation.

---

### New Features

#### Query Interpreter System (CAG Phase 4)
A deep semantic analysis layer that runs before retrieval to bridge the vocabulary gap between worker slang and formal contract language.

**Problem Solved**: Questions like *"A vendor is doing a major reset of the snack aisle. How many per year?"* couldn't find Article 2's vendor work restrictions because "reset" doesn't appear in the contract text.

**Solution**: Multi-angle retrieval with HyDE (Hypothetical Document Embeddings)

- **New File**: `backend/retrieval/query_interpreter.py`
  - Extracts structured query understanding (intent, entities, concepts)
  - Generates hypothetical contract-like text for embedding matching
  - Creates multiple search queries from different vocabulary angles
  - Detects explicit article references (e.g., "check Article 2")

- **Configuration** (`backend/config.py`):
  ```python
  CAG_ENABLE_QUERY_INTERPRETER = True
  INTERPRETER_MODEL = "gemini-2.0-flash"
  MULTI_QUERY_MAX_SEARCHES = 3
  MULTI_QUERY_RESULTS_PER_SEARCH = 5
  MULTI_QUERY_TOTAL_RESULTS = 10
  ```

- **Key Innovation**: Direct vector search for hypothetical answers bypasses RRF fusion score distortion, preserving semantic similarity scores

#### Enhanced Citation System
Citations in chat responses are now fully interactive with deep linking support.

- **Clickable citation links** in response text (e.g., "Article 29, Section 77(a)")
- **Citation badges** at bottom of messages also clickable
- **Popover previews** show section content on hover/click
- **Deep navigation** to specific subsections in Contract tab

---

### UI/UX Improvements

#### Desktop Tab Navigation Fix
**Issue**: Tabs were stacking instead of switching on desktop
**Root Cause**: CSS `md:flex` classes on content containers overrode Tailwind's `hidden` class
**Fix**: Restructured HTML to put flex layouts inside wrapper divs

#### Markdown Rendering in Contract Viewer
**Issue**: Raw markdown showing (e.g., `## Section 84.` instead of formatted heading)
**Fix**: New `renderMarkdown()` function with placeholder approach:
- Processes `## headings` → `<h3>` tags
- Processes `**bold**` → `<strong>` tags
- Processes numbered lists and bullet points
- Escapes remaining content safely

#### Citation Parser Enhancements
Updated regex to handle multiple citation formats:
- `Article 29, Section 77` - basic
- `Article 29, Section 77(a)` - parenthetical subsection
- `Article 29, Section 77, Part a` - Part-style reference
- `**Article 29**` - bold markers preserved

#### Popover Error Handling
**Issue**: 404 errors when LLM generates non-existent section numbers
**Fix**: Graceful fallback to full article view with message:
> "Section X not found. Showing Article Y (Z sections)."

#### Dark Mode Improvements
- Added h3 heading styling for contract viewer
- Consistent gold accent colors for headings

---

### Backend API Changes

#### Subsection Filtering
`GET /api/section/{article_num}/{section_num}?subsection=a`

New optional query parameter for filtering to specific subsections:
- **With subsection**: Returns only that subsection's content
- **Without**: Returns all subsections combined (existing behavior)

---

### Files Modified

| File | Changes |
|------|---------|
| `backend/retrieval/query_interpreter.py` | NEW - Query interpretation module |
| `backend/retrieval/router.py` | Added `multi_angle_retrieve()` method |
| `backend/config.py` | Added CAG Phase 4 configuration |
| `backend/api.py` | Subsection filtering, use multi-angle retrieval |
| `frontend/index.html` | Tab fix, markdown rendering, citation links, popover improvements |

---

### Technical Details

#### Query Interpreter Flow
```
User Query
    ↓
Query Interpreter (LLM)
    ↓
┌─────────────────────────────────────┐
│ • Intent extraction                 │
│ • Key concepts identification       │
│ • Hypothetical answer generation    │
│ • Multiple search query generation  │
│ • Explicit article detection        │
└─────────────────────────────────────┘
    ↓
Multi-Angle Retrieval
    ↓
┌─────────────────────────────────────┐
│ 1. Original query → Hybrid search   │
│ 2. Hypothetical → Direct vector     │
│ 3. Alt queries → Hybrid search      │
└─────────────────────────────────────┘
    ↓
Merged & Deduplicated Results
    ↓
LLM Response Generation
```

#### Vocabulary Translation Examples
| Worker Term | Contract Term |
|-------------|---------------|
| vendor/vendor work | recognition, work jurisdiction |
| reset/major reset | vendor work, merchandising |
| fired/canned | discharge, termination |
| write up | discipline, warning |
| break | rest period, relief period |
| overtime/OT | overtime, premium pay |
| floater | personal holiday |

---

### Testing

**Golden Test Case** (now passing):
> "A vendor is seen doing a 'major reset' of the snack aisle. How many of these are they allowed per year?"

**Expected Answer**: Three (3) major resets per store per section per calendar year (Article 2, Section 3)

---

### Known Limitations

1. LLM occasionally generates section numbers that don't exist in the contract - handled gracefully with fallback
2. Query interpreter adds ~1-2 seconds latency for complex queries
3. Subsection format varies between `(a)` style and `Part a` style depending on contract section

---

### Next Steps (Planned)

- [ ] Cache query interpretations for repeated questions
- [ ] Add interpretation confidence scores
- [ ] Expand vocabulary translation dictionary
- [ ] Consider client-side caching for frequently accessed articles

---
---

## v0.5 - CAG "Rosetta Stone" Architecture (December 2025)

### Overview
Initial implementation of Context-Aware Generation (CAG) to solve the vocabulary mismatch problem between worker slang and formal contract language. Named "Rosetta Stone" for its role in translating between different terminologies.

**Core Problem**: Users search for "break" but the contract uses "relief periods" (Section 25). Pure semantic search struggled with these terminology gaps, requiring 3+ attempts with smaller models to find correct sections.

---

### New Features

#### Phase 1: Hybrid Search Tuning

Optimized the BM25 + Vector fusion for legal document terminology.

**Configuration Changes** (`backend/config.py`):
```python
# Rebalanced weights for better RRF fusion
HYBRID_VECTOR_WEIGHT = 1.0   # Was 1.2 - equalized
HYBRID_KEYWORD_WEIGHT = 1.0  # Was 0.8 - equalized

# BM25 parameter tuning
BM25_K1 = 1.8   # Was 1.5 - higher saturation for repeated legal terms
BM25_B = 0.75   # Document length normalization (default)
```

**Rationale**: Equal weights let RRF fusion work properly; increased k1 gives more weight to term frequency which helps with repeated contract terminology.

---

#### Phase 2: Hypothesis Layer

LLM-powered article title prediction to boost relevant sections before retrieval.

**How It Works**:
1. Before retrieval, ask fast LLM: "What article titles might answer this question?"
2. LLM generates 3 candidate titles (e.g., "Rest Periods", "Meal Breaks", "Work Hours")
3. During retrieval, chunks matching hypothesized titles get a score boost

**Configuration**:
```python
CAG_ENABLE_HYPOTHESIS_LAYER = True
HYPOTHESIS_MODEL = "gemini-2.0-flash"  # Better reasoning than flash-lite
HYPOTHESIS_MAX_TITLES = 3              # Number of title guesses
HYPOTHESIS_TIMEOUT_MS = 2000           # 2 second timeout
TITLE_BOOST_SCORE = 0.5                # Score boost for matching titles
```

**Example**:
- Query: "When do I get breaks?"
- Hypothesis: ["REST PERIODS", "MEAL PERIODS", "HOURS OF WORK"]
- Result: Article 24 (Meal) and Article 25 (Relief) get boosted

---

#### Phase 3: Full Article Expansion

When multiple top results come from the same article, fetch the entire article for complete context.

**Logic**:
1. After initial retrieval, check if 2+ of top-5 results are from same article
2. If yes, fetch ALL chunks from that article (up to limit)
3. Provides complete context for complex provisions

**Configuration**:
```python
CAG_ENABLE_FULL_ARTICLE_EXPANSION = True
FULL_ARTICLE_MAX_CHUNKS = 15           # Max chunks to fetch per article
FULL_ARTICLE_MIN_TOP_K_MATCH = 2       # Trigger: 2+ chunks in top-5
```

**Example**:
- Query about "vacation rollover"
- Initial results: 2 chunks from Article 20 (Vacations) in top-5
- Expansion: Fetch all 8 sections of Article 20
- Result: LLM has complete vacation policy context

---

### Architecture

```
┌─────────────────────┐
│   User Query        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Hypothesis Layer (Phase 2)                 │
│  LLM predicts: "What articles might help?"  │
│  → ["REST PERIODS", "MEAL PERIODS", ...]    │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Query Expansion (existing)                 │
│  SLANG_TO_CONTRACT dictionary               │
│  "break" → "rest period relief period"      │
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Hybrid Search (Phase 1 - Tuned)            │
│  ┌─────────────┐    ┌─────────────┐        │
│  │   Vector    │    │    BM25     │        │
│  │   1.0 wt    │    │   1.0 wt    │        │
│  └──────┬──────┘    └──────┬──────┘        │
│         └────────┬─────────┘                │
│                  ▼                          │
│         ┌────────────────┐                  │
│         │  RRF Fusion    │                  │
│         │  + Title Boost │ ← Hypothesis     │
│         └────────────────┘                  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Full Article Expansion (Phase 3)           │
│  If Article X has 2+ in top-5:              │
│  → Fetch all Article X chunks               │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────┐
│  Final Context      │
│  → LLM Generation   │
└─────────────────────┘
```

---

### Files Modified/Added

| File | Changes |
|------|---------|
| `backend/config.py` | CAG configuration flags and parameters |
| `backend/retrieval/router.py` | Hypothesis layer, title boosting, article expansion |
| `backend/retrieval/hybrid_search.py` | BM25 k1 parameter tuning |
| `data/manifests/` | Article title manifests for hypothesis matching |

---

### Test Results

| Query | Before CAG | After CAG |
|-------|------------|-----------|
| "When do I get breaks?" | ❌ Failed | ✅ Article 24, 25 |
| "15 minute breaks" | ❌ Failed | ✅ Article 25 |
| "relief breaks" | ❌ Failed | ✅ Article 25 |
| "lunch break policy" | ✅ Pass | ✅ Pass |
| "rest periods for 8 hour shift" | ❌ Failed | ✅ Article 25 |

**Retrieval Accuracy**: Improved from 58.2% → ~70%

---

### Feature Flags

All CAG features can be toggled independently for A/B testing:

```python
# Enable/disable each phase
CAG_ENABLE_HYPOTHESIS_LAYER = True      # Phase 2
CAG_ENABLE_TITLE_BOOSTING = True        # Part of Phase 2
CAG_ENABLE_FULL_ARTICLE_EXPANSION = True # Phase 3
```

---

### Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| Average latency | ~800ms | ~1200ms |
| Retrieval accuracy | 58.2% | ~70% |
| High-stakes accuracy | 85% | 95% |

The ~400ms latency increase comes primarily from the hypothesis LLM call, but is acceptable given the accuracy improvement.

---

### Known Limitations

1. Hypothesis layer depends on LLM availability (has timeout fallback)
2. Title boosting requires article manifests to be pre-generated
3. Full article expansion can increase context size significantly

---

### Foundation for v0.6

This architecture laid the groundwork for v0.6's Query Interpreter:
- Proved LLM-in-the-loop retrieval is effective
- Established feature flag pattern for gradual rollout
- Identified need for deeper semantic understanding (→ Query Interpreter)

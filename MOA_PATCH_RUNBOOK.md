# MOA Patch Runbook (v0.9.0)

## 1) Register shared MOA/source document

Use shared source-doc storage for MOAs that affect multiple contracts:

`data/source_docs/<doc_type>/<source_doc_id>/`

Expected artifacts in each folder:

- `original.pdf`
- `extracted.json` (optional)
- `extracted.md` (optional)
- `metadata.json` (document date + ratified date + hashes + impacted contracts)

CLI:

```bash
python -m backend.ingest.register_source_doc \
  --source-doc-id albertsons_safeway_moa_2025_07_05 \
  --doc-type moa \
  --document-date 2025-07-05 \
  --ratified-date 2025-07-05 \
  --pdf-path "D:/Downloads/Signed+MOA+-+July+5,+2025+(Safeway).pdf" \
  --json-path "D:/Downloads/safeway_moa_parsed.json" \
  --md-path "D:/Downloads/safeway_moa_parsed.md" \
  --contract-id local7_safeway_pueblo_clerks_2022 \
  --contract-id local7_safeway_denver_clerks_2022 \
  --party "UFCW Local 7" \
  --party "Safeway"
```

## 2) Add patch JSON

Create:

`data/contracts/<contract_id>/amendments/<patch_id>.json`

Use schema `moa_patch_v0_9_0` and include `expected_prev_hash` for every operation.
Patch may reference either:

- `source_pdf` (legacy/in-contract path), or
- `source_doc_id` (recommended shared source-doc registry)

Optional draft generator (from parsed MOA markdown):

```bash
python -m backend.ingest.generate_patch_drafts \
  --source-doc-id <source_doc_id>
```

Default targeting behavior:

- If `--contract-id` is omitted, targets come from `metadata.contract_ids` (or `linked_contract_ids`).
- Use `--exclude-contract-id <id>` (repeatable) to skip contracts from metadata.
- Use `--contract-id <id>` (repeatable) to override metadata targets explicitly.
- Use `--strict` to fail the command when any target contract fails.

Target report:

- `data/source_docs/<doc_type>/<source_doc_id>/patch_draft_generation_report.json`

Per-contract draft files:

- `data/contracts/<contract_id>/amendments/drafts/<draft_patch_id>.json`
- `data/contracts/<contract_id>/amendments/drafts/<draft_patch_id>.report.json`

Example explicit override:

```bash
python -m backend.ingest.generate_patch_drafts \
  --source-doc-id <source_doc_id> \
  --contract-id local7_safeway_pueblo_clerks_2022 \
  --exclude-contract-id local7_safeway_pueblo_meat_2022 \
  --strict
```

## 3) Compute expected hashes (authoring aid)

Section hash:

```bash
python -m backend.ingest.materialize_effective \
  --contract-id <contract_id> \
  --hash-section-anchor <anchor_id>
```

Table row hash:

```bash
python -m backend.ingest.materialize_effective \
  --contract-id <contract_id> \
  --hash-row-key <row_key> \
  --table-id appendix_a_wage_rows
```

## 4) Materialize effective snapshot

```bash
python -m backend.ingest.materialize_effective \
  --contract-id <contract_id> \
  --effective-version-id <effective_version_id> \
  --patch-id <patch_id>
```

## 4b) Rebase a stale patch (expected_prev_hash refresh)

Generate a rebased patch file:

```bash
python -m backend.ingest.materialize_effective \
  --contract-id <contract_id> \
  --rebase-patch-id <patch_id>
```

Overwrite the existing patch in place:

```bash
python -m backend.ingest.materialize_effective \
  --contract-id <contract_id> \
  --rebase-patch-id <patch_id> \
  --rebase-in-place
```

Write rebased output to an explicit path:

```bash
python -m backend.ingest.materialize_effective \
  --contract-id <contract_id> \
  --rebase-patch-id <patch_id> \
  --rebase-output data/contracts/<contract_id>/amendments/<patch_id>.rebased.json
```

Outputs:

- `data/contracts/<contract_id>/effective/<effective_version_id>/effective_contract.json`
- `data/contracts/<contract_id>/effective/<effective_version_id>/effective_markdown.md`
- `data/contracts/<contract_id>/effective/<effective_version_id>/build_log.json`
- `data/contracts/<contract_id>/effective/<effective_version_id>/patch_chain.json`
- `data/contracts/<contract_id>/effective/<effective_version_id>/index_inputs/*`
- `data/contracts/<contract_id>/effective/latest.json` (unless `--no-update-latest`)
  - includes `effective_version_id` and `effective_content_hash`

## 5) Rebuild retrieval index

```bash
python -m backend.ingest.rebuild_index --contract-id <contract_id>
```

`resolve_chunk_file()` and `resolve_wage_file()` now prefer latest effective `index_inputs`, so retrieval and wage lookup resolve to effective artifacts by default.

## 6) Verify

- Confirm `build_log.json` has `"status": "success"` and operation `before_hash/after_hash`.
- On apply failures, inspect `build_log.json.errors[].diagnostics`:
  - `expected_prev_hash` vs `actual_current_hash`
  - `last_touch.patch_id` / `last_touch.op_id`
  - `incoming_vs_current_diff`
- Confirm `effective_contract.json` has:
  - `effective_version_id`
  - `effective_content_hash`
  - `amendments_applied`
  - `source_documents.amendment_source_doc_ids` when shared docs are used
  - per-section / per-row provenance entries including MOA refs.
- Confirm `patch_chain.json` has:
  - ordered `applied_patch_ids`
  - per-patch `patch_payload_sha256` / `patch_file_sha256`
  - `effective_content_hash` matching `effective_contract.json` and `latest.json`.
- Confirm `/api/query` response includes:
  - `effective_version_id`
  - `amendments_applied`
  - source provenance (`source_type`, `provenance`).

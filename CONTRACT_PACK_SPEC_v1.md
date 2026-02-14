# Contract Pack Spec v1

## Purpose

Define a deterministic, auditable package format and acceptance gate for scaling KARL from single-contract operation to 100+ and 1000+ multi-tenant contracts.

The contract-pack system is ingestion-owned. Benchmark metadata consumes accepted pack hashes.

## Canonical Package Layout

Each package lives at:

`data/contracts/<contract_id>/`

Required subfolders:

- `source/`
- `manifests/`
- `chunks/`
- `wages/`
- `ontology/`

Optional but strongly recommended:

- `tables/`
- `pack/`

## Required Artifacts

- `manifests/<contract_id>.json`
- `chunks/contract_chunks_enriched_<contract_id>.json` (or `contract_chunks_<contract_id>.json`)
- `wages/wage_tables_<contract_id>.json` when `manifest.has_appendix_a=true`
- `ontology/classification_ontology.json` when manifest classifications are present
- `ontology/ingestion_review_queue.json` when unresolved/ambiguous ingestion issues exist

Review-loop companion artifact (human-maintained):

- `ontology/manual_classification_overrides.json`

## Acceptance Scorecard

Generated at:

`data/contracts/<contract_id>/pack/pack_scorecard.json`

Scorecard fields:

- `scorecard_version`
- `generated_at_utc`
- `strict_mode`
- `contract_id`
- `artifact_hashes`
- `pack_hash`
- `checks[]` (`id`, `severity`, `status`, `message`, `metrics`)
- `summary` (`required_failed`, `advisory_failed`, `pass`)

## Gate Severity Model

- `required`: fail blocks pack acceptance.
- `advisory`: fail is warning unless strict mode is enabled.

Strict mode treats advisory failures as blocking.

## Current Gate Families

- Manifest existence + schema validity
- Chunk existence + non-empty + contract scoping
- Article coverage against manifest article titles
- Table ref integrity (`chunk.table_refs` resolves to table registry IDs)
- Wage artifact existence/integrity
- Canonical wage row schema integrity (`canonical_wage_rows`)
- Canonical wage conflict/ambiguity diagnostics
- Primary wage lookup integrity
- Critical alias wage resolution (shopper/dug variants)
- Classification ontology existence + alias integrity + manifest decision coverage
- Ingestion review queue presence/schema when unresolved ingestion issues exist
- Advisory appendix-table coverage signal

## Pack Registry

Accepted packs are recorded in:

`data/contracts/pack_registry.json`

Registry captures:

- `contract_id`
- `pack_hash`
- `scorecard_path`
- acceptance timestamp

## Runtime Sync Policy

Onboarding may run in either mode:

- permissive: sync runtime artifacts even if pack gates fail
- enforced: block runtime sync when required gates fail

Enforced mode is required for production release paths.

## CLI Commands

Run onboarding with scorecards:

```bash
python scripts/onboard_contract_packages.py --package <contract_id>
```

Enforce required gates before runtime sync:

```bash
python scripts/onboard_contract_packages.py --package <contract_id> --enforce-pack-gates
```

Strict mode (advisory failures also block):

```bash
python scripts/onboard_contract_packages.py --package <contract_id> --enforce-pack-gates --strict-pack-gates
```

Run acceptance directly:

```bash
python -m backend.ingest.pack_acceptance --package <contract_id>
python -m backend.ingest.pack_acceptance --package <contract_id> --strict
```

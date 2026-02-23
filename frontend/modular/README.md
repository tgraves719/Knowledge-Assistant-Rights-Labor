# Modular Frontend (Primary App)

This folder is now the primary frontend app with JS externalized.

- App logic lives in `src/app.js`.
- Entry point is `main.js`.
- Primary URL: `/`
- Alias URL: `/modular`
- Assets: `/static/modular/*`

## Why this baseline exists

1. Keep behavior parity while we split concerns.
2. Refactor in slices without breaking the user flow.
3. Maintain a clear modular structure for scale features.

## Refactor slices (next)

1. `src/modules/api-client.js`
2. `src/modules/citation-router.js`
3. `src/modules/contract-navigator.js`
4. `src/modules/pdf-surface.js`
5. `src/modules/chat-surface.js`
6. `src/modules/settings-surface.js`
7. `src/modules/steward-onboarding.js` (active steward flow)
8. `src/modules/member-onboarding.js` (active member flow)

## Rule for migration

Each slice must keep behavior parity and pass manual smoke checks before moving to the next slice.

## Contract Tab Source Modes

- Contract tab now loads `/api/contract-history` for the active contract.
- PDF source selector supports `Effective (Auto)`, `Base PDF`, and `MOA PDF` when amendment PDFs exist.
- Citation clicks forward provenance hints (`source_type`, `source_pdf`, `source_page`) to `/api/pdf-location` so amended language can open directly in MOA pages while preserving base navigation mode.
- Citation badges now show provenance tags (`MOA`, `Base`, `MOA+Base`) so users can quickly tell whether language is amended or inherited.

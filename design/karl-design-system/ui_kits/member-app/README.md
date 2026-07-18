# Member App UI Kit

Interactive recreation of the KARL member-facing chat app (`frontend/modular/index.html` in the source repo), scoped to the three primary tabs a union member uses day-to-day:

- **Chat** — Q&A with quick-action suggestions, citation-linked answers, and the animated union-blue "thinking" header.
- **Contract** — table-of-contents navigation into the contract text.
- **Settings** — dark mode, data-contribution consent toggle, profile fields, delete-my-data.

This is a cosmetic recreation, not the production app: no real retrieval, no backend, no onboarding flow (see `design-doc.md`/`member-onboarding.js` in the source repo for the full animated shield-puzzle onboarding — omitted here as out of scope for a single demo screen). Colors are the shipped UFCW Local 7 demo tenant; recolor `--union-*` tokens per-union for a real deployment.

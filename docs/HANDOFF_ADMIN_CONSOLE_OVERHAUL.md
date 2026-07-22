# Agent Handoff: Admin Console Design Overhaul + Fancy QR Business Card

> Paste this whole file as your first message in a fresh session. It is
> self-contained. Everything you need is either in this repo or in the design
> system zip noted below.

## What this project is

**KARL** (Knowledge Assistant for Rights & Labor) is a per-union contract-Q&A
chatbot. Members scan a QR code, join with zero friction, and get contract
answers with citations. **KARL Stewardship (KARLS)** is the parent org. The
codebase is a FastAPI backend + vanilla-JS frontend monorepo. The member app is
already skinned to a real design system (deep union-blue + gold, Inter +
Playfair-italic). The **admin consoles are not** — they're generic slate
Tailwind — and that's what this task fixes.

## The job (two parts)

1. **Restyle the two admin consoles** to the KARL Design System, keeping 100% of
   their functionality.
2. **Redesign the printable QR "business card"** so it looks like the member
   *onboarding scene* (dark union-blue gradient, gold glow, Playfair italic
   hero) instead of the current plain light card.

The user explicitly asked for "a fancy QR code printed on a business card, in
the design of the onboarding / beginning text style."

## Read first (in this order)

1. `C:\Users\Thomas\Downloads\KARL Design System.zip` — the design system. Unzip
   and read `readme.md`, `SKILL.md`, `tokens/*.css` (colors/typography/spacing/
   effects/base), and the `guidelines/*.card.html` specimens. **The `tokens/`
   values are the source of truth** — pull them in as CSS variables.
2. `docs/DESIGN_SYSTEM_NOTES.md` — the repo's own design notes.
3. `frontend/modular/join.html` — the member QR landing page; it *is* the
   onboarding scene the business card should echo (dark blue gradient, gold
   radial glow, Playfair italic "Know your contract.", single gold CTA).
4. `frontend/modular/src/modules/member-onboarding.js` /
   `steward-onboarding.js` — onboarding animation/aesthetic reference.
5. `frontend/modular/admin.html`, `superadmin.html`, `admin.js` — what you're
   restyling.

## Design system cheat-sheet (from the zip)

- **Union palette (demo tenant UFCW Local 7):** blue `#0D3B54` → `#14506E` →
  `#1B6B8A`; gold `#D4A029` / `#E8B84A`. **Gold is reserved for the single
  highest-emphasis element on screen** (primary CTA / active state) — never as a
  fill on large areas.
- **Stewardship (org) palette:** black + white only, no accent.
- **Neutrals:** slate scale (`--ink-950`…`--ink-50`, `--paper`).
- **Type:** Inter (UI, 300–700). Playfair Display *italic 600* is reserved
  **only** for the onboarding hero — use it on the business card, not in console
  chrome. Iosevka (substituted with **JetBrains Mono** — see the zip's font note;
  the real `.ttc` may be re-attached later) is the org/mono face.
- **Semantic states:** emerald/amber/rose/blue at *light-tint backgrounds with
  saturated text* — never solid semantic fills on large areas.
- **Cards:** `border` + `shadow-sm` (near-invisible). Modals escalate to
  `shadow-xl`/`2xl`. No colored left-border-accent cards.
- **Radius:** a real scale, 4px → 28px + full pills. Not one-radius-fits-all.
- **Header gradient:** animated diagonal `linear-gradient(-45deg, #0D3B54,
  #14506E, #1B6B8A …)`, slow 14s drift. Respect `prefers-reduced-motion`.
- **No** photography/illustration/emoji in product chrome. Icons are inline
  Heroicons-style outline SVG, 24×24, stroke-width 2.

## Hard constraints (never violate)

1. **Preserve every DOM hook `admin.js` binds to.** `admin.js` is ~2,900 lines
   and wires the consoles almost entirely by element **`id`** (plus some
   `onclick=` and `name=` and `querySelectorAll` on classes/data-attrs). You may
   restyle classes, wrap elements, and change layout — but **do not rename or
   remove any `id`, `onclick`, `name`, `data-*` hook, or form field** that JS
   reads. Before changing an element, grep `admin.js` for its id. If in doubt,
   keep it. A restyle that silently breaks "create invite" or "revoke" is a
   failure.
2. **Don't change data flow / logic in `admin.js`.** Styling-adjacent tweaks are
   fine; behavior changes are out of scope.
3. **CSP / offline:** the app forbids external font/CDN loads. Serve fonts
   same-origin (the app already serves woff2 from `frontend/org/fonts/` and
   `/static`) or inline as `@font-face` data URIs. No `fonts.googleapis.com`.
4. **Don't touch** the sign-in gates (`_render_sign_in_gate` in `backend/api.py`)
   or the invite/QR endpoints' behavior — only the card's *appearance*.
5. **Both consoles share `admin.js`.** `admin.html` is the union-admin console
   (served at `/u/{slug}/admin`, and it holds the invite/QR panel).
   `superadmin.html` is the platform console (served at `/karl/`, no QR panel).
   Keep them visually consistent.

## Files you'll touch

| File | What |
|---|---|
| `frontend/modular/admin.html` | Union-admin console shell — restyle. Has the QR panel. |
| `frontend/modular/superadmin.html` | `/karl/` platform console shell — restyle. |
| `frontend/modular/admin.js` | Shared logic — **preserve all id/onclick/name hooks**; only styling-adjacent edits. |
| `backend/platform/routers/admin.py` → `invite_printable_card()` | The printable card HTML — **redesign to the onboarding scene**. Leave `invite_qr_image()` (svg/png) working. |
| (reference only) `frontend/modular/join.html`, `src/modules/*onboarding.js` | Onboarding aesthetic to echo. |

## The fancy QR business card — spec

`invite_printable_card()` in `backend/platform/routers/admin.py` currently emits
a plain light card with a segno PNG QR as a data URI. Redesign that HTML to:

- **Business-card proportions** (3.5in × 2in front; consider a 2-sided layout —
  QR-forward front, details back — with print CSS).
- **Onboarding scene look:** dark union-blue gradient background + soft gold/blue
  radial glow (lift the exact gradient from `join.html`), **Playfair Display
  italic** hero line (e.g. "Know your contract." or a union-specific line), gold
  reserved for the one accent (a hairline, the code chip, or CTA text).
- **QR prominent and high-contrast** (keep it on a light chip so it scans — a
  pure-dark QR on dark won't scan; segno can render on a white rounded tile).
- The join URL, the code, the union name, the audience (Member/Steward), and the
  placement label.
- **Print-optimized:** `@media print` removes the on-screen "Print / Save PDF"
  button, sets exact card size, avoids page-break inside the card, and keeps the
  gradient (`-webkit-print-color-adjust: exact`).
- Self-contained (QR as data URI, fonts inline/same-origin) — it opens in a new
  tab and prints.

Keep the existing `invite_qr_image()` SVG/PNG download endpoints untouched.

## Optional (user left this open earlier)

The QR/invite panel lives **only** in the union-admin console (`admin.html`).
From `/karl/` (superadmin) there's currently no direct route to it. A nice add:
an "Open union admin console" link per union on `/karl/` → `/u/{slug}/admin`.
Low effort, no duplicated UI. Do this only if it fits the pass.

## Environment / workflow

- **Repo state:** `main` and branch `test/m2-invite-flow-and-m1-corrections` are
  identical and deployed (HEAD `aebc145` at handoff). Work on a **new branch off
  `main`**.
- **Run locally:** system Python boots the app without Postgres for static/HTML
  routes (`uvicorn backend.api:app`) — good enough to eyeball the card endpoint
  and console shells. Full tenant/admin data needs `KARL_POSTGRES_URL`. The
  platform test venv is `.venv/` (see `docs/LOCAL_SETUP_WINDOWS.md` and the
  memory note on the test env).
- **Verify the card** by hitting `GET /api/admin/unions/{union_id}/invites/{invite_id}/card`
  (needs an admin session) or by unit-rendering `invite_printable_card` with a
  fake request/invite/union (see `backend/test_platform_invites.py::
  test_invite_qr_and_printable_card_render` for the pattern).
- **Verify the consoles** by loading `/u/{slug}/admin` signed in as a union admin
  and exercising every panel (unions, documents, **invites: create / list /
  revoke / download PNG / printable card**, users, quotas, telemetry, tracking
  policy). Nothing should throw.
- **CI:** `.github/workflows/eval-ci.yml` runs on push. Pure frontend/card
  changes won't affect the wage/eval gates. Don't remove `data/test_set/*_results.json`
  churn into commits — those are eval *outputs*, not fixtures.
- **Deploy** (frontend/card = no migrations): SSH `ssh -i ~/.ssh/karl_deploy
  root@159.203.91.194`, then as the `karl` user (root trips git's ownership
  guard):
  ```
  sudo -u karl bash -lc "cd /home/karl/karl && \
    git pull origin <your-branch> && \
    docker compose -f docker-compose.prod.yml build && \
    docker compose -f docker-compose.prod.yml up -d"
  ```
  Build first (old container keeps serving), then `up -d` recreates. See
  `docs/DEPLOYMENT_RUNBOOK.md`.

## Validation sequence

1. `python -m py_compile backend/platform/routers/admin.py` and
   `node --check frontend/modular/admin.js`.
2. `.venv/Scripts/python -m pytest backend/test_platform_invites.py -q` (invite +
   card render tests must stay green; update the card-render assertions if you
   change the HTML markers).
3. Manually load both consoles signed in; exercise every invite action.
4. Print-preview the business card (Chrome → Print) at Letter and confirm the
   card size, gradient, and QR scannability.
5. Don't regress the sign-in gates or the member app.

## What success looks like

- Both admin consoles read as the same brand as the member app — union-blue
  surfaces, gold used sparingly for the one primary action, Inter type scale,
  semantic-tint statuses, the design system's radius/shadow.
- **Every `admin.js`-driven action still works** — zero broken bindings.
- The printable QR card looks like the onboarding scene (dark gradient, gold
  glow, Playfair italic) and prints cleanly as a real business card, QR scannable.

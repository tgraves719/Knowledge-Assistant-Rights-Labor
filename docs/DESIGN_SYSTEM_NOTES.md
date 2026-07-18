# KARL design system — working notes (2026-07-17)

The canonical design system lives in `design/karl-design-system/` (vendored from Thomas's
"KARL Design System" package; start with its `readme.md` and `tokens/colors.css`).
These notes record the decisions applied to this repo and the rules future UI work must follow.

## The two identities — never mix them

| | **KARL Stewardship (org)** | **KARL (the app)** |
|---|---|---|
| Who | The parent org: governance docs, legal instruments, the design system's own chrome, and the **root website** (what you see at karlstewardship.com *without* a QR code) | The per-union tenant product: member chat, contract viewer, settings, the QR join/onboarding flow, admin consoles |
| Color | **Black + white only. No accent color, ever.** | Tenant palette. Default/demo tenant is UFCW Local 7: deep teal-blue `#0D3B54→#1B6B8A` surfaces, warm gold `#D4A029` (light `#E8B84A` on dark) as the *single* accent, reserved for the one highest-emphasis element on screen |
| Type | Monospace org typeface (Iosevka; JetBrains Mono substitutes until the real `.ttc` files land — see design-system readme) | Inter for UI text; **Playfair Display italic 600 exclusively for the onboarding hero moment** ("Know your contract.") on a dark scene — never in ordinary chrome |
| Existing exemplar | `legal/instruments/karl_instruments.html` (white paper sheet, black ink, mono, print-first) | `frontend/modular/index.html` + `frontend/modular/join.html` |

Thomas's landing-page HTML sketches (Decision D5) are the **org site** — when they land, they
style the root of karlstewardship.com in the black/white mono identity. The app never adopts
that identity, and the org site never borrows union colors.

## Decisions applied in this repo (2026-07-17)

1. **The app is permanently dark.** Per Thomas. `frontend/modular/index.html` boots with
   `<html class="dark">`, `initDarkMode()` always applies the dark theme, the Settings
   light/dark toggle row was removed, and `toggleDarkMode()` is a safe no-op. The dark theme
   is the design system's first-class dark: slate-900/800 surfaces, gold accent lightened to
   `#E8B84A`, active toggles gold. `member-host.html` (the tenant shell around the member
   widget) uses the dark union-blue scene instead of the old light gradient.
2. **The QR join page (`frontend/modular/join.html`) is the onboarding scene.** Dark
   union-blue animated gradient (14s idle drift, `prefers-reduced-motion` respected), soft
   gold/blue radial glow, Playfair italic hero ("Know your contract."), translucent
   white/10 tiles, and a single gold CTA — the one gold element on screen. One tap calls
   `POST /api/auth/session/join-guest` and lands the member in the app (which runs member
   onboarding for first-time sessions). No sign-in, no form.
3. **Voice rules apply to all member-facing copy** (from the design-system readme):
   contract-grounded with citations, calm not clinical, second person plain language, never
   claims outcomes, no emoji, no marketing exclamation points.

## Open items

- Swap JetBrains Mono → real Iosevka when Thomas supplies the `SGr-Iosevka-*.ttc` files
  (one-file change in `design/karl-design-system/tokens/typography.css`).
- Org-site HTML (D5) pending from Thomas → becomes the root route (`/` currently redirects
  to the superadmin surface; must be replaced before karlstewardship.com goes live — tracked
  in the pilot plan, Milestone 3).
- Admin/superadmin consoles predate the design system and still use their own light styling;
  restyling them is explicitly out of scope for the pilot (internal tools), revisit post-1.0.
- Tab-bar icons in the design-system demo kit use placeholder emoji — the product itself uses
  outline SVG (Heroicons-style); keep it that way.

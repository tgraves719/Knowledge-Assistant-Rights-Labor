# KARL Design System

**KARL — Knowledge Assistant for Rights and Labor** is a contract-intelligence chatbot for union members and stewards: ask a plain-language question about your collective bargaining agreement and KARL answers from the contract itself, with citations to article/section/page. **KARLS (KARL Stewardship)** is the parent organization that stewards KARL as a product and licenses/serves it to unions — each union runs its own tenant, skinned in its own colors.

This design system covers both layers:
- **KARL Stewardship** — the org's own black-and-white identity (its docs, governance material, this design system's own chrome).
- **KARL the app** — the per-union-themed product, documented here using the shipped default/demo tenant, **UFCW Local 7**.

## Sources this system was built from

- GitHub: [`tgraves719/Knowledge-Assistant-Rights-Labor`](https://github.com/tgraves719/Knowledge-Assistant-Rights-Labor) (branch `main`) — full backend + frontend monorepo. Key files read: `README.md`, `design-doc.md`, `frontend/modular/index.html`, `frontend/modular/styles/generated.css` (compiled Tailwind, source of every color/spacing/radius/shadow value in `tokens/`), `frontend/modular/src/modules/member-onboarding.js`, `frontend/embed/karl-member.js`.
- Uploaded: `uploads/unnamed.png` — the KARL Stewardship org mark (raised fist gripping a smartphone).
- Uploaded (requested but **not actually present**): `SGr-Iosevka-*.ttc` font files — see Font substitution below.

**If you have access to this repo, go read more of it than this system captured** — in particular `frontend/modular/src/app.js` (291KB, the full chat/contract/settings app logic), `frontend/modular/admin.html` + `admin.js` (union-admin console — not recreated here), `frontend/modular/superadmin.html` (platform operator console — not recreated here), and `docs/PRODUCTION_FOUNDATION.md` for the multi-tenant architecture. This system recreates the **member** chat surface only; admin and super-admin consoles were out of scope for this pass.

## Font substitution — please re-attach

The brief specified `SGr-Iosevka-{Thin…ExtraBold}.ttc` as the org typeface, but only a logo PNG was actually found in `uploads/`. Iosevka isn't distributed on Google Fonts under its own name, so **JetBrains Mono** (full 100–800 weight range, true italics, open license) stands in as `--font-mono` for now. **Please re-attach the real Iosevka `.ttc`/`.ttf` files** and we'll swap the `@font-face` source in `tokens/typography.css` — every token and component already points at the `--font-mono` alias, so it's a one-file change. Inter (UI text) and Playfair Display italic (onboarding headline) are unchanged — both come directly from the product's own `<head>`.

## Content fundamentals

KARL's voice comes straight from `design-doc.md` and the README's "Core Principles" — it is deliberately unbureaucratic but exact:

- **Contract-grounded, never invented.** Every substantive claim is backed by a citation. When it can't find one, KARL says so plainly rather than softening it: *"I can't find that in your contract pack."* Refusal is a feature, stated as a feature.
- **Calm, not clinical.** The product is described as behaving like "a calm, citation-obsessed steward-in-your-pocket" — warm but precise, never chatty or exclamation-heavy.
- **Second person, plain language.** UI copy addresses the member directly ("What's the Sunday premium for a courtesy clerk?", "Ask about your contract…") and translates contract legalese into everyday phrasing, while still surfacing the exact clause language and article/section number.
- **Never claims outcomes.** House style bans language like "you will win this grievance." Responses are framed as *"The contract says X"* / *"This may be worth discussing with your steward"* — informational, not advisory.
- **Radically honest about limitations.** The public README states outright that its own historical "100% (55/55)" benchmark is a *development-set retrieval hit-rate, not an accuracy claim* — the org's writing style leans into precise caveats rather than marketing gloss, even in its own docs.
- **No emoji, no marketing exclamation points.** Copy in the actual product UI is spare: short labels ("Chat", "Contract", "Settings"), sentence-case body text, uppercase only for small eyebrow labels (e.g. "KARL MEMBER WIDGET").
- **Governance-forward.** Even casual product copy nods to worker control — "Union-first," "Worker-controlled," "Anti-surveillance" are treated as product features worth stating in plain text, not buried in a privacy policy.

## Visual foundations

- **Color:** Two independent palettes. KARL Stewardship (the org) is black-and-white only — no accent color, ever. The KARL app is tenant-themed; the shipped demo tenant (UFCW Local 7) uses a deep teal-blue (`#0D3B54`→`#1B6B8A`) as the primary surface color and a warm gold (`#D4A029`/`#E8B84A`) as the single accent — gold is reserved for the *one* highest-emphasis thing on screen (active tab, primary CTA, citation links, "thinking" glow). Semantic states (success/warning/danger/info) use standard emerald/amber/rose/blue at very light tint backgrounds with saturated text — never solid semantic fills on large areas.
- **Type:** Inter for all UI text (300–700). Playfair Display, italic, weight 600, is reserved *exclusively* for the member-onboarding hero moment ("Know your contract.") on a dark scene — it never appears in ordinary UI chrome. A monospace face (Iosevka, substituted — see above) is the org/stewardship typeface for KARLS' own materials.
- **Backgrounds:** No photography, no illustration, no hand-drawn texture anywhere in the product. The only "imagery" is the animated diagonal gradient on the chat header (`linear-gradient(-45deg, #0D3B54, #14506E, #1B6B8A, …)`, slow 14s drift at idle, sped to 3.2s during "thinking") and a matching darker version behind the onboarding modal with a soft radial gold/blue glow. Everything else is flat color.
- **Animation:** Purposeful and sparing — an animated gradient signals AI "thinking" (this is the *only* loading affordance, no spinners on primary states); a two-half shield mark splits apart and reveals a scanning-paper motif while retrieving; onboarding step cards fade+scale in with a spring easing (`cubic-bezier(0.34,1.56,0.64,1)`); citation popovers fade+slide up 8px over 200ms. `prefers-reduced-motion` is respected everywhere (avatar animations disable entirely). No infinite decorative loops outside the explicit "thinking" and idle-header-drift states.
- **Hover/press states:** Hover mostly deepens or brightens the existing color (`ufcw-blue` → `ufcw-blue-mid`; white/10 tiles → gold/30) rather than introducing a new color. No scale/shrink press states observed — presses are color-only.
- **Borders & shadows:** Cards are `border border-slate-200` + `shadow-sm` (a near-invisible 1px/2px shadow) — never a heavy drop shadow on a resting card. Modals and popovers escalate to `shadow-xl`/`shadow-2xl`. No colored left-border-accent cards anywhere in the source.
- **Corner radii:** A real, wide scale is in active use — `4px` (chat-bubble tail corner) up to `28px` (onboarding modal card, embed-widget shell), plus full pills for avatars/toggles/tab highlights. Nothing in the product snaps to a single "one radius fits all" system.
- **Transparency & blur:** Used specifically for two things — translucent white tiles over the header gradient (`bg-white/10`, `backdrop-blur`) and modal backdrops (`rgba(30,58,138,0.8)` + `blur(4px)`). Never used as a generic "glass" decoration elsewhere.
- **Dark mode:** A real, first-class theme (not just an inverted filter) — surfaces flip to slate-900/800, the gold accent gets *lighter* (`#E8B84A`) for contrast, and settings toggles that are normally union-blue become gold when active in dark mode.
- **Layout:** Mobile-first with a fixed bottom tab bar and fixed header; desktop promotes the tab bar to a fixed top bar under the header. Chat, Contract, and Settings are the three primary tabs; Contract splits into a TOC list + text/PDF panel.

## Iconography

The product's icons are **inline SVG, Heroicons-style outline glyphs** (24×24 viewbox, `stroke-width="2"`, rounded line caps) — e.g. the citation-popover close (X) and "open in contract viewer" external-link icons in `index.html`. These are simple enough to be redrawn faithfully as outline strokes; no icon font or sprite sheet is used, and PNG icons don't appear anywhere. Emoji are **not** used in the real product (this design system's UI-kit demo uses a few emoji as icon placeholders in the tab bar for speed — replace with proper outline SVGs, matching the popover close/external-link icons, before shipping). No unicode-character icon system is used natively by the product.

## Repository index

```
KARL Design System/
├── styles.css                  ← single entry point, @imports everything below
├── tokens/
│   ├── colors.css              (stewardship + union + semantic palettes)
│   ├── typography.css          (font stacks, type scale — @font substitution flagged)
│   ├── spacing.css             (space/radius scale)
│   ├── effects.css             (shadow/motion/glow tokens)
│   └── base.css                (minimal element resets, link colors)
├── assets/
│   ├── logos/karl-stewardship-mark.png   (org logo — the only real logo provided)
│   └── fonts/                            (empty — see font substitution note)
├── guidelines/                  (12 foundation specimen cards: Colors, Type, Spacing, Brand)
├── components/
│   ├── core/        Button, Badge, Card
│   ├── forms/        Input, EmploymentOption, Toggle
│   ├── chat/          ChatBubble, CitationLink, QuickActionCard
│   ├── navigation/  TabBar
│   └── brand/        ShieldMark
├── ui_kits/
│   └── member-app/   index.html — interactive Chat / Contract / Settings recreation
└── SKILL.md          (Claude Code-compatible skill wrapper)
```

### Components

- **Button** (`primary`/`gold`/`secondary`/`ghost`/`danger`, 3 sizes)
- **Badge** (`neutral`/`citation`/`success`/`warning`/`danger`/`info` pills)
- **Card** (generic bordered surface)
- **Input** (text field, amber focus ring)
- **EmploymentOption** (two-up selectable classification card, from onboarding)
- **Toggle** (pill switch, gold when active)
- **ChatBubble** (assistant/user message bubble)
- **CitationLink** (inline dotted citation link)
- **QuickActionCard** (translucent suggestion tile for the header gradient)
- **TabBar** (app navigation, glowing active state)
- **ShieldMark** (Karl's tenant-colored split-shield brand mark/avatar)

### Intentional additions

None of the above were invented beyond what the source defines — every component has a direct counterpart in `frontend/modular/index.html` / `generated.css` (quick-action tiles, citation badges/links, chat bubbles, tab bar, employment cards, dark-mode toggle, shield avatar). No admin/super-admin-only components (data tables, ops dashboards) were built — flagged as out of scope below.

## Caveats & what to iterate on next

- **Font files never arrived.** Re-attach the real `SGr-Iosevka-*.ttc` files to get the actual org typeface instead of the JetBrains Mono substitution.
- **Admin & super-admin consoles are not recreated.** `admin.html`/`admin.js` (146KB) and `superadmin.html` define a large surface (tenant management, ingestion review queues, telemetry dashboards) this pass didn't touch — say the word and I'll scope a second UI kit from that code.
- **Member onboarding flow (the animated shield-puzzle sequence) is summarized, not rebuilt.** It's a substantial, highly-animated flow (`member-onboarding.js`, 900+ lines) — worth its own dedicated prototype if you want it pixel-accurate.
- **Icons are placeholder emoji in the tab bar demo** — the real product uses outline SVG icons; I didn't have a full icon inventory to enumerate, so I flagged the two I could verify (close/external-link) rather than guessing the rest.

**Please review the color/type/component choices above and tell me what to fix** — especially: is UFCW Local 7 the right demo tenant to anchor this on, or should the system be more explicitly "bring your own union palette" from the start? And should I build the admin console UI kit next?

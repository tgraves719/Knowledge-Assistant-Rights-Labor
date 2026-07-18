# Quarantined 2026-07-17 — corrupt MOA application

The patch `local7_safeway_moa_2025_07_05.json` (and the effective version + `latest.json` pointer materialized from it) was quarantined because all 7 of its `replace_section` operations mis-anchor **Denver Retail** MOA language onto Pueblo **Meat** sections by section-number coincidence.

Evidence:
- MOA page 1 (signed PDF, `local7_safeway_pueblo_clerks_2022/source/Signed+MOA+-+July+5,+2025+(Safeway).pdf`): "The changed contract language set forth in these proposals represent changes to the **Safeway Denver Retail Bargaining Unit language**…"
- Example corruptions: meat `art7_sec14` (Head Meat Cutter definition) overwritten by the Retail DUG Shopper definition; meat `art33_sec95` (Transfers from Store to Store) overwritten by the Retail funeral-leave section (MOA p.14 "Section 95" = funeral leave in Retail numbering).
- The draft report (`../drafts/*.report.json`) shows the generator fuzzy-matched 30 Retail-numbered candidates against the meat base; 23 were skipped, 7 "matched" by numbering coincidence and were bulk-approved at confidences as low as 0.75.
- The patch also contains **zero** wage-row operations, though the ratified 2025 MOA raised meat wages.

Until a correctly mapped meat patch is drafted and human-approved, the meat pack's effective state falls back to the 2022 base text. A proper patch must: (1) include only MOA items marked "(Applies only to Meat Contracts)" or genuinely generic items, mapped to Pueblo Meat section numbers by content; (2) include meat wage schedule rows from the TA attachments; (3) set the conforming termination date for the Pueblo Meat book.

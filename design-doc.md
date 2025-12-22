# Design Doc: Union Pocket Rep Prototype

## Purpose

Build a prototype “union chatbot” named **Karl** that behaves like a calm, citation-obsessed steward-in-your-pocket. It answers contract and process questions, supports members during stressful moments, and reduces steward time spent on repetitive lookups—without pretending to be legal counsel or a replacement for representation.

The prototype must demonstrate three things:
- It can reliably cite the contract.
- It can refuse safely when unsure.
- It can route people to the right human quickly.

## Non-goals

This prototype is **not**:
- A grievance-filing system  
- Legal advice  
- A bargaining strategy generator  
- A surveillance or monitoring product  

It will not scrape employer systems.  
It will not infer facts that are not present in the contract pack.  
It will not store member conversations by default.

## Users

**Primary member user**  
Someone who needs quick clarity on rights, procedures, timelines, scheduling, pay rules, discipline, and “what do I do next?”

**Primary steward user**  
Someone who needs fast retrieval of clauses, definitions, past proposals, and a clean summary of a member’s situation to triage and act.

**Secondary user**  
Orientation/new hires who need a guided explanation of the union, the contract, and how to get help.

## Core Value Proposition

**For members**  
Immediate, private, low-friction access to accurate contract knowledge and “next step” structure during stress.

**For stewards**  
Fewer repetitive questions, faster clause retrieval, cleaner intake summaries, and a portable knowledge base.

**For the union**  
Consistent information delivery, better documentation, and eventually (in later phases) aggregated trend visibility from opt-in worker-contributed data.

## Product Principles

- The assistant is **contract-grounded**. Every substantive claim is backed by a citation to a specific document, article, and section. When it cannot cite, it says it cannot find the answer in the provided sources and offers the safest next action, usually “contact your steward.”
- The assistant is **conservative under uncertainty**. It does not guess.
- The assistant **supports humans**. It routes to stewards and staff reps, and it never discourages human contact.
- The assistant **minimizes data**. “Private mode” is the default, and opt-in is explicit, revocable, and understandable.

## Prototype Scope

Phase 1 prototype includes three flows:
- Contract Q&A  
- Orientation  
- Contact routing  

It also includes an optional **meeting support** flow that provides scripts and checklists, but does not offer case strategy.

### Contract Q&A

The user asks a question.  
The assistant responds with:
- A plain-language explanation  
- Direct quote snippet(s)  
- Citations  

It highlights any deadlines mentioned in the text and asks only the minimal clarifying questions needed to select the right clause.

### Orientation

A guided, interactive onboarding that teaches roles, rights, and how to use the contract, adapting to classification and employment status.

### Contact Routing

“Who is my steward?” plus quick links to email/text and escalation paths.

## Data and Consent Model

Default behavior is **private**: conversations are processed but not retained beyond what is technically necessary to respond.

Opt-in behavior allows the union to retain anonymized excerpts and structured metadata for trend analysis and improving the assistant. Opt-in is presented as a separate choice from using the tool. Users get full functionality even if they opt out.

Consent language must be plain and non-coercive. Users can view what they have shared and request deletion. The system should make it easy to switch modes at any time.

The assistant should avoid collecting sensitive identifiers unless necessary for routing, and even then it should be optional. For the prototype, store no PII unless the union explicitly decides to pilot with a secure directory for steward contacts.

## Knowledge Sources

The assistant reads from a **contract pack**, which is a curated set of documents such as:
- The CBA  
- MOUs / side letters  
- Relevant policies the union distributes  
- Optionally, a steward directory  

Each document is parsed and chunked by structure:
- Article  
- Section  
- Subsection  
- Definitions  
- Appendices  

Chunks preserve headings and numbering because users and stewards need “Article 12.3” as much as they need the content.

The assistant must surface citations in every answer, ideally with article/section and a short quote.

## Interaction Design

The assistant starts by asking what the user needs:
- Contract question  
- Orientation  
- “I’m in a meeting”

### Contract Questions

It asks only the clarifiers that affect which clause applies, such as:
- Job classification  
- Shift  
- Probation status  
- Location (if multiple agreements exist)

### Meeting Support

It immediately offers the safest universal script if the meeting could lead to discipline, and it offers to help the user contact their steward.

It then provides a short checklist:
- What to write down  
- Who was present  
- What documents were given  

It does not tell the user what to admit or deny; it focuses on documenting facts.

### Contact Routing

It asks the minimum needed to find the correct steward coverage, then presents:
- Email / text buttons  
- A suggested message subject and body

## Safety and Guardrails

- The system must include a **citation-only policy**. If the retrieved sources do not directly support the answer, the assistant responds with:  
  > “I can’t find that in your contract pack”  
  and suggests the next step.
- High-stakes categories trigger escalation language by default, including:
  - Discipline  
  - Termination  
  - Harassment  
  - Safety  
  - Discrimination  
  - Retaliation  
  - Immigration-related workplace issues  

The assistant remains helpful by providing rights scripts, documentation checklists, and contact routing, but avoids guidance that could be interpreted as legal advice or representation.

The assistant should never claim outcomes (“you will win this grievance”). It should frame responses as:
- “The contract says X”  
- “This may be worth discussing with your steward”

## Technical Approach

Use a **retrieval-first architecture**.

Documents are ingested, chunked, and indexed. A retrieval layer selects the most relevant chunks for a query. The assistant generates an answer constrained to those chunks and produces citations.

A strict verification step checks that each paragraph of the answer is supported by at least one retrieved chunk. If not, the assistant either revises to remove unsupported content or refuses.

For the prototype:
- A lightweight web app is sufficient.
- The backend can be a simple API that takes a user query, performs retrieval, and returns:
  - Answer  
  - Citations  
  - Suggested next actions  

The UI can be minimal:
- Chat window  
- Mode toggle (Private vs Contribute)  
- “Contact steward” panel  

## Core Features and Acceptance Criteria

**Contract Q&A**  
Successful if, across a set of common union questions, the assistant returns correct citations and does not hallucinate clauses. The “refuse when uncertain” rate should be high rather than risky.

**Orientation**  
Successful if a new hire can complete it in under fifteen minutes, remembers who to contact, and understands at least three core rights and where to find them.

**Contact Routing**  
Successful if a user can reach the correct steward in under one minute from the assistant, with prefilled messaging.

**Meeting Support**  
Successful if it reduces panic and produces a clean factual summary the user can hand to a steward, without the assistant giving strategy.

## Pilot Plan

Pick one bargaining unit and one contract pack.

First:
- Run the prototype with a small group of stewards.
- Use a fixed set of test questions and scenarios.
- Measure time saved, accuracy, and where the assistant should refuse more often.

Then:
- Pilot orientation with new members.
- Measure completion, confidence, and whether it reduces repetitive steward questions.

## Risks and Mitigations

**Risk: hallucinated rights**  
Mitigation: retrieval-only, citation gating, and refusal by default.

**Risk: trust failure due to data fears**  
Mitigation: private mode default, explicit opt-in, revocable consent, and human-readable data policy.

**Risk: political resistance inside the union**  
Mitigation: position as steward support and member education, not replacement; pilot with steward champions; keep the initial scope intentionally narrow.

**Risk: fragmentation across locals/contracts**  
Mitigation: “contract packs” as installable units with clear versioning and provenance.

## Deliverables for the Prototype

- A working demo that can ingest a contract pack and answer common questions with citations  
- A short orientation flow  
- Steward contact routing  
- Visible refusal when source material is insufficient  
- A consent toggle  
- An explainer page that makes the data model legible

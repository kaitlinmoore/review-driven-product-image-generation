# Team Handoff Documents

These docs are for team members drafting the writeup and presentation. They
distill what the project actually built, why specific decisions were made,
and how the pieces fit together.

## How to use this folder

Read in order. The numbering reflects the suggested sequence:

1. **`01_glossary.md`** — Plain-language definitions of technical terms
   you'll need to use accurately in the writeup. Skim first, then keep open
   as a reference while reading the other docs.

2. **`02_pipeline_overview.md`** — The end-to-end architecture, with a
   diagram. The mental model for understanding what every other doc refers
   to when it says "the pipeline."

3. **`03_data_processing.md`** — How the 6 canonical products were chosen,
   how raw reviews became filtered prompts, what artifacts the data
   pipeline produced. Covers Phases 0–2.

4. **`05_decisions_log.md`** — Chronological record of the methodology
   decisions made during the project. The single best resource for
   answering "why did they do it this way?" in the writeup.

5. **`04_evaluation_plan.md`** — *(to be written after Phase 3 results
   land.)* The Q1/Q2/Q3/Q4 framework and which artifacts answer each
   question.

6. **`06_artifact_map.md`** — *(to be written after the evaluation plan.)*
   For each report section, which on-disk files contain the supporting
   evidence.

Numbers 4 and 6 are deferred because they depend on results that are still
generating.

## Audience assumption

These docs assume a smart reader who has not been in the day-to-day
implementation conversation. They explain the pipeline at a level a
sociology or policy student should be able to follow, while preserving the
technical accuracy needed to draft the report without misrepresenting what
was built.

## Style notes for the writeup

- **No individual attribution in writeup prose.** Refer to "the pipeline,"
  "the design," "the implementation" rather than naming team members in
  technical descriptions. Team-member acknowledgments belong in a separate
  contributions section, if any.
- **The AI-use disclosure is its own section.** Do not attribute coding
  decisions to AI tools inside methodology prose; the disclosure section
  handles that separately.
- **Use the terms in the glossary verbatim** when describing technical
  components. If a term feels too jargon-heavy for a sentence, rephrase
  rather than paraphrase the definition — accuracy matters more than ease
  of reading for the methodology section.

# AI Use Disclosure

This document discloses how AI tools were used in producing this project,
covering both the AI components that are part of the pipeline itself and the
AI assistants used in the development process.

## AI Models That Are Components of the Pipeline

The project's research question requires the pipeline to use AI models as
production components. These are documented in detail in `REPRODUCIBILITY.md`
and `docs/PROMPTS.md`. Summary:

| Model | Provider | Role in pipeline |
|---|---|---|
| GPT-4o-mini | OpenAI | Per-review visual content gate (binary classifier) |
| GPT-4o | OpenAI | Initial prompt synthesis from reviews + metadata |
| GPT-4o (vision) | OpenAI | Structured feature extraction from images |
| Mistral-7B-Instruct-v0.3 | Mistral AI / HuggingFace | Prompt refinement and self-rating in the agent loop (4-bit quantized) |
| FLUX.1-schnell | Black Forest Labs (via fal-ai through HuggingFace InferenceClient) | Image generation, open-weights model |
| gpt-image-1.5 | OpenAI | Image generation, proprietary model |
| CLIP, DINOv2, SigLIP | OpenAI / Meta / Google (via HuggingFace) | Post-hoc image-vs-image similarity evaluation |

All five LLM prompts that drive these components are documented verbatim in
`docs/PROMPTS.md`, with SHA-256 hashes baked into the replay cache so any
edit to a prompt automatically invalidates affected results.

## AI Tools Used During Development

**Claude Code** (Anthropic) was used as a coding assistant and writing
collaborator throughout the project. Models used include Sonnet 4.6, Opus 4.6, 
and Opus 4.7 Specific uses:

- **Code drafting and review.** Portions of the Python scripts
  in `exploration/` were drafted with Claude Code's assistance
  and reviewed by the team before integration. Scripts in 'src/' were
  written and then revised and debugged in collaboration with Claude Code.
  The team tested the code by running the full pipeline end-to-end and by
  spot-checking outputs against expected behavior.
- **Documentation.** `README.md`, `REPRODUCIBILITY.md`, `docs/PROMPTS.md`,
  `docs/decisions_log.md`, and `docs/artifact_map.md` were drafted
  collaboratively with Claude Code. The team edited and approved final
  versions.
- **Report and presentation drafting.** Claude Code contributed to
  structural framing and revision of the final report. Claude.ai was used
  in drafting presentation slides and figures. Numerical claims, analytical
  interpretations, and conclusions were verified by the team before inclusion.
- **Auditing and verification.** Claude Code performed cell-by-cell
  verification of all report tables against the underlying data in
  `eval_results/summary.csv`, identified inconsistencies, and proposed
  corrections. The team approved each correction before applying it.
- **Implementation support.** Architectural design and algorithmic choices
  originated with the team. Claude Code was used in auditing the conversion
  to repo structure (from original team notebook) and to audit and debug orchestration.

## Human-Led Decisions

The following were human decisions, with AI used in advisory or
implementation roles only:

- Research question framing and assignment-scope interpretation
- Product selection (criteria, candidate evaluation, final choices)
- Surrogate-projected adaptive iteration control: algorithmic design and
  mathematical framework
- The 13-field structured feature schema and its scoring functions
- Three-family evaluation methodology (in-loop, image-vs-image, structured
  features) and the v1/v2/v3 ablation design
- Final scope decisions about what to include in the canonical submission
  vs. exclude (development experiments not part of the final scope are
  preserved on the `experiments-archive` branch)
- All commits, pushes, and final approval of submitted work
- Final analytical interpretations and conclusions in the report

## Verification

AI-generated content was verified through:

- End-to-end pipeline execution under both record and replay modes
- Cell-by-cell verification of report tables against
  `eval_results/summary.csv` at full precision
- Cross-referencing of factual claims against source data files
- Manual review of all written content before submission
- Replay-mode reproduction of canonical results, confirming
  determinism of cached pipeline outputs

## Scope of This Disclosure

This disclosure covers the team's known and documented uses of AI in
producing the submitted package. Individual team members may have used
additional general-purpose AI tools (e.g., autocomplete, search, grammar
checking) in routine ways consistent with normal academic and professional
practice. Such background use is not exhaustively enumerated here.

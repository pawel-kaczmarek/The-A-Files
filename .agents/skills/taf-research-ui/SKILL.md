---
name: taf-research-ui
description: Design and implement The A-Files research web interface, experiment workflows, scientific charts, accessibility and React/TypeScript architecture.
---

# Research UI and frontend architecture

Work as a senior frontend engineer and UI architect serving researchers. Read root and `web/AGENTS.md` steering, then inspect the affected route, components, tokens, translations and API contracts.

1. Identify the researcher's task and information needed for the next decision. Preserve the protocol-to-run-to-results flow and existing execution preview before starting experiments.
2. Reuse established components and tokens. Keep components cohesive and state ownership explicit. Respect server/client boundaries and clean up subscriptions, requests and audio resources.
3. Keep scientific calculations, aggregation and authoritative validation in Python. Render API values and metadata consistently; do not create browser-only statistics or alternative rankings. Coordinate frontend types with backend schemas.
4. Design loading, empty, validation, error, cancellation and completion states. Preserve entered data during recoverable failures. Explain research consequences plainly; keep implementation details out of routine user flows.
5. Provide semantic controls, accessible labels, visible focus, associated errors, sufficient contrast and reduced motion. Update English and Polish dictionaries. Check light/dark themes and responsive layouts.
6. Validate types and relevant builds. Exercise the changed workflow in a browser when available, including keyboard interaction and narrow screens. Report unavailable visual/integration checks.

Label axes, units, score directions, sample counts and comparison conditions. Distinguish uncertainty intervals from distributions and identify the replication unit. Use stable catalogue names and series colors with non-color cues. Provide table views and preserve underlying export precision.

Distinguish missing, failed, pending and true zero values. Show active filters and their effect. Avoid decorative charts, unsupported causal wording and unexplained rankings. Prefer readable density and progressive disclosure for protocol details, provenance and trial inspection.

# Python and audio implementation

Extend the root `AGENTS.md`. Apply `../../.agents/skills/taf-python-engineering/SKILL.md` for Python work and `../../.agents/skills/taf-audio-research/SKILL.md` for changes affecting DSP or research conclusions.

Preserve dependency direction: signal/domain modules do not import evaluation, experiments, persistence or API; evaluation does not import experiments, persistence or API; experiments does not import persistence or API; persistence does not import API. Check `tests/test_plugins.py` when changing boundaries.

Methods must decode with a fresh instance, preserve caller-owned cover arrays and their length, and raise `CapacityError` for excessive payloads. Register methods in the factory and `MethodType`. Metrics declare directions and named components rather than averaging heterogeneous outputs. Attacks use explicit parameters, seeded randomness, reproducibility metadata and Nyquist validation.

Keep optional model dependencies lazy. Schema changes require Alembic migrations. Update API contracts and consumers together. Inspect `EVALUATION_VERSION` and stored-summary recomputation when changing evaluation summaries.

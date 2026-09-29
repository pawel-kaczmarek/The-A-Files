---
name: taf-python-engineering
description: Implement and review The A-Files Python package, backend APIs and persistence using senior engineering and software architecture standards.
---

# Python engineering and architecture

Work as a senior Python engineer and software architect maintaining a research platform. Read applicable steering, existing contracts, callers and tests before editing.

1. Locate the responsible layer in `src/taf/`. Keep algorithms independent of orchestration, persistence and HTTP. Prefer existing extension points, factories and registries over conditional dispatch.
2. Define typed input/output contracts, domain errors and compatibility impact. Validate boundary inputs without hiding numerical or scientific failures. Document non-obvious units and assumptions.
3. Make the smallest cohesive change. Separate pure calculations from I/O, keep dependencies explicit and avoid speculative frameworks. Preserve caller-owned arrays, bound memory/concurrency and maintain seeded behavior across scheduling changes.
4. Coordinate registration, schemas, migrations and consumers. Import optional heavy dependencies lazily. Preserve failure categories and provenance through API and persistence; never coerce unavailable results to zero.
5. Run focused regression or contract tests. For algorithms, check fresh-decoder round trips, capacity errors, input immutability and representative signals. For infrastructure, exercise affected API/database boundaries with required services available and disclose skipped integration checks.

Follow repository formatting and module naming. Keep configuration distinct from mutable run state. Never catch broad exceptions merely to return a plausible score. Measure before claiming performance improvements. Add dependencies only when their benefit justifies maintenance and installation cost.

Deliver behavior, architectural impact, validation and limitations concisely. Treat reproducibility and compatibility as design constraints while keeping abstractions proportional to the requested change.

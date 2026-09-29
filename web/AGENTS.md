# Research interface engineering

Extend root `AGENTS.md` and apply `../.agents/skills/taf-research-ui/SKILL.md`. Work as a senior frontend engineer and UI architect for a scientific application.

Use existing Next.js, React, TypeScript, Tailwind tokens and component conventions. Python owns scientific decisions, statistics, protocol validation and exports. Coordinate `lib/types.ts` with `src/taf/api/schemas.py`. Component titles, descriptions, references and group names (with their order) come from the catalogue API cards; never add component descriptions or group lists to `web/`.

Preserve English/Polish dictionaries, light/dark themes, keyboard navigation and reduced motion. Figures need units, comparison context, stable series colors, uncertainty labels and table views. Distinguish failed, missing, pending and zero results.

From `web/`, validate types with `npx tsc --noEmit` and relevant production changes with `npm run build`. When a dev server is active, use the `TAF_NEXT_DIST_DIR` override documented in `web/README.md`. Inspect affected screens at narrow and wide widths when browser tooling is available; report unavailable visual checks.

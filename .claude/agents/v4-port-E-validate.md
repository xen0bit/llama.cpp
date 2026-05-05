# v4-port-E-validate: validation harness (built first)

## Goal
Build the gate scripts that all subsequent v4-port-* tasks use to validate their work. Output: `tests/v4-port/{gate-loader,gate-coherence,gate-speed,gate-tools,run-all-gates}.sh` + `tool-call-fixture.json` + `README.md`.

## Plan reference
**Implementation plan at `docs/superpowers/plans/2026-05-04-v4-port-overnight.md` Task 1 (Steps 1.1 through 1.8).** Follow it exactly. Each script's content is fully specified in the plan.

## Gate (must pass before code-review)
Smoke test from Step 1.7: against the current binary (which doesn't have V4 yet), `gate-loader.sh` should FAIL (exit non-zero) with `FAIL: arch not deepseek4`. This proves the gate is correctly assertive — we want it to fail when V4 isn't there yet.

## Files
- Create: `tests/v4-port/gate-loader.sh`
- Create: `tests/v4-port/gate-coherence.sh`
- Create: `tests/v4-port/gate-speed.sh`
- Create: `tests/v4-port/gate-tools.sh`
- Create: `tests/v4-port/tool-call-fixture.json`
- Create: `tests/v4-port/run-all-gates.sh`
- Already created: `tests/v4-port/README.md`

## Out of scope
- Don't run any of the script content for "real" yet — V4 isn't ported. Just verify scripts exist, are executable, and gate-loader.sh exits non-zero on baseline.

## Definition of done
- All scripts exist and chmod +x
- `gate-loader.sh` smoke-tests correctly (returns non-zero against baseline binary)
- All committed on `feat/v4-port`
- Pushed to `mine`

# v4-port-F-integration: final integration gate + completion report

## Goal
Run all gates end-to-end against the integrated branch, write the completion report, push to `mine`, and (if all gates pass) merge `feat/v4-port` into `mine/master`.

## Plan reference
**Implementation plan at `docs/superpowers/plans/2026-05-04-v4-port-overnight.md` Task 6 (Steps 6.1 through 6.5).** Follow it exactly.

## Depends on
v4-port-D-tools must be in state `done` (or at least committed) before this task starts.

## Gate (must pass)
```bash
V4_GGUF=/Users/cchuter/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  ./tests/v4-port/run-all-gates.sh
```
Expected output ends with: `ALL GATES PASS`.

## Tasks
1. Run `run-all-gates.sh`, capture output to `/tmp/v4-port-final.log`
2. Write completion report to `docs/plans/v4-port-completion.md` per the plan's Step 6.2 template — fill in actual gate results, current SHA, push URL
3. Commit the completion report
4. Push `feat/v4-port` to `mine`
5. **If ALL gates passed**: merge `feat/v4-port` into `mine/master` via `git push mine feat/v4-port:master`. (Per dev-team.json, may_merge_to_master=true.)
6. **If any gate failed**: skip the merge. Leave `feat/v4-port` for tomorrow's review.
7. Print final status block per the plan's Step 6.5

## Definition of done
- Completion report written, committed, pushed
- If success: `mine/master` updated to match `feat/v4-port`
- Final status block printed (visible in orchestrator output)
- Task state machine: `done`

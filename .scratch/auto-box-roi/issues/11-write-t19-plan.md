# Write the frozen T19 plan

Type: task (AFK)
Status: resolved
Blocked by: 04, 05, 06, 07, 08, 09, 10

## Question

Assemble every decision on the map into `tickets/T19_auto_box_roi.md` (Goal, Decisions, Requirements, Files, Acceptance, in the style of T17/T18), add T19 to `ROADMAP.md`, and check that nothing on the map is left undecided. Resolved when the plan is committed.

## Answer

(2026-10-16) Written as `tickets/T19_auto_box_roi.md`:
- **Decisions**: §2, one subsection per area, each linking to the ticket that holds its reasoning.
- **Rest of the plan**: requirements, files and acceptance check.
- **ROADMAP**: T19 added to §5 "Extra experiments" and to the §9 checklist (unticked, not run).

While writing it, the scenario references were corrected: full-image models are the odd S01–S15 and union-ROI models the even S02–S16, as `submit_all.sh` numbers them. Nothing on the map was left undecided.

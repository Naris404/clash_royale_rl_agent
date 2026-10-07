# Issue tracker: GitHub

Issues and specs for this repo live in GitHub Issues for `Naris404/clash_royale_rl_agent`.
Use the `gh` CLI when installed; otherwise use the GitHub MCP issue tools
(`issue_read`, `issue_write`, `add_issue_comment`, `list_issues`, `sub_issue_write`).

## Conventions

- **Spec**: one issue per feature, titled `Spec: <feature>`, label `spec`.
- **Ticket**: one issue per implementation ticket, titled `<NN>: <slug>`, linked to its spec as a sub-issue.
- **Blocking**: a `**Blocked by:** #N, #M` line near the top of the issue body; a ticket is unblocked when every listed issue is closed.
- **Triage state**: recorded with labels from `triage-labels.md`.
- **Comments**: conversation history goes in issue comments.

## When a skill says "publish to the issue tracker"

Create a GitHub issue (apply the appropriate triage label).

## When a skill says "fetch the relevant ticket"

Read the issue by number: `gh issue view <n> --comments`, or MCP `issue_read`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is an issue labelled `map`; each **child** ticket is a sub-issue.

- **Child ticket**: sub-issue of the map, labelled with its type (`research`/`prototype`/`grilling`/`task`).
- **Blocking**: `**Blocked by:** #N` line in the body.
- **Frontier**: open, unblocked, unclaimed children; lowest number first.
- **Claim**: add the `claimed` label before any work.
- **Resolve**: comment the answer, close the issue, remove `claimed`, then append a context pointer to the map body.

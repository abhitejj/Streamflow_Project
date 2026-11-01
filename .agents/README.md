# Antigravity agent structure

Antigravity discovers a workspace agent at
`.agents/agents/<agent-name>/agent.md`. This package supplies eleven agents in
that supported layout. Each can be selected as a main agent or invoked as a
subagent.

Use `ml-project-coordinator` as the working project's main agent. It classifies
each request and delegates to the relevant specialists, so you don't need to
select a specialist for routine work. Start a session with:

```bash
agy --agent ml-project-coordinator
```

The coordinator cannot become the CLI's automatic default through this agent
file. Select it once when you start a project session, then give ordinary
requests in that session.

For global registration, copy each directory from `.agents/agents/` to
`~/.gemini/config/agents/`, preserving `<agent-name>/agent.md`. Then reopen
Antigravity CLI and use `/agents`, or launch a direct session with
`agy --agent <agent-name>`. Do not copy an agent over an existing global agent
with the same name without reviewing the conflict.

The custom-agent layout and frontmatter are based on the current official
[Antigravity custom subagents documentation](https://antigravity.google/docs/subagents?tab=cli).
Project-local shared rules live in `.agents/AGENTS.md` because the mirrored root
`AGENTS.md` is read-only in this workspace.

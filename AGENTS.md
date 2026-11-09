# BrainChat: rules for coding agents

BrainChat reads Twitch chat around a Moment and says what chat is doing. Real chat
must never reach a hosted AI service, and a coding agent is one.

## Never open the data folder

- The data folder is wherever `CHATBRAIN_DATA` points. Agents
  never read, list, search, copy, hash or summarise the data folder's contents or
  Label Studio's stored tasks and exports, by any tool: file tools, shell commands
  (`cat`, `ls`, `grep`, `find`, `type`, `dir`), Python, `sqlite3`, Docker, or Label
  Studio's pages and API.
- Agents may run the repo's own `uv run chatbrain …` commands, and may start the
  Label Studio container bound to `127.0.0.1` when a task says so, since
  those commands print counts and status only.
- This rule is the guard, for every tool, shell commands included.
- If a task seems to need real data, stop and ask the owner. The owner runs those
  steps.

## Tests use made-up chat only

- Tests build their data in pytest `tmp_path`. Never point a test, fixture or script
  at the data folder or at a real export.
- Made-up chat uses channels `fakechan01`…, speakers `s1`…, emote IDs `fake-<name>`.
- Never store or export `login`, `display_name`, `commenter_login`, `sender_hash`,
  any per-user ID stable across windows, or message hashes.

## Working here

- Windows Python 3.12 via `uv`, CPU only. Run the suite with `uv run pytest -q`.
- The label scheme lives in `config/labels_v1.json`; every value change bumps its
  version.
- Services bind to `127.0.0.1` only: Ollama `11434`, Label Studio `14610`, the
  Moment page `14612`.

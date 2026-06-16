# pi-keeper

A [Pi](https://github.com/earendil-works/pi-coding-agent) companion for local [llama.cpp](https://github.com/ggml-org/llama.cpp) servers — keep the context window lean and stop the server re-reading your whole chat every turn.

## Why This Exists

Run Pi against a local llama.cpp server for a while and two things bite you:

- **The window fills with junk.** Tool output dominates context — one file read or log dump is 20–50 KB. Stack a few and Pi has to summarize and drop old turns (compaction), losing detail.
- **The server keeps re-reading.** Whenever the conversation changes, llama.cpp re-processes the whole prompt from scratch (prefill) — seconds of dead time every turn on a big chat, and far worse on recurrent/hybrid models that can't reuse a partial cache.

pi-keeper leans on llama.cpp's own slot and prompt-cache machinery to dodge both. The big lever: most of what fills your window is tool output you'll never look at again, so it's written to disk and replaced with a one-line pointer. The model pulls the full thing back only if it needs it.

## What you get

Powered by the llama.cpp server (see [The server](#the-server)):

- **Off-context reads.** `keeper_read` / `keeper_debug` run in a separate server session, so a big file or a deep debugging pass never lands in your main chat.
- **Cache-aware rewinds.** `/keeper rollback` tells the server to reuse its cached progress instead of re-reading the conversation.
- **Recurrent/hybrid support.** Our fork keeps the prompt cache working for recurrent/hybrid models (e.g. `qwen35`) that upstream gives up on — so even those skip the re-read after a slot swap.

Works with any provider, no server needed:

- **Spill big outputs.** Anything over 8000 characters → saved to a file, replaced with a one-line pointer. The baseline win.
- **Durable notes.** Anything in `AGENTS.md` is pinned into the system prompt every turn, so it survives compaction. You write these yourself — there's no auto-remember yet.

This is what a spilled output looks like in the chat — a pointer instead of 52 KB:

```
[pi-keeper] saved read output (52 KB) → ref=obs_1a2b3c4d. Preview:
export function build(...) { ...
```

The model reads any slice back with `keeper_recall("obs_1a2b3c4d")`.

## Install

```bash
pi install https://github.com/nonml/pi-keeper
```

Type `/keeper` to check status. Spilling and notes work immediately; the rest needs the server below.

## The server

The server-powered features need a llama.cpp build from our fork (branch `pi-keeper`).

**Build the fork:**

```bash
git clone -b pi-keeper https://github.com/nonml/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON          # leave off -DGGML_CUDA=ON if you have no NVIDIA GPU
cmake --build build --config Release -j
```

**Already track upstream llama.cpp?** Cherry-pick the one commit instead of cloning the fork:

```bash
git remote add nonml https://github.com/nonml/llama.cpp
git fetch nonml pi-keeper
git cherry-pick pi-keeper          # if it conflicts: fix the files, then git cherry-pick --continue
cmake --build build --config Release -j
```

**Run it:**

```bash
llama-server -m your-model.gguf --slot-save-path ./slots --cache-ram 8192 --parallel 2
```

- **`--slot-save-path` is required.** Without it the slot features are off and side work silently falls back to normal reads — no error, it just doesn't help.
- `--cache-ram` lets the server stash your chat during side work and restore it. Size it for one snapshot (see [Memory](#memory-recurrenthybrid-models)).
- `--parallel 2` gives side work its own slot (best). `--parallel 1` works too, as long as `--cache-ram` is set.

pi-keeper finds the server on its own.

## Memory (recurrent/hybrid models)

For recurrent/hybrid models the server snapshots your *whole* chat state, which is large. Make `--cache-ram` (in MB) big enough to hold at least one snapshot, or your chat gets dropped from the cache and re-read anyway.

| Context | f16 KV | q8_0 KV | q4_0 KV |
|--------:|:------:|:-------:|:-------:|
| 16k | ~1.1 GB | ~0.6 GB | ~0.35 GB |
| 33k | ~2.4 GB | **1.3 GB** | ~0.7 GB |
| 64k | ~4.5 GB | ~2.5 GB | ~1.4 GB |
| 80k | ~5.6 GB | ~3.1 GB | ~1.7 GB |

Measured on `qwen35` 27B at a q8_0 KV cache (33k tokens → 1.3 GB, ~40 KB/token); other cells are rough. Size depends on chat length and KV-cache type (`--cache-type-k/-v`), **not** the model's weight quant.

## Commands

| Command | What it does |
|---------|--------------|
| `/keeper` | Show status (server, what's on/off, where notes live) |
| `/keeper rollback [n]` | Undo the last `n` messages you sent (default 1) |
| `/keeper spill on\|off` | Hide big tool outputs (on by default) |
| `/keeper side on\|off` | Read/think off to the side (on by default) |

## How It Works

pi-keeper is a set of Pi hooks plus a thin, capability-probing client for the server.

**In the chat (any provider):**

- **Spill.** A `tool_result` hook catches any output over 8000 chars (`PI_KEEPER_SPILL_CHARS`), writes it to `PI_KEEPER_WORKDIR`, and swaps in a `ref=…` pointer with a short preview. `keeper_recall(ref, start, n)` reads any slice back, so nothing is lost — it's just out of the window until needed.
- **Notes.** A `before_agent_start` hook reads `AGENTS.md` and appends it to the system prompt every turn, so it's re-sent after each compaction.

**On the llama.cpp server:**

- **Fast rewind (pin).** A `before_provider_request` hook adds `cache_prompt: true` and a fixed `id_slot` to each request. When Pi rewinds and resends a shorter context, llama.cpp matches the cached prefix and processes only the difference instead of re-reading the whole prompt.
- **Off-context reads.** With `--parallel ≥ 2`, side work runs on its own slot and is freed with a rollback when done — your main slot is never touched. On a single slot it borrows the main slot; the server's pinned-slot swap stashes the displaced conversation into the host-memory prompt cache (`--cache-ram`) and reloads it on your next turn, so your chat isn't re-prefilled.
- **Recurrent/hybrid.** Upstream disables the prompt cache for these — their state can't be sliced by position. The fork keeps it by storing the *whole* sequence state and reusing a cached entry only when its tokens are an exact prefix of the new prompt: restore it, decode just the delta, never slice. Bounded by `--cache-ram` with least-recently-used eviction.
- **Probing.** The client checks `/health`, `/slots`, and `/props` at startup and on a short TTL, and turns each feature on only when the server advertises it (`prompt_cache`, slot count, etc.). Anything missing → it falls back to a normal read. Nothing breaks.

**Deep dive:** the server side — checkpoint / rollback / fork, slot save/restore, and the whole-sequence prompt cache — is documented in the fork at [`tools/server/README-checkpoint.md`](https://github.com/nonml/llama.cpp/blob/pi-keeper/tools/server/README-checkpoint.md).

## Limitations

- **Notes are manual.** Nothing auto-fills `AGENTS.md` yet — it does nothing until you put something in it.
- **The server-powered features need the fork.** On a plain provider those tools just do a normal read (still works, just not off-context).
- **Recurrent/hybrid models use a lot of RAM.** See the table above; under-sized `--cache-ram` means your chat gets evicted and re-read anyway.

## Settings (optional env vars)

| Variable | Default | What it does |
|----------|---------|--------------|
| `PI_KEEPER_SERVER` | `http://127.0.0.1:8080` | where your llama.cpp server is |
| `PI_KEEPER_WORKDIR` | next to Pi's files | where saved outputs and notes go |
| `PI_KEEPER_SPILL_CHARS` | `8000` | hide outputs longer than this |
| `PI_KEEPER_MAIN_SLOT` | `0` | your chat's slot |
| `PI_KEEPER_SIDE_SLOT` | `2` | slot for side work (needs `--parallel ≥ 3`) |
| `PI_KEEPER_MULTIPLEX` | `0` | force single-slot side work when the server's cache state is unknown |

## License

MIT

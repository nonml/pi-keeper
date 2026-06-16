/**
 * pi-keeper — keep Pi's working context lean and persistent.
 *
 * Three jobs, layered so each works on its own and the whole thing degrades
 * gracefully when the optional local server is absent:
 *
 *   1. Durable memory      facts in AGENTS.md, re-injected every turn, survive compaction
 *   2. Output spilling     large tool outputs go to disk + leave a short recall pointer
 *   3. Off-context work     read_doc / deep_debug run in an isolated side-session so
 *                          heavy reads/reasoning never bloat the main context
 *
 * NOTE: this extension does NOT do its own context compaction. An earlier version
 * masked old tool results in the SENT context once usage crossed a budget; that
 * rewrote history mid-stream and (on a recurrent/MTP-draft model with no mid-prompt
 * checkpoints) forced a FULL prompt reprocess on every budget crossing, while still
 * not keeping context under the window. Compaction is Pi's job — its native
 * summarizer owns overflow. We only spill/point and run heavy work off-context.
 *
 * ADAPTIVE: jobs 1–2 need no server. Job 3 lights up when a local llama.cpp server
 * exposing KV slot endpoints is reachable. Capability is PROBED at runtime, cached
 * briefly, and re-probed when stale — nothing is hardcoded on/off. Server down ⇒
 * plain Pi behavior, no errors.
 *
 * Pi integration seams (verified against the installed build):
 *   tool_result          dist/core/agent-session.js  _installAgentToolHooks
 *   before_agent_start   system-prompt injection of durable memory
 *
 * Config via env (all optional): PI_KEEPER_SERVER, PI_KEEPER_WORKDIR, PI_KEEPER_SPILL_CHARS,
 *   PI_KEEPER_SIDE_SLOT.
 */
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { LlamaServerClient, type Completion } from "./server.ts";
import { DiskState } from "./state.ts";

const TAG = "[pi-keeper]";

// ---------------------------------------------------------------------------
// Tunables (env-overridable; a few also toggleable at runtime via /keeper).
// ---------------------------------------------------------------------------
const CFG = {
	// Spill tool outputs larger than this many chars to disk + leave a pointer.
	spillThresholdChars: intEnv("PI_KEEPER_SPILL_CHARS", 8000),
	// Dedicated server slot for side-sessions (server must run with enough slots).
	sideSlot: intEnv("PI_KEEPER_SIDE_SLOT", 2),
	// Stable slot for the MAIN loop, so the KV prefix cache stays warm across turns.
	mainSlot: intEnv("PI_KEEPER_MAIN_SLOT", 0),
};

const runtime = {
	spillEnabled: true,
	// Pin id_slot + force cache_prompt on outgoing requests so Pi's rewinds reuse
	// the KV prefix cache (the practical "fast rollback"). Only applied when our
	// local server is reachable; harmless toggle off for other providers.
	pinEnabled: true,
	// Off-context side-sessions (keeper_read/keeper_debug). With ≥2 slots they run on a
	// dedicated slot (always safe). With a single slot they multiplex on the main slot, which
	// only returns to a cache hit if the server's prompt cache stashes + reloads the displaced
	// conversation on the slot swap. The host-memory cache does this even for recurrent/hybrid
	// models now (whole-sequence prefix-only mode). Toggle with `/keeper side on|off`.
	sideEnabled: true,
	// Allow single-slot multiplex even when the server doesn't report whether its prompt cache is
	// on. OFF by default: if the cache is off, multiplexing wipes the main slot and forces a full
	// reprocess. When the server reports the cache IS on, multiplex is enabled automatically (no
	// opt-in). Set PI_KEEPER_MULTIPLEX=1 only to force it on a server that doesn't report cache state.
	multiplexOptIn: boolEnv("PI_KEEPER_MULTIPLEX", false),
	spillCount: 0,
};

/**
 * Plan how a side-session runs against the live server.
 *  - ≥2 slots: a dedicated side slot. The main slot is never touched, so this is always safe;
 *              the side slot's KV is freed with a rollback when the side-session ends. (stash: none)
 *  - 1 slot:   it has to multiplex on the MAIN slot, so the server's prompt cache must stash the
 *              displaced conversation and reload it on return, or the next main turn re-prefills:
 *                · cache reported ON  → multiplex; the pinned-slot swap auto-stashes/reloads the
 *                  main convo (works for recurrent/hybrid too, in whole-sequence mode). (stash: cache)
 *                · cache reported OFF → refuse (inline fallback).
 *                · unknown            → stay off unless the user opts in with PI_KEEPER_MULTIPLEX=1.
 * Returns null when side-sessions are disabled/unsafe/unreachable (callers fall back to inline).
 */
type SidePlan = { slot: number; multiplex: boolean; stash: "none" | "cache" };
function sidePlan(client: LlamaServerClient): SidePlan | null {
	if (!runtime.sideEnabled || !client.capabilities.reachable) return null;
	const caps = client.capabilities;
	if (caps.slotCount >= 2) return { slot: Math.min(CFG.sideSlot, caps.slotCount - 1), multiplex: false, stash: "none" };
	// single slot: displacing the main convo is only safe if the server's prompt cache restores it.
	if (caps.promptCache === true) return { slot: CFG.mainSlot, multiplex: true, stash: "cache" };
	if (caps.promptCache === false) return null; // cache off → multiplex would force a full reprocess
	if (!runtime.multiplexOptIn) return null; // unknown → safe default (inline); opt in to force
	return { slot: CFG.mainSlot, multiplex: true, stash: "cache" };
}

/** One-line side-session status for the doctor — says WHY it falls back to inline when it does. */
function sideStatus(client: LlamaServerClient): string {
	if (!runtime.sideEnabled) return "";
	const caps = client.capabilities;
	if (!caps.reachable) return "(no server → inline fallback)";
	const plan = sidePlan(client);
	if (plan) {
		return plan.multiplex ? `(single-slot multiplex on slot ${plan.slot}, prompt-cache restore)` : `(dedicated slot ${plan.slot})`;
	}
	if (caps.promptCache === false) return "(1 slot + prompt cache off → inline; use --parallel 2 or enable --cache-ram)";
	return "(1 slot → inline; set PI_KEEPER_MULTIPLEX=1 to force, or use --parallel 2)";
}

export default async function (pi: ExtensionAPI) {
	const client = new LlamaServerClient();
	let disk: DiskState | null = null;

	// Best-effort startup probe — short timeout, never blocks Pi on a dead server.
	await client.probe().catch(() => {});

	// ---- session lifecycle: stand up disk-backed durable memory -------------
	pi.on("session_start", async (_event, ctx) => {
		// Keep working memory OUT of the project tree — co-locate it with Pi's own
		// session data under ~/.pi/agent/sessions/<encoded-cwd>/ (same encoding Pi uses
		// in core/session-manager.js getDefaultSessionDir). PI_KEEPER_WORKDIR still overrides.
		const workdir = process.env.PI_KEEPER_WORKDIR ?? piSessionDir(ctx.cwd);
		disk = new DiskState(workdir);
		const caps = client.capabilities;
		ctx.ui.setStatus(
			"keeper",
			caps.reachable ? `keeper: server${caps.fork ? "+fork" : caps.slots ? "+slots" : ""}` : "keeper: local",
		);
	});

	// ---- before each prompt: inject durable facts into the system prompt ----
	pi.on("before_agent_start", async (event, _ctx) => {
		if (!disk) return;
		const facts = disk.readAgentsMd().trim();
		// Only inject if there's more than the empty scaffold.
		if (facts.length < 60) return;
		return {
			systemPrompt:
				event.systemPrompt +
				"\n\n## Durable working memory (pi-keeper)\n" +
				"The following facts persist across compaction. Trust them.\n\n" +
				clip(facts, 6000),
		};
	});

	// ---- before_provider_request: pin slot + warm the KV prefix cache --------
	// Adds `cache_prompt:true` and a stable `id_slot` to the outgoing payload so
	// llama.cpp reuses the cached prefix when Pi resends an edited/shorter context
	// (i.e. on /keeper-rollback). This is the practical "fast rollback" — no custom
	// provider needed. Gated on our server being reachable so other providers are
	// never touched. llama.cpp parses id_slot/cache_prompt from the request body.
	pi.on("before_provider_request", (event) => {
		if (!runtime.pinEnabled || !client.capabilities.reachable) return;
		const p = event.payload as Record<string, unknown> | undefined;
		// Only touch OpenAI-style chat payloads (what the llama.cpp provider sends).
		if (!p || typeof p !== "object" || !Array.isArray((p as any).messages)) return;
		p.cache_prompt = true;
		if (p.id_slot == null) p.id_slot = CFG.mainSlot;
		return p;
	});

	// ---- tool_result: spill huge outputs to disk, leave a recall pointer ----
	pi.on("tool_result", async (event) => {
		if (!runtime.spillEnabled || !disk || event.isError) return;
		const text = textOf(event.content);
		if (text.length <= CFG.spillThresholdChars) return;
		const rec = disk.spill(text, event.toolName);
		runtime.spillCount++;
		return {
			content: [
				{
					type: "text",
					text:
						`${TAG} ${event.toolName} output spilled (${rec.bytes} bytes). ` +
						`Full text saved as ref=${rec.ref}. ` +
						`Call keeper_recall with this ref to read a slice if you need it.\n\n` +
						`Preview:\n${text.slice(0, 1200)}`,
				},
			],
		};
	});

	// ---- keeper_read tool: off-context document read --------------------------
	// Server up  -> distill the doc on a side slot; only the extract enters main
	//               context (the full file is spilled for recall).
	// Server down -> hand the file (or a pointer if huge) straight back to Pi.
	pi.registerTool({
		name: "keeper_read",
		label: "Read Doc (side-session)",
		description:
			"Read a file/document and return only what the goal needs. When the local " +
			"server is available this runs in an isolated side-session so the full text " +
			"never bloats the main context.",
		promptSnippet: "Read a document and extract only what the goal needs (off-context when possible)",
		promptGuidelines: [
			"Use keeper_read instead of read when you only need a focused answer from a large file and want to keep main context small.",
		],
		parameters: Type.Object({
			path: Type.String({ description: "Path to the file/document to read" }),
			goal: Type.String({ description: "What to extract or answer from the document" }),
		}),
		async execute(_id, params, signal) {
			const text = readFileSafe(params.path);
			if (text == null) return errorResult(`Could not read ${params.path}`);
			await client.probe(signal);
			const plan = sidePlan(client);
			if (plan) {
				const extract = await sideSessionDistill(client, text, params.goal, plan, signal);
				if (extract != null) {
					if (disk && text.length > CFG.spillThresholdChars) disk.spill(text, "keeper_read");
					return {
						content: [{ type: "text", text: extract }],
						details: { mode: plan.multiplex ? "multiplex" : "side-session", path: params.path },
					};
				}
			}
			// Fallback: return the document (spill-and-point if very large).
			if (disk && text.length > CFG.spillThresholdChars) {
				const rec = disk.spill(text, "keeper_read");
				return {
					content: [
						{
							type: "text",
							text: `${TAG} ${params.path} is large; saved as ref=${rec.ref}. Preview:\n${text.slice(0, 2000)}`,
						},
					],
					details: { mode: "spill", ref: rec.ref },
				};
			}
			return { content: [{ type: "text", text }], details: { mode: "direct", path: params.path } };
		},
	});

	// ---- keeper_debug tool: focused reasoning in a side-session ---------------
	pi.registerTool({
		name: "keeper_debug",
		label: "Deep Debug (side-session)",
		description:
			"Spin up an isolated reasoning pass over recent context to find a root cause, " +
			"without spending main-context budget. Falls back to an inline note if the " +
			"local server is unavailable.",
		promptSnippet: "Run an isolated deep-reasoning pass (root-cause) off the main context",
		parameters: Type.Object({
			question: Type.String({ description: "The specific thing to analyze / root-cause" }),
		}),
		async execute(_id, params, signal, _onUpdate, ctx) {
			await client.probe(signal);
			const plan = sidePlan(client);
			if (!plan) {
				return {
					content: [
						{ type: "text", text: `${TAG} side-session unavailable (server unreachable) — analyze inline: ${params.question}` },
					],
					details: { mode: "fallback" },
				};
			}
			const recent = recentBranchText(ctx, 4000);
			const prompt = `Recent context:\n${recent}\n\nTask: ${params.question}\nFind the root cause. Be concise and concrete.\n`;
			const comp = await runSideSession(client, plan, prompt, 384, signal);
			if (!comp) {
				return {
					content: [{ type: "text", text: `${TAG} side-session call failed — analyze inline: ${params.question}` }],
					details: { mode: "fallback" },
				};
			}
			return {
				content: [{ type: "text", text: comp.text }],
				details: { mode: plan.multiplex ? "multiplex" : "side-session" },
			};
		},
	});

	// ---- keeper_recall tool: read a slice of a spilled observation ----------
	pi.registerTool({
		name: "keeper_recall",
		label: "Recall Spilled Output",
		description: "Read a slice of a previously spilled tool output by its ref (obs_xxxxxxxx).",
		parameters: Type.Object({
			ref: Type.String({ description: "The ref returned when the output was spilled, e.g. obs_1a2b3c4d5e6f" }),
			start: Type.Optional(Type.Number({ description: "Start offset in chars (default 0)" })),
			n: Type.Optional(Type.Number({ description: "Number of chars to read (default 4000)" })),
		}),
		async execute(_id, params) {
			if (!disk) return errorResult("disk state not initialized");
			const slice = disk.readSlice(params.ref, params.start ?? 0, params.n ?? 4000);
			if (slice == null) return errorResult(`No spilled output for ref=${params.ref}`);
			return { content: [{ type: "text", text: slice }], details: { ref: params.ref } };
		},
	});

	// ---- /keeper: single command, sub-command dispatcher ----------------------
	//   /keeper              → doctor (status)
	//   /keeper doctor       → status / diagnostics
	//   /keeper probe        → force a fresh server capability probe
	//   /keeper spill|pin|side on|off → toggle a feature
	//   /keeper rollback [n] → rewind to the n-th most recent user message
	pi.registerCommand("keeper", {
		description: "pi-keeper controls — status, server probe, rollback, and feature toggles",
		getArgumentCompletions: (prefix) => {
			const raw = prefix ?? "";
			const parts = raw.replace(/^\s+/, "").split(/\s+/);
			const trailingSpace = /\s$/.test(raw);
			const sub = (parts[0] ?? "").toLowerCase();
			// Completing the sub-command's ARGUMENT (2nd token), or the sub-command itself?
			const onArg = parts.length >= 2 || (parts.length === 1 && sub !== "" && trailingSpace);
			const argPrefix = (parts[1] ?? "").toLowerCase();

			if (onArg) {
				// on|off for the toggles — annotated with the current live state.
				if (sub === "spill" || sub === "pin" || sub === "side") {
					const cur = curToggle(sub);
					return [
						{ value: "on", label: "on", description: `enable ${sub}${cur ? "  (already on)" : ""}` },
						{ value: "off", label: "off", description: `disable ${sub}${cur ? "" : "  (already off)"}` },
					].filter((i) => i.value.startsWith(argPrefix));
				}
				// a small count hint for rollback
				if (sub === "rollback") {
					return [1, 2, 3]
						.map((n) => ({ value: String(n), label: String(n), description: `rewind ${n} user message${n > 1 ? "s" : ""} back` }))
						.filter((i) => i.value.startsWith(argPrefix));
				}
				return null;
			}

			// The sub-commands themselves — one-line help + live state, shown right in the popup.
			const subs = [
				{ value: "doctor", label: "doctor", description: "show status — server, slots, spill/pin/side, workdir" },
				{ value: "help", label: "help", description: "list all /keeper sub-commands" },
				{ value: "probe", label: "probe", description: "re-check the local llama.cpp server capabilities" },
				{ value: "rollback", label: "rollback", description: "rewind to the n-th most recent user message" },
				{ value: "spill", label: "spill", description: `big tool outputs → disk + pointer (now ${onoff(runtime.spillEnabled)})` },
				{ value: "pin", label: "pin", description: `reuse the KV cache on rewind (now ${onoff(runtime.pinEnabled)})` },
				{ value: "side", label: "side", description: `off-context keeper_read / keeper_debug (now ${onoff(runtime.sideEnabled)})` },
			];
			return subs.filter((s) => s.value.startsWith(sub));
		},
		handler: async (args, ctx) => {
			const parts = args.trim().split(/\s+/).filter(Boolean);
			const sub = (parts[0] ?? "help").toLowerCase();

			if (sub === "probe") {
				const caps = await client.probe(undefined, Date.now() + 10 ** 9); // force fresh
				ctx.ui.notify(
					`server ${caps.reachable ? "UP" : "down"} @ ${client.base}` +
						(caps.reachable ? ` — slots=${caps.slots} fork=${caps.fork} (${caps.slotCount} slot/s)` : ""),
					caps.reachable ? "info" : "warning",
				);
				return;
			}

			if (sub === "spill" || sub === "pin" || sub === "side") {
				const on = (parts[1] ?? "").toLowerCase() === "on";
				const off = (parts[1] ?? "").toLowerCase() === "off";
				if (!on && !off) {
					ctx.ui.notify(`usage: /keeper ${sub} on|off  (currently ${onoff(curToggle(sub))})`, "warning");
					return;
				}
				if (sub === "spill") runtime.spillEnabled = on;
				else if (sub === "pin") runtime.pinEnabled = on;
				else runtime.sideEnabled = on;
				ctx.ui.notify(`${sub} ${on ? "enabled" : "disabled"} — ${toggleBlurb(sub, on)}`, "info");
				return;
			}

			if (sub === "rollback") {
				// Main-loop undo via Pi's session tree (robust across providers). True
				// KV-slot rollback is used only inside side-sessions, where we own the slot.
				const msgs = ctx.sessionManager.getBranch().filter((e) => e.type === "message" && e.message.role === "user");
				if (msgs.length === 0) {
					ctx.ui.notify("No user messages to roll back to", "warning");
					return;
				}
				const n = parseInt(parts[1] ?? "", 10);
				const target = msgs[msgs.length - (isNaN(n) ? 1 : Math.min(Math.max(n, 1), msgs.length))];
				const res = await ctx.navigateTree(target.id, { summarize: true, label: "ctx-rollback" });
				ctx.ui.notify(res.cancelled ? "Rollback cancelled" : "Rolled back", "info");
				return;
			}

			// Bare `/keeper` (or `/keeper help`) lists the sub-commands; status needs `/keeper doctor`.
			if (sub === "help") {
				ctx.ui.notify(
					"/keeper sub-commands:\n" +
						"  doctor          status — server, slots, what's on/off, workdir\n" +
						"  probe           re-check the local llama.cpp server\n" +
						"  rollback [n]    rewind to the n-th most recent message you sent\n" +
						"  spill on|off    spill big tool outputs to disk\n" +
						"  pin on|off      reuse the server cache on rewind\n" +
						"  side on|off     off-context reading/reasoning (turn off on a single-slot server)",
					"info",
				);
				return;
			}
			if (sub !== "doctor") {
				ctx.ui.notify(`unknown subcommand "${sub}" — try /keeper help`, "warning");
				return;
			}
			const caps = client.capabilities;
			const cacheState = caps.promptCache === false ? "off" : caps.promptCache ? "on" : "unknown";
			const lines = [
				`server:   ${caps.reachable ? "UP" : "down"} @ ${client.base}`,
				`  slots:  ${caps.slots ? "enabled" : "DISABLED"}  fork=${caps.fork}  (${caps.slotCount} slot/s)`,
				`  cache:  prompt cache ${caps.reachable ? cacheState : "?"}`,
			];
			if (caps.reachable && !caps.slots) {
				lines.push("  → slot actions off; relaunch llama-server with --slot-save-path <dir> (0 disk cost)");
			}
			if (caps.reachable && caps.slotCount < 2 && caps.promptCache !== true) {
				lines.push("  → 1 slot + cache not on: side-sessions fall back to inline (safe); run --parallel 2 for a dedicated slot, or enable --cache-ram");
			}
			lines.push(
				`spill:    ${runtime.spillEnabled ? "on" : "off"} (>${CFG.spillThresholdChars} chars)  count=${runtime.spillCount}`,
				`pin:      ${runtime.pinEnabled ? "on" : "off"} (id_slot=${CFG.mainSlot}, cache_prompt) — warms KV on rewind`,
				`side:     ${runtime.sideEnabled ? "on" : "off"} ${sideStatus(client)}`,
				`workdir:  ${disk?.workdir ?? "(not initialized)"}`,
				"",
				"more:     /keeper probe · /keeper rollback [n] · /keeper spill|pin|side on|off",
			);
			ctx.ui.notify(lines.join("\n"), "info");
		},
	});
}

// ===========================================================================
// helpers
// ===========================================================================

/**
 * Pi's per-project session directory for a cwd:
 *   <agentDir>/sessions/--<cwd, leading slash stripped, [/\\:]->->--
 * Mirrors getDefaultSessionDir() in Pi's core/session-manager.js so our durable
 * state (AGENTS.md, event_log.jsonl, observations/) lands next to Pi's session
 * files instead of polluting the project tree. PI_AGENT_DIR overrides the root.
 */
function piSessionDir(cwd: string): string {
	const agentEnv = process.env.PI_AGENT_DIR;
	const agentDir = agentEnv
		? agentEnv.replace(/^~(?=$|[/\\])/, os.homedir())
		: path.join(os.homedir(), ".pi", "agent");
	const safePath = `--${cwd.replace(/^[/\\]/, "").replace(/[/\\:]/g, "-")}--`;
	return path.join(agentDir, "sessions", safePath);
}

/**
 * Run a side-session completion against the planned slot, handling teardown so the MAIN
 * conversation always survives:
 *  - stash "none" (dedicated slot): free the side slot's KV with a rollback afterward.
 *  - stash "cache": rely on the server prompt cache to auto-stash/reload the displaced main convo.
 *    On the next main turn the pinned-slot swap restores it from cache (recurrent/hybrid included),
 *    so there's no re-prefill — nothing for us to do here on teardown.
 * Never throws (the client calls are fail-safe); returns null when the completion failed.
 */
async function runSideSession(
	client: LlamaServerClient,
	plan: SidePlan,
	prompt: string,
	nPredict: number,
	signal?: AbortSignal,
): Promise<Completion | null> {
	let comp: Completion | null = null;
	try {
		comp = await client.complete(plan.slot, prompt, { nPredict, signal });
	} finally {
		if (plan.stash === "none" && comp?.generationId != null) {
			// free the dedicated side slot's KV cells
			await client.rollback(plan.slot, 0, comp.generationId, signal);
		}
		// stash "cache": leave it; the prompt cache reloads the main convo on the next main turn
	}
	return comp;
}

/** Distill a document down to what the goal needs, on the side slot. */
async function sideSessionDistill(
	client: LlamaServerClient,
	docText: string,
	goal: string,
	plan: SidePlan,
	signal?: AbortSignal,
): Promise<string | null> {
	const prompt =
		`Document:\n${clip(docText, 40000)}\n\n` + `Task: ${goal}\nExtract ONLY what is needed to satisfy the task. Be brief.\n`;
	const comp = await runSideSession(client, plan, prompt, 256, signal);
	if (!comp) return null;
	return comp.text.trim() || null;
}

/** Concatenate the text of recent branch messages, newest-last, capped. */
function recentBranchText(ctx: ExtensionContext, maxChars: number): string {
	const branch = ctx.sessionManager.getBranch();
	const parts: string[] = [];
	for (let i = branch.length - 1; i >= 0 && parts.join("\n").length < maxChars; i--) {
		const e = branch[i] as any;
		if (e?.type === "message" && Array.isArray(e.message?.content)) {
			const t = textOf(e.message.content);
			if (t) parts.unshift(`[${e.message.role}] ${t}`);
		}
	}
	return clip(parts.join("\n"), maxChars);
}

function textOf(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.map((c) => (c && typeof c === "object" && (c as any).type === "text" ? String((c as any).text ?? "") : ""))
		.join("");
}

function readFileSafe(p: string): string | null {
	try {
		return fs.readFileSync(p, "utf8");
	} catch {
		return null;
	}
}

function errorResult(msg: string) {
	return { content: [{ type: "text", text: `${TAG} ${msg}` }], isError: true, details: {} };
}

function clip(s: string, n: number): string {
	return s.length > n ? s.slice(0, n) + `\n…[clipped ${s.length - n} chars]` : s;
}
function intEnv(name: string, dflt: number): number {
	const v = parseInt(process.env[name] ?? "", 10);
	return Number.isFinite(v) ? v : dflt;
}
function boolEnv(name: string, dflt: boolean): boolean {
	const v = (process.env[name] ?? "").trim().toLowerCase();
	if (v === "") return dflt;
	return v === "1" || v === "true" || v === "on" || v === "yes";
}

/** "on"/"off" for a boolean — used in completions, the doctor, and toggle confirmations. */
function onoff(b: boolean): string {
	return b ? "on" : "off";
}
/** Current live value of a toggle sub-command. */
function curToggle(sub: string): boolean {
	return sub === "spill" ? runtime.spillEnabled : sub === "pin" ? runtime.pinEnabled : runtime.sideEnabled;
}
/** Plain-language consequence of a toggle, shown when the user flips it. */
function toggleBlurb(sub: string, on: boolean): string {
	if (sub === "spill") return on ? "big tool outputs go to disk + leave a recall pointer" : "tool outputs stay inline (context can grow faster)";
	if (sub === "pin") return on ? "rewinds reuse the warm KV prefix cache" : "slot pinning off (other providers are untouched regardless)";
	return on
		? "keeper_read / keeper_debug run off the main context when a server is available"
		: "keeper_read reads inline, keeper_debug notes inline — no KV-cache churn on single-slot servers";
}

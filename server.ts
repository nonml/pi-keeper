/**
 * server.ts — adaptive client for a local llama.cpp KV-slot server.
 *
 * Talks to a llama.cpp build that exposes per-slot KV checkpoint/rollback/fork
 * (the endpoints below). As a Pi extension client it is:
 *  - every call is abort-aware and fail-safe (never throws into Pi's loop)
 *  - capabilities are PROBED at runtime, cached with a short TTL, and re-probed
 *    when stale. Nothing here is hardcoded on/off — if the server is down or a
 *    given endpoint is missing, the relevant capability simply reports false and
 *    callers fall back to Pi's standard flow.
 *
 * Native llama.cpp endpoints live at the ROOT of the server, NOT under /v1:
 *    POST /completion ................ returns checkpoint_pos, generation_id,
 *                                      n_ctx_slot, fill_pct
 *    POST /slots/:id?action=rollback . body {pos, generation_id}; gen_id guards stale
 *    POST /slots/:id?action=checkpoint
 *    POST /slots/fork ................ body {src_slot, dst_slot, p0, p1}
 *                                      (NOTE: the Python harness used {src,dst,..};
 *                                       the real server expects {src_slot,dst_slot,..})
 */

/** What the live server supports right now. All independently gated. */
export interface ServerCapabilities {
	/** Server answered a health probe. */
	reachable: boolean;
	/** Slot actions (checkpoint/rollback/fork) enabled — needs `--slot-save-path`. */
	slots: boolean;
	/** Fork usable: slot actions enabled AND ≥2 slots (needs `--parallel ≥ 2`). */
	fork: boolean;
	/** Number of server slots (from GET /slots), i.e. the `--parallel` count. */
	slotCount: number;
	/**
	 * Is the prompt cache usable? Recurrent/hybrid models disable it server-side (their state
	 * can't be serialized by token count), which makes single-slot side-sessions unsafe.
	 * true = usable, false = disabled, null = the server doesn't report it (unknown).
	 */
	promptCache: boolean | null;
	/** When this snapshot was taken (epoch ms). */
	at: number;
}

export interface Completion {
	text: string;
	checkpointPos: number | null;
	generationId: number | null;
	nCtxSlot: number | null;
	fillPct: number;
	stopType: string | null;
	raw: Record<string, unknown>;
}

const NONE: ServerCapabilities = {
	reachable: false,
	slots: false,
	fork: false,
	slotCount: 0,
	promptCache: null,
	at: 0,
};

export class LlamaServerClient {
	/** Root URL (no /v1). e.g. http://127.0.0.1:8080 */
	readonly base: string;
	private readonly timeoutMs: number;
	private readonly probeTtlMs: number;
	private _caps: ServerCapabilities = NONE;
	private _probing: Promise<ServerCapabilities> | null = null;

	constructor(opts?: { base?: string; timeoutMs?: number; probeTtlMs?: number }) {
		this.base = normalizeRoot(
			opts?.base ??
				process.env.PI_KEEPER_SERVER ??
				// Defaults to the common llama.cpp provider baseUrl (minus /v1).
				"http://127.0.0.1:8080",
		);
		this.timeoutMs = opts?.timeoutMs ?? 600_000;
		this.probeTtlMs = opts?.probeTtlMs ?? 15_000;
	}

	/** Last known capabilities without forcing a network call. */
	get capabilities(): ServerCapabilities {
		return this._caps;
	}

	/**
	 * Return current capabilities, re-probing if the cached snapshot is stale.
	 * Cheap, deduped (concurrent callers share one probe), and never throws.
	 */
	async probe(signal?: AbortSignal, now = Date.now()): Promise<ServerCapabilities> {
		if (now - this._caps.at < this.probeTtlMs) return this._caps;
		if (this._probing) return this._probing;
		this._probing = this._doProbe(signal, now).finally(() => {
			this._probing = null;
		});
		return this._probing;
	}

	private async _doProbe(signal: AbortSignal | undefined, now: number): Promise<ServerCapabilities> {
		// 1) reachable? /health is the cheapest signal.
		const health = await this._raw("GET", "/health", undefined, signal, 2_000);
		if (!health.ok) {
			this._caps = NONE;
			return this._caps;
		}
		// 2) slot count from GET /slots (returns the per-slot array; 200 even without
		//    --slot-save-path). Tells us the --parallel count.
		let slotCount = 0;
		const slotsList = await this._raw("GET", "/slots", undefined, signal, 2_000);
		if (slotsList.ok && Array.isArray(slotsList.json)) slotCount = slotsList.json.length;
		// 2b) is the prompt cache usable? Recurrent/hybrid models disable it server-side, which
		//     makes single-slot multiplex unsafe. Not all builds report it; absent => null (unknown).
		//     Reads an optional /props flag so this lights up automatically once the server exposes it.
		let promptCache: boolean | null = null;
		const props = await this._raw("GET", "/props", undefined, signal, 2_000);
		if (props.ok && props.json && typeof props.json === "object") {
			const pj = props.json as Record<string, unknown>;
			if (typeof pj.prompt_cache === "boolean") promptCache = pj.prompt_cache;
			else if (typeof pj.recurrent_or_hybrid === "boolean") promptCache = !pj.recurrent_or_hybrid;
		}
		// 3) are slot ACTIONS enabled? They are gated behind `--slot-save-path` and
		//    return HTTP 501 (not_supported) when off. Probe NON-destructively with an
		//    invalid rollback: the save-path guard fires first (501 = disabled), else
		//    the handler rejects the bad pos (400 = enabled). Never mutates state.
		let slots = false;
		const act = await this._raw("POST", "/slots/0?action=rollback", { pos: -1, generation_id: -1 }, signal, 2_000);
		slots = act.status === 400; // 400 => reached handler => actions enabled; 501 => disabled
		// 4) fork needs slot actions AND a destination slot (≥2 slots / --parallel ≥ 2).
		const fork = slots && slotCount >= 2;

		this._caps = { reachable: true, slots, fork, slotCount, promptCache, at: now };
		return this._caps;
	}

	/** /completion — generation + KV bookkeeping. null on any failure (caller falls back). */
	async complete(
		slotId: number,
		prompt: string,
		opts?: { nPredict?: number; stop?: string[]; signal?: AbortSignal; sampling?: Record<string, unknown> },
	): Promise<Completion | null> {
		const body = {
			prompt,
			id_slot: slotId,
			n_predict: opts?.nPredict ?? 512,
			cache_prompt: true, // reuse the cached prefix — essential
			stop: opts?.stop ?? [],
			...(opts?.sampling ?? {}),
		};
		const r = await this._raw("POST", "/completion", body, opts?.signal);
		if (!r.ok || !r.json) return null;
		const j = r.json as Record<string, unknown>;
		return {
			text: str(j.content),
			checkpointPos: numOrNull(j.checkpoint_pos),
			generationId: numOrNull(j.generation_id),
			nCtxSlot: numOrNull(j.n_ctx_slot),
			fillPct: typeof j.fill_pct === "number" ? j.fill_pct : 0,
			stopType: typeof j.stop_type === "string" ? j.stop_type : null,
			raw: j,
		};
	}

	/** Truncate-to-pos. generation_id (-1 = no check) guards stale rollbacks. */
	async rollback(slotId: number, pos: number, generationId: number, signal?: AbortSignal): Promise<boolean> {
		const r = await this._raw("POST", `/slots/${slotId}?action=rollback`, { pos, generation_id: generationId }, signal);
		return r.ok;
	}

	async checkpoint(slotId: number, signal?: AbortSignal): Promise<boolean> {
		const r = await this._raw("POST", `/slots/${slotId}?action=checkpoint`, {}, signal);
		return r.ok;
	}

	/** Metadata-only slot fork. Real server field names: src_slot/dst_slot. */
	async fork(srcSlot: number, dstSlot: number, p0: number, p1: number, signal?: AbortSignal): Promise<boolean> {
		const r = await this._raw("POST", "/slots/fork", { src_slot: srcSlot, dst_slot: dstSlot, p0, p1 }, signal);
		return r.ok;
	}

	/**
	 * Live KV fill fraction (0..1) for a slot, derived from GET /slots
	 * (n_prompt_tokens / n_ctx). This is the server's *real* occupancy — sharper than a
	 * token estimate and correct for quantized/MTP caches. null when unavailable.
	 */
	async slotFill(slotId: number, signal?: AbortSignal): Promise<number | null> {
		const r = await this._raw("GET", "/slots", undefined, signal, 2_000);
		if (!r.ok || !Array.isArray(r.json)) return null;
		const s = (r.json as Array<Record<string, unknown>>).find((x) => x && x.id === slotId);
		if (!s) return null;
		const used = typeof s.n_prompt_tokens === "number" ? s.n_prompt_tokens : null;
		const ctxN = typeof s.n_ctx === "number" ? s.n_ctx : null;
		if (used == null || ctxN == null || ctxN <= 0) return null;
		return used / ctxN;
	}

	/** Low-level request. Never throws; returns a structured result. */
	private async _raw(
		method: "GET" | "POST",
		path: string,
		body: unknown,
		signal?: AbortSignal,
		timeoutMs = this.timeoutMs,
	): Promise<{ ok: boolean; status: number; json: unknown }> {
		const ctrl = new AbortController();
		const timer = setTimeout(() => ctrl.abort(), timeoutMs);
		const onAbort = () => ctrl.abort();
		signal?.addEventListener("abort", onAbort, { once: true });
		try {
			const res = await fetch(this.base + path, {
				method,
				headers: body === undefined ? undefined : { "Content-Type": "application/json" },
				body: body === undefined ? undefined : JSON.stringify(body),
				signal: ctrl.signal,
			});
			let json: unknown = null;
			try {
				json = await res.json();
			} catch {
				/* non-JSON body is fine for health/probe */
			}
			return { ok: res.ok, status: res.status, json };
		} catch {
			return { ok: false, status: 0, json: null };
		} finally {
			clearTimeout(timer);
			signal?.removeEventListener("abort", onAbort);
		}
	}
}

function normalizeRoot(url: string): string {
	return url.replace(/\/+$/, "").replace(/\/v1$/, "");
}
function numOrNull(v: unknown): number | null {
	return typeof v === "number" ? v : null;
}
function str(v: unknown): string {
	return typeof v === "string" ? v : "";
}

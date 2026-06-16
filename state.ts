/**
 * state.ts — durable, on-disk working memory.
 *
 * This tier needs NO server. It is the safety net that makes dropping/masking old
 * turns safe:
 *   AGENTS.md       distilled, re-injected-every-turn facts (load-bearing)
 *   event_log.jsonl append-only transcript of spilled/masked content (grep-able backstop)
 *   observations/   big tool outputs spilled to disk, referenced by a short pointer
 */
import * as fs from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";

export interface SpillRecord {
	ref: string;
	bytes: number;
	tool: string;
	at: number;
}

export class DiskState {
	readonly workdir: string;
	readonly agentsMd: string;
	readonly eventLog: string;
	readonly obsDir: string;

	constructor(workdir: string) {
		this.workdir = workdir;
		this.obsDir = path.join(workdir, "observations");
		this.agentsMd = path.join(workdir, "AGENTS.md");
		this.eventLog = path.join(workdir, "event_log.jsonl");
		fs.mkdirSync(this.obsDir, { recursive: true });
		if (!fs.existsSync(this.agentsMd)) {
			fs.writeFileSync(this.agentsMd, "# Project Working Memory\n\n## Durable facts\n");
		}
	}

	readAgentsMd(): string {
		try {
			return fs.readFileSync(this.agentsMd, "utf8");
		} catch {
			return "";
		}
	}

	appendFact(fact: string): void {
		if (!fact.trim()) return;
		try {
			fs.appendFileSync(this.agentsMd, "\n" + fact.trim() + "\n");
		} catch {
			/* non-fatal */
		}
	}

	appendEvent(record: Record<string, unknown>): void {
		try {
			fs.appendFileSync(this.eventLog, JSON.stringify({ t: Date.now(), ...record }) + "\n");
		} catch {
			/* non-fatal */
		}
	}

	/** Spill a large blob, return a stable pointer ref (content-addressed). */
	spill(text: string, tool: string): SpillRecord {
		const ref = "obs_" + crypto.createHash("sha256").update(text).digest("hex").slice(0, 12);
		const file = path.join(this.obsDir, ref + ".txt");
		try {
			if (!fs.existsSync(file)) fs.writeFileSync(file, text);
		} catch {
			/* non-fatal */
		}
		const record: SpillRecord = { ref, bytes: Buffer.byteLength(text), tool, at: Date.now() };
		this.appendEvent({ kind: "spill", ...record });
		return record;
	}

	/** Re-fetch a slice of a spilled observation (used by the read-back tool). */
	readSlice(ref: string, start = 0, n = 4000): string | null {
		const file = path.join(this.obsDir, path.basename(ref) + ".txt");
		try {
			const full = fs.readFileSync(file, "utf8");
			return full.slice(start, start + n);
		} catch {
			return null;
		}
	}
}

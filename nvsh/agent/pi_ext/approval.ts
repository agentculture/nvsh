// nvsh pi approval extension (task t11).
//
// Pure forwarder: every decision comes from `nvsh approve check/add/audit`
// (Python, nvsh.approvals -- task t5). This file never decides policy
// itself -- it only asks the CLI and relays the result. Dependency-free:
// only Node builtins and pi's own ExtensionAPI type are imported.
//
// Wire shape: pi's own `ctx.ui.select()` only emits
// {type, id, method, title, options, timeout} over the rpc extension_ui
// sub-protocol -- there is no room for a custom "command" field. So the
// payload rides inside `title` as one machine-readable JSON envelope,
// {"nvsh":"approval","v":1,"tool":"bash","command":<raw>,"reason":<text>},
// and PiAgent._map_event parses it back out. The envelope never contains
// human prompt text: nvsh renders the panel itself, and a title built as
// prose would come back as Proposal.command and be run verbatim
// (deviation d8). See docs/pi-rpc.md.

import { spawnSync } from "node:child_process";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Where nvsh is. `nvsh setup` exports NVSH_BIN into the operator's rc file
// and PiAgent passes it -- with XDG_CONFIG_HOME/XDG_RUNTIME_DIR -- into pi's
// env, so this extension reads exactly the store the panel writes. The PATH
// lookup is only the fallback for a pi started by hand. Deviation d21: a
// daemon-spawned pi whose PATH has no nvsh failed every spawnSync with
// ENOENT, the empty stdout parsed as "ask", and an already-approved pattern
// was asked about again on every tool call with nothing said anywhere.
const NVSH_BIN = process.env.NVSH_BIN || "nvsh";

interface Run {
  ok: boolean;
  stdout: string;
  // Set only when nvsh never ran at all (ENOENT, EACCES, a signal), never
  // for a command nvsh ran and rejected -- that is plain `ok: false`.
  failure: string | null;
}

function nvsh(args: string[]): Run {
  const result = spawnSync(NVSH_BIN, args, { encoding: "utf8" });
  if (result.error || result.status === null) {
    const why = result.error ? String((result.error as Error).message) : "no exit status";
    return { ok: false, stdout: "", failure: `could not run ${NVSH_BIN}: ${why}` };
  }
  return { ok: result.status === 0, stdout: result.stdout || "", failure: null };
}

function checkCommand(command: string): { decision: string; failure: string | null } {
  const run = nvsh(["approve", "check", command, "--json"]);
  if (run.failure) {
    return { decision: "ask", failure: run.failure };
  }
  try {
    return { decision: String(JSON.parse(run.stdout).decision || "ask"), failure: null };
  } catch {
    return { decision: "ask", failure: `could not run ${NVSH_BIN}: unreadable approve check output` };
  }
}

function blockedBy(run: Run, refusal: string): { block: true; reason: string } {
  return { block: true, reason: run.failure || refusal };
}

function audit(command: string, decision: string): void {
  nvsh(["approve", "audit", "--tool", "bash", "--command", command, "--decision", decision, "--json"]);
}

export default function (pi: ExtensionAPI) {
  pi.on("tool_call", async (event: any, ctx: any) => {
    if (event.toolName !== "bash") {
      return { block: true, reason: "nvsh v1 allows only the bash tool" };
    }

    const command = String((event.input && event.input.command) || "");
    // Checked afresh on *every* tool call, never memoized: the pattern the
    // operator widens halfway through a turn has to apply to the rest of it.
    const decision = checkCommand(command);

    // nvsh itself is unreachable. Say so and block: treating an
    // infrastructure failure as "ask" buries it under a dialog the
    // operator's answer cannot fix, forever (d21).
    if (decision.failure) {
      return { block: true, reason: `nvsh could not check this tool call -- ${decision.failure}` };
    }

    // Already approved (persisted or held for this session) -- allow.
    if (decision.decision === "user" || decision.decision === "session") {
      audit(command, decision.decision);
      return;
    }

    // The model's stated reason, when the tool schema carries one. pi
    // 0.84.2's bash tool takes only {command, timeout}, so this is usually
    // "" -- it is read defensively so a future schema needs no change here.
    const reason = String(
      (event.input && (event.input.reason || event.input.description)) || "",
    );
    const payload = JSON.stringify({
      nvsh: "approval",
      v: 1,
      tool: "bash",
      command,
      reason,
    });
    const choice = await ctx.ui.select(payload, [
      "once",
      "session",
      "user",
      "deny",
    ]);

    if (choice === undefined || choice === "deny") {
      audit(command, "deny");
      return { block: true, reason: "operator denied this command" };
    }

    if (choice === "once") {
      audit(command, "once");
      return;
    }

    // choice === "session": the exact line, unwidened, for this login
    // session. `--session` writes to
    // $XDG_RUNTIME_DIR/nvsh/session-approvals.toml (deviation d15), so the
    // approval outlives this spawnSync and the next `approve check` -- from
    // any process of this login -- matches it. Before d15 it was held in
    // the CLI process's memory and was gone before this call returned.
    if (choice === "session") {
      const added = nvsh(["approve", "add", command, "--session", "--json"]);
      if (!added.ok) {
        audit(command, "block");
        return blockedBy(added, "nvsh refused to approve this pattern for the session");
      }
      audit(command, "session");
      return;
    }

    // choice === "user": widen to "<first word> *" per the spec, so
    // Approvals.add's own refusal rules (sudo/rm/bare '*') still apply --
    // e.g. "sudo rm -rf /x" always asks even after this branch is tried.
    const firstWord = command.trim().split(/\s+/, 1)[0] || command;
    const pattern = `${firstWord} *`;
    const added = nvsh(["approve", "add", pattern, "--json"]);
    if (!added.ok) {
      audit(command, "block");
      return blockedBy(added, "nvsh refused to approve this pattern for the user");
    }
    audit(command, "user");
    return;
  });
}

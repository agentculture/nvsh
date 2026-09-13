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
// command line rides inside `title`, on its own line prefixed
// "command: ", and PiAgent._map_event's fallback chain
// (command -> message -> title) picks it up from `title` since `message`
// is absent for `select`. See docs/pi-rpc.md.

import { spawnSync } from "node:child_process";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const NVSH_BIN = process.env.NVSH_BIN || "nvsh";

function nvsh(args: string[]): { ok: boolean; stdout: string } {
  const result = spawnSync(NVSH_BIN, args, { encoding: "utf8" });
  return { ok: result.status === 0, stdout: result.stdout || "" };
}

function checkCommand(command: string): { decision: string; pattern: string | null } {
  const { stdout } = nvsh(["approve", "check", command, "--json"]);
  try {
    return JSON.parse(stdout);
  } catch {
    return { decision: "ask", pattern: null };
  }
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
    const decision = checkCommand(command);

    // Already approved (persisted or held for this session) -- allow.
    if (decision.decision === "user" || decision.decision === "session") {
      audit(command, decision.decision);
      return;
    }

    const choice = await ctx.ui.select(`nvsh: run this command?\ncommand: ${command}`, [
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

    if (choice === "session") {
      const added = nvsh(["approve", "add", command, "--session", "--json"]);
      if (!added.ok) {
        audit(command, "block");
        return { block: true, reason: "nvsh refused to approve this pattern for the session" };
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
      return { block: true, reason: "nvsh refused to approve this pattern for the user" };
    }
    audit(command, "user");
    return;
  });
}

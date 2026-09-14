# nvsh demo recordings

Terminal recordings taken during the three-machine verification pass
(see [`../verification.md`](../verification.md)). Both are plain
[asciicast v2](https://docs.asciinema.org/manual/asciicast/v2/) files — one
JSON header line, then one `[time, "o", "data"]` line per output chunk.

| File | Machine | What it shows |
|------|---------|----------------|
| `spark-cuda-oom.cast` | DGX Spark, Ghostty, pi/associate backend | A CUDA-out-of-memory-style `RuntimeError` triggers the failure panel; the agent diagnoses it against the unified-memory platform block and proposes a command, which the operator declines with `Esc`. Re-recorded after the d2-d8 fixes, so the proposal is the bare command; it also shows the daemon timeout still open on the Spark (deviation w1 in [`../verification.md`](../verification.md)). |
| `orin-missing-package.cast` | Jetson AGX Orin over ssh, openai-compat backend | A missing TensorRT tool (`trtexec`, exit 127) triggers the panel through the offline-first openai-compat fallback. |

Both recordings were scrubbed at record time: the model endpoint and its
bearer token are replaced with `[the LAN gateway]` / `[redacted]`, so no
endpoint or credential is committed here.

## Playing them

With asciinema, if it is installed:

```bash
asciinema play docs/demos/spark-cuda-oom.cast
```

Without it — on an air-gapped Jetson, for instance — the recorder that made
these files replays them too, using only the standard library:

```bash
python3 scripts/record-cast.py play docs/demos/spark-cuda-oom.cast
python3 scripts/record-cast.py play docs/demos/orin-missing-package.cast --speed 4
```

`--speed N` divides every delay by `N`; `--idle-limit S` caps any single gap
at `S` seconds, which matters here because an agent turn can take a minute
or more of real time.

## Recording another one

```bash
python3 scripts/record-cast.py record out.cast -- bash -i          # interactive
python3 scripts/record-cast.py record out.cast \
    --feed 'ls /nope\r|120' --feed '^[|5' -- bash -i               # scripted
```

`--feed 'keys|seconds'` drives the session with no human at the keyboard
(`\r` Enter, `\t` Tab, `^G` Ctrl+G, `^C` Ctrl+C, `^[` Esc) and then waits
the given number of seconds. `--replace-from FILE` takes `OLD=NEW` lines and
rewrites every captured chunk before it is written, which is how a session
recorded against a private endpoint is scrubbed without the secret ever
reaching the `.cast` file or the command line.

## Re-recording

The repo also carries a second set of recordings, one per device
(`demo-spark.cast`, `demo-thor.cast`, `demo-orin.cast`, rendered to
`demo-spark.svg`, `demo-thor.svg`, `demo-orin.svg`) built by
`scripts/demo-record.py` rather than by hand with `record-cast.py`. That
script drives the real hook -> client -> daemon -> adapter -> panel path
inside a throwaway `$HOME`, but with `[aliases].default` pinned to the
`demo` adapter (`nvsh/agent/demo.py`, backed by
`nvsh/agent/demo_fixture.json`) instead of a real backend, so the reply is a
**scripted fixture**, not a live model call: no harness, no key, no
network, the same wording every time. It runs a fixed three-step feed --
fail, approve, retry -- scrubs the hostname, user name and any LAN IPv4
with `record-cast.py --replace-from`, and stops the sandbox daemon before
exiting. Everything else about the machine in the recording -- kernel,
`nvsh --version`, and whatever `nvsh/platform/` detects -- is that device's
real platform, because the driver runs there over ssh; only the agent
backend is faked.

**Re-recording is a manual maintainer step.** No CI job regenerates these
casts or SVGs, or checks whether they are still fresh against the current
code; a maintainer decides when the demo has drifted enough to redo it, by
hand, on each real device.

nvsh has no third-party runtime dependencies, so a device does not need it
installed to run the driver: `scripts/demo-record.py` inserts the repo root
onto `sys.path` itself (see its `REPO_ROOT` bootstrap) and imports
`nvsh`/`nvsh.shell.render` straight from an unpacked checkout, so a plain
`git archive` of the branch, copied over and unpacked, is enough. The one
thing to get right is which `nvsh` ends up on the recorded `$PATH`:
`resolve_nvsh_bin()` prefers a real `nvsh` already on `PATH` on the device,
but only if `nvsh --version` matches this checkout's `nvsh.__version__`; a
mismatched or missing one falls back to a small wrapper script the driver
plants itself, which runs `python3 -m nvsh` with the unpacked tree on
`PYTHONPATH`. So an older `uv tool install`-ed nvsh on a device's `PATH` is
silently bypassed rather than used by mistake.

To redo one device, from the dev box:

```bash
git archive agent/demo/t7 | ssh spark 'mkdir -p /tmp/nvsh-src && tar -x -C /tmp/nvsh-src'
ssh spark 'PYTHONPATH=/tmp/nvsh-src python3 /tmp/nvsh-src/scripts/demo-record.py /tmp/demo-spark.cast'
scp spark:/tmp/demo-spark.cast docs/demos/demo-spark.cast
scripts/demo-render.sh docs/demos/demo-spark.cast docs/demos/demo-spark.svg
```

Repeat the same three steps (record over ssh, `scp` the `.cast` back,
render) for `thor` and `orin`, swapping the device name and output paths;
`demo-record.py --keep-sandbox` leaves the throwaway `$HOME` in place on
the device for troubleshooting instead of deleting it, and
`--wait-panel`/`--wait-approve`/`--wait-retry` widen the fixed waits on a
device where the daemon's first spawn or the panel's stream is slow. See
"Playing them" above for how to preview a re-recorded `.cast` before
rendering it, and `scripts/demo-render.sh`'s header comment for the `agg`
fallback when `npx` is unavailable.

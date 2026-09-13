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

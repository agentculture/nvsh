"""Force greedy decoding on a served model directory (issue 46, deviation d3).

Qwen3.5-0.8B ships no ``generation_config.json``, and nvsh's Tier 2 request
sets no ``temperature`` (docs/tier2.md), so a served snapshot missing the
file samples at vLLM's default temperature of 1.0 instead of the greedy
decoding the tool-decision model was trained and measured under. Every model
directory this pipeline serves -- the stock snapshot, a merged fine-tune, a
quantized AWQ export, and a healed checkpoint -- must carry a
``generation_config.json`` with ``temperature`` pinned to 0 and ``do_sample``
set to false.

    python scripts/lfm-finetune/gen_config.py write <model-dir>
    python scripts/lfm-finetune/gen_config.py check <model-dir>
    python scripts/lfm-finetune/gen_config.py stock-copy <snapshot-dir> <out-dir> [--force]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

GEN_CONFIG_FILE = "generation_config.json"
CONFIG_FILE = "config.json"

#: Token ids worth carrying over from config.json when generation_config.json
#: does not exist yet -- copied only when present, never invented.
_TOKEN_ID_KEYS = ("eos_token_id", "bos_token_id", "pad_token_id")


def _token_ids_from_config(model_dir: Path) -> dict:
    """Token ids for a fresh generation_config.json, read from config.json.

    Looks at the top level first, then ``text_config`` (Qwen3.5's own
    config.json nests ``eos_token_id`` there and has no top-level id at all).
    A key present in both places keeps its top-level value. Missing entirely
    (no config.json, or neither location has the key) means the key is left
    out rather than guessed.
    """
    config_path = model_dir / CONFIG_FILE
    if not config_path.is_file():
        return {}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config") if isinstance(config, dict) else None
    ids = {}
    for key in _TOKEN_ID_KEYS:
        if isinstance(config, dict) and config.get(key) is not None:
            ids[key] = config[key]
        elif isinstance(text_config, dict) and text_config.get(key) is not None:
            ids[key] = text_config[key]
    return ids


def _tokenizer_eos_id(model_dir: Path) -> int | None:
    """The id of the tokenizer's own eos token (its chat end-of-turn), if it can be read.

    Read from tokenizer_config.json's ``eos_token`` and tokenizer.json's
    ``added_tokens`` -- no transformers needed. None when either is missing.
    """
    tok_config = model_dir / "tokenizer_config.json"
    tok_json = model_dir / "tokenizer.json"
    if not (tok_config.is_file() and tok_json.is_file()):
        return None
    eos = json.loads(tok_config.read_text(encoding="utf-8")).get("eos_token")
    if isinstance(eos, dict):
        eos = eos.get("content")
    for token in json.loads(tok_json.read_text(encoding="utf-8")).get("added_tokens", []):
        if token.get("content") == eos:
            return token.get("id")
    return None


def _with_tokenizer_eos(ids: dict, model_dir: Path) -> dict:
    """*ids* with the tokenizer's end-of-turn added to eos_token_id, first.

    Qwen3.5's config.json names <|endoftext|> while its chat turn ends with
    <|im_end|>; Qwen's own instruct generation configs list both, so a
    consumer that trusts this file (HF generate, GGUF conversion) stops at
    the end of the turn.
    """
    turn_end = _tokenizer_eos_id(model_dir)
    if turn_end is None:
        return ids
    current = ids.get("eos_token_id")
    listed = current if isinstance(current, list) else ([] if current is None else [current])
    if turn_end in listed:
        return ids
    merged = [turn_end, *listed]
    return {**ids, "eos_token_id": merged[0] if len(merged) == 1 else merged}


def write(model_dir: Path, temperature: float = 0.0) -> dict:
    """Write or update *model_dir*'s generation_config.json for greedy decoding.

    An existing file keeps every other key (e.g. ``eos_token_id``) and only
    has ``temperature``/``do_sample`` forced. A missing file is created with
    those two keys plus whatever token ids config.json names. Returns the
    payload written.
    """
    gen_path = model_dir / GEN_CONFIG_FILE
    if gen_path.is_file():
        payload = json.loads(gen_path.read_text(encoding="utf-8"))
    else:
        payload = _token_ids_from_config(model_dir)
    payload = _with_tokenizer_eos(payload, model_dir)
    payload["temperature"] = temperature
    payload["do_sample"] = False
    gen_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def check(model_dir: Path) -> str | None:
    """``None`` when *model_dir* has a valid greedy-decoding generation_config.json,
    else a one-line reason."""
    gen_path = model_dir / GEN_CONFIG_FILE
    if not gen_path.is_file():
        return f"no {GEN_CONFIG_FILE} in {model_dir}"
    payload = json.loads(gen_path.read_text(encoding="utf-8"))
    if payload.get("temperature") != 0 and payload.get("temperature") != 0.0:
        return f"temperature is {payload.get('temperature')!r}, not 0"
    if payload.get("do_sample") is not False:
        return f"do_sample is {payload.get('do_sample')!r}, not false"
    return None


def stock_copy(snapshot_dir: Path, out_dir: Path, force: bool = False) -> None:
    """Copy *snapshot_dir* to *out_dir* with symlinks resolved, then write().

    A Hugging Face snapshot directory is a tree of symlinks into a shared
    blob store; resolving them makes *out_dir* a self-contained copy that can
    be bind-mounted into a container. Refuses a non-empty *out_dir* unless
    *force* is set, in which case it is replaced.
    """
    if out_dir.exists():
        if any(out_dir.iterdir()):
            if not force:
                raise FileExistsError(f"{out_dir} exists and is not empty (use --force)")
            shutil.rmtree(out_dir)
        else:
            out_dir.rmdir()
    shutil.copytree(snapshot_dir, out_dir, symlinks=False)
    write(out_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    write_p = sub.add_parser(
        "write", help="write/update generation_config.json for greedy decoding"
    )
    write_p.add_argument("model_dir", type=Path)

    check_p = sub.add_parser(
        "check", help="check generation_config.json is set for greedy decoding"
    )
    check_p.add_argument("model_dir", type=Path)

    copy_p = sub.add_parser("stock-copy", help="copy a snapshot with symlinks resolved, then write")
    copy_p.add_argument("snapshot_dir", type=Path)
    copy_p.add_argument("out_dir", type=Path)
    copy_p.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "write":
        write(args.model_dir)
        return 0

    if args.command == "check":
        reason = check(args.model_dir)
        if reason is None:
            print("ok")
            return 0
        print(reason, file=sys.stderr)
        return 1

    if args.command == "stock-copy":
        try:
            stock_copy(args.snapshot_dir, args.out_dir, force=args.force)
        except FileExistsError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0

    return 1  # pragma: no cover - argparse's `required=True` makes this unreachable


if __name__ == "__main__":
    sys.exit(main())

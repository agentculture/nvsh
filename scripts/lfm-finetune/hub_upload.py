"""Upload one scanned bundle to a PRIVATE Hub repository and fetch it back (issue 46, t27).

This is the only place in the fine-tune pipeline that talks to the Hugging
Face Hub; ``pipeline.sh upload-bundle`` (a release bundle) and ``pipeline.sh
upload`` (a run's merged checkpoint) are its only callers. It refuses,
before any Hub call:

- a repository id that is not ``jetson-ai-lab/qwen3.5-0.8b-nvsh-<suffix>``
  (lower-case letters, digits and ``-``);
- a run without ``FINAL=1`` in the environment;
- a bundle ``scan_bundle.py verify`` does not pass (no scan, findings, or a
  file changed after the scan);
- an unset token variable. The token is read only from the variable
  ``--token-env`` names (injected by the operator with ``grant run
  --inject``) and never printed.

Then it creates the repository private (``exist_ok``), sets it private again
(an existing repository might not be), uploads the folder, downloads that
exact commit into a temporary folder next to the bundle and compares the
sha256 of every file both ways: any changed, missing or extra file fails
loudly (a ``.gitattributes`` the Hub adds on its own is the one expected
extra). Last it reads the repository's ``private`` flag back, prints it, and
fails unless it is true. Nothing here ever makes a repository public: public
visibility waits for the operator's per-repository approval, outside this
script.

    FINAL=1 python scripts/lfm-finetune/hub_upload.py --bundle WORK/bundles/tool-jev \
        --repo jetson-ai-lab/qwen3.5-0.8b-nvsh-tool-jev --repo-type model --token-env HF_TOKEN

``huggingface_hub`` comes from the training environment (the repository's own
environment has no third-party packages); ``pipeline.sh`` puts it on
``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

_HERE = Path(__file__).resolve().parent

#: Every repository this script may write to starts with this.
ALLOWED_PREFIX = "jetson-ai-lab/qwen3.5-0.8b-nvsh-"
_REPO_RE = re.compile(r"^" + re.escape(ALLOWED_PREFIX) + r"[a-z0-9]+(?:-[a-z0-9]+)*$")

REPO_TYPES = ("model", "dataset")

#: Remote-only files the Hub itself writes into a new repository.
EXPECTED_REMOTE_EXTRAS = frozenset({".gitattributes"})

#: What ``snapshot_download(local_dir=...)`` adds next to the files.
_DOWNLOAD_CACHE = ".cache"


class UploadError(ValueError):
    """A refusal or a failed check; the message never holds the token."""


def check_repo(repo: str) -> None:
    if not _REPO_RE.match(repo):
        raise UploadError(
            f"refusing {repo!r}: uploads go only to {ALLOWED_PREFIX}<suffix>"
            " (lower-case letters, digits and '-')"
        )


def _scan_bundle():
    spec = importlib.util.spec_from_file_location(
        "hub_upload_scan_bundle", _HERE / "scan_bundle.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_no_symlinks(bundle: Path) -> None:
    """Refuse any symlink in *bundle*: scanning and uploading follow links, so one
    pointing outside the folder (the sealed held-out, a key file) would ship."""
    links = sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_symlink())
    if links:
        raise UploadError(f"refusing {bundle}: it holds symlink(s) {', '.join(links)}")


def check_inventory(local: Path, remote_files: list[str]) -> list[str]:
    """Every file the repository holds that the bundle does not, or the reverse.

    The fetched copy's ``.cache/`` is download metadata and is skipped when
    hashing, so the repository's own file list is what proves nothing extra
    (a stale ``.cache/held-out.json``, say) is there."""
    mine = set(digests(local))
    theirs = set(remote_files)
    extra = theirs - mine - EXPECTED_REMOTE_EXTRAS  # the Hub may add these on its own
    problems = [f"{rel}: in the repository but not in the bundle" for rel in sorted(extra)]
    problems += [f"{rel}: missing from the repository" for rel in sorted(mine - theirs)]
    return problems


def digests(folder: Path, *, skip_cache: bool = False) -> dict[str, str]:
    """``relative POSIX path -> sha256`` for every regular file under *folder*."""
    found: dict[str, str] = {}
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        rel = path.relative_to(folder).as_posix()
        if skip_cache and rel.split("/", 1)[0] == _DOWNLOAD_CACHE:
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        found[rel] = digest.hexdigest()
    return found


def compare(local: Path, fetched: Path) -> list[str]:
    """Every difference between the local bundle and the fetched copy, one line each."""
    mine = digests(local)
    theirs = digests(fetched, skip_cache=True)
    problems = []
    for rel in sorted(set(mine) | set(theirs)):
        if rel not in theirs:
            problems.append(f"{rel}: missing from the fetched copy")
        elif rel not in mine:
            if rel not in EXPECTED_REMOTE_EXTRAS:
                problems.append(f"{rel}: in the repository but not in the bundle")
        elif mine[rel] != theirs[rel]:
            problems.append(f"{rel}: sha256 differs")
    return problems


def _set_private(api: Any, repo: str, repo_type: str) -> None:
    """Make *repo* private: ``update_repo_settings`` (huggingface_hub 1.x), or the
    older ``update_repo_visibility``. Never anything but private=True."""
    if hasattr(api, "update_repo_settings"):
        api.update_repo_settings(repo, private=True, repo_type=repo_type)
    else:
        api.update_repo_visibility(repo, private=True, repo_type=repo_type)


def upload(
    *,
    bundle: Path,
    repo: str,
    repo_type: str,
    token_env: str,
    hub: Any = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Upload *bundle* to *repo* privately and check it; return a summary."""
    environ = os.environ if environ is None else environ
    check_repo(repo)
    if repo_type not in REPO_TYPES:
        raise UploadError(f"unknown repo type {repo_type!r} (one of {', '.join(REPO_TYPES)})")
    if environ.get("FINAL") != "1":
        raise UploadError("refusing to upload without FINAL=1 set")
    if not bundle.is_dir():
        raise UploadError(f"{bundle} is not a folder")
    check_no_symlinks(bundle)
    reason = _scan_bundle().verify(bundle)
    if reason is not None:
        raise UploadError(f"scan_bundle.py verify failed for {bundle}: {reason}")
    hub_token = environ.get(token_env)
    if not hub_token:
        raise UploadError(
            f"{token_env} is not set (e.g. grant run --inject {token_env}=<secret name> -- ...)"
        )
    if hub is None:
        hub = importlib.import_module("huggingface_hub")

    api = hub.HfApi(token=hub_token)
    api.create_repo(repo, repo_type=repo_type, private=True, exist_ok=True)
    _set_private(api, repo, repo_type)
    commit = api.upload_folder(
        folder_path=str(bundle),
        repo_id=repo,
        repo_type=repo_type,
        commit_message=f"nvsh issue 46: upload {bundle.name}",
    )
    revision = getattr(commit, "oid", None)
    print(f"uploaded {bundle} to {repo} ({repo_type}, commit {revision})")

    with tempfile.TemporaryDirectory(prefix=".fetch-", dir=bundle.parent) as fetched:
        hub.snapshot_download(
            repo, repo_type=repo_type, revision=revision, token=hub_token, local_dir=fetched
        )
        problems = compare(bundle, Path(fetched))
        problems += check_inventory(
            bundle, api.list_repo_files(repo, repo_type=repo_type, revision=revision)
        )
        files = len(digests(bundle))
    if problems:
        raise UploadError(
            f"the copy fetched back from {repo} differs from {bundle}:\n  " + "\n  ".join(problems)
        )
    print(f"fetched {repo} back: {files} files byte-identical to {bundle}")

    private = getattr(api.repo_info(repo, repo_type=repo_type), "private", None)
    print(f"{repo}: private={private}")
    if private is not True:
        raise UploadError(f"{repo} is not private (the Hub reports private={private})")
    return {
        "repo": repo,
        "repo_type": repo_type,
        "revision": revision,
        "files": files,
        "private": private,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--repo-type", required=True, choices=REPO_TYPES)
    parser.add_argument(
        "--token-env", required=True, help="the NAME of the variable holding the Hub token"
    )
    args = parser.parse_args(argv)
    try:
        upload(
            bundle=args.bundle,
            repo=args.repo,
            repo_type=args.repo_type,
            token_env=args.token_env,
        )
    except Exception as exc:  # noqa: BLE001 -- every failure is reported, never the token
        message = str(exc)
        hub_token = os.environ.get(args.token_env)
        if hub_token:
            message = message.replace(hub_token, "<token>")
        print(f"hub_upload: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

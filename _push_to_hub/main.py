#!/usr/bin/env python3
"""Push trained models and tokenizer constants to Hugging Face Hub.

This script scans specified output folders for model checkpoints (Lightning
checkpoints and common PyTorch checkpoint files), collects nearby config files
and the `constants/` folder when present, and pushes them to a new or existing
HF repo under the given namespace.

It mirrors the spirit of the `push_to_hub` helpers found in the project's
`DiacritizationModule` and `ArabicDiacritizationTokenizer` implementations but
works directly from existing checkpoints on disk.

Usage examples:
  # dry-run to see discovered checkpoints and planned repo ids
  python _push_to_hub.py --dry-run

  # push found checkpoints to 'my-username' namespace using token from env
  python _push_to_hub.py --namespace my-username

  # specify additional roots to search
  python _push_to_hub.py --roots outputs/results results-final

Requirements: pip install huggingface_hub torch
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import tempfile
import textwrap
from pathlib import Path
from typing import List, Optional

try:
    from huggingface_hub import create_repo, Repository, HfApi
except Exception:  # pragma: no cover - runtime requirement
    create_repo = None
    Repository = None
    HfApi = None

try:
    import torch
except Exception:  # allow script to run without torch for dry-run
    torch = None


CHECKPOINT_GLOBS = [
    # "**/checkpoints/*.ckpt",
    "**/checkpoints/*best*.ckpt",
    "**/checkpoints/*.pt",
    "**/checkpoints/*.pth",
    # "**/*.ckpt",
    "**/*.pt",
    "**/*.pth",
    "**/pytorch_model.bin",
    "**/model.pt",
    "**/model.pth",
]


def find_checkpoints(roots: List[str]) -> List[Path]:
    found = []
    for root in roots:
        for pat in CHECKPOINT_GLOBS:
            for p in Path(root).glob(pat):
                # ignore hidden dirs
                if any(part.startswith(".") for part in p.parts):
                    continue
                found.append(p.resolve())
    # deduplicate while preserving order
    seen = set()
    out = []
    for p in found:
        if str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out


def sanitize_repo_name(rel_path: str, max_len: int = 80) -> str:
    # convert path to safe repo name: replace separators and spaces
    # replace path separators, spaces and plus signs which can be problematic
    name = rel_path.replace(os.sep, "-").replace("/", "-").replace(" ", "-").replace("+", "-")
    # remove characters that are problematic for HF repo names
    # don't allow '+' in final repo name (some tools and URLs treat it specially)
    safe = "".join(c for c in name if c.isalnum() or c in "-_.")
    # collapse repeated dashes
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe[:max_len].strip("-_.+") or "model"

def get_model_dataset(ckpt_path: Path) -> str:
    # get the model dataset from the checkpoint path
    dataset = ckpt_path.parent.parent.parent.parent.name
    model = ckpt_path.parent.parent.parent.parent.parent.name
    return f"diac-{model}-{dataset}"

def find_constants_dir(start: Path) -> Optional[Path]:
    # walk upwards from start to find a 'constants' folder
    cur = start
    for _ in range(6):  # limit upward search depth
        cand = cur / "constants"
        if cand.is_dir():
            return cand.resolve()
        if cur.parent == cur:
            break
        cur = cur.parent
    return None

def find_config_files(start: Path) -> List[Path]:
    # look for common config files in ancestor or same dirs
    paths = []
    cand_names = ["config.json", "hparams.yaml", "hparams.yml", "config.yml", "config.yaml"]
    cur = start
    for _ in range(6):
        for name in cand_names:
            p = cur / name
            if p.exists():
                paths.append(p.resolve())
        if cur.parent == cur:
            break
        cur = cur.parent
    return paths


def push_single_checkpoint(
    checkpoint: Path,
    repo_id: str,
    hf_token: Optional[str],
    include_constants: bool = True,
    dry_run: bool = False,
    upload_method: str = "git",
):
    """Collect files and push to `repo_id`. Returns tmpdir path on success.
    If dry_run is True, do not perform network operations and only print planned actions.
    """
    print(f"\n==> Processing checkpoint: {checkpoint}")

    # determine ancestor base for naming and adjacent files
    base_dir = checkpoint.parent

    # collect candidate files
    files_to_copy = []

    # copy the checkpoint itself
    files_to_copy.append((checkpoint, checkpoint.name))

    # try to find configs
    cfgs = find_config_files(base_dir)
    for c in cfgs:
        files_to_copy.append((c, c.name))

    # try to find constants folder
    constants_dir = find_constants_dir(base_dir) if include_constants else None

    planned = {
        "repo_id": repo_id,
        "checkpoint": str(checkpoint),
        "configs": [str(p) for p in cfgs],
        "constants_dir": str(constants_dir) if constants_dir is not None else None,
    }

    if dry_run:
        print("Planned push:")
        print(textwrap.indent(textwrap.dedent(f"""
            repo: {planned['repo_id']}
            checkpoint: {planned['checkpoint']}
            configs: {planned['configs']}
            constants: {planned['constants_dir']}
        """), "  "))
        return None

    if create_repo is None or Repository is None:
        raise RuntimeError("huggingface_hub is required to push models. Install it with `pip install huggingface_hub`.")

    # create repo (id may include namespace)
    if hf_token is None:
        print("Warning: No Hugging Face token provided (use --token or set HF_TOKEN).\n"
              "Creating repositories or pushing to private repos will likely fail without a token.")

    try:
        # prefer create_repo helper when available
        if create_repo is not None:
            create_repo(repo_id=repo_id, exist_ok=True, token=hf_token)
        elif HfApi is not None:
            HfApi().create_repo(repo_id, exist_ok=True, token=hf_token)
    except Exception as e:
        # Surface create_repo failures so the user can act (token/permissions/etc.)
        print(f"Warning: create_repo failed for '{repo_id}': {e}")

    tmpdir = tempfile.mkdtemp(prefix="push_hub_")

    # For git-based uploads we need to clone the repo first and then copy files into it.
    repo = None
    if upload_method == "git":
        try:
            repo = Repository(tmpdir, clone_from=repo_id, use_auth_token=hf_token)
        except Exception as e:
            # Provide a clearer actionable message when clone fails
            raise RuntimeError(
                f"Failed to clone repository '{repo_id}': {e}\n"
                "This usually means the repo does not exist or your token is missing/invalid.\n"
                "If you intended to create the repo automatically, ensure you provided a valid token with `--token` or set HF_TOKEN in the environment and that the token has `repo` scope.\n"
                "You can also create the repository manually on the Hugging Face website and re-run the script."
            )

    # copy files into tmpdir root
    for src, dest_name in files_to_copy:
        target = os.path.join(tmpdir, dest_name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(src, target)

    # debug: list what was copied into the tmpdir
    print("Files copied into temporary repo:")
    for root, dirs, files in os.walk(tmpdir):
        rel = os.path.relpath(root, tmpdir)
        for f in files:
            p = os.path.join(root, f)
            try:
                size = os.path.getsize(p)
            except Exception:
                size = "?"
            print(f" - {os.path.join(rel, f)} (size={size})")

    # show git status before committing to help debug failures
    try:
        import subprocess
        status = subprocess.run(["git", "status", "--porcelain"], cwd=tmpdir, capture_output=True, text=True)
        print("\nGit status (before add/commit):")
        print(status.stdout.strip() or "(clean)")
    except Exception as e:
        print("Could not run git status:", e)

    # copy constants folder if found
    if constants_dir:
        dest_constants = os.path.join(tmpdir, "constants")
        if os.path.exists(dest_constants):
            shutil.rmtree(dest_constants)
        shutil.copytree(constants_dir, dest_constants)

    # attempt to generate a pytorch model state file (pytorch_model.bin) when possible
    if torch is not None:
        try:
            ck = torch.load(str(checkpoint), map_location="cpu")
            state = None
            if isinstance(ck, dict) and "state_dict" in ck:
                # lightning checkpoint
                sd = ck["state_dict"]
                # convert keys like 'model.xxx' -> 'xxx'
                state = {k.replace("model.", ""): v for k, v in sd.items()}
            elif isinstance(ck, dict):
                # plain state dict
                state = ck

            if state is not None:
                out_path = os.path.join(tmpdir, "pytorch_model.bin")
                torch.save(state, out_path)
                print(f"Saved extracted state_dict to {out_path}")
        except Exception:
            # not fatal
            pass

    # write a small README describing source
    readme = os.path.join(tmpdir, "README.md")
    with open(readme, "w", encoding="utf-8") as fh:
        fh.write("# Uploaded model\n\n")
        fh.write(f"Source checkpoint: {checkpoint}\n\n")
        if cfgs:
            fh.write("Included config files:\n")
            for c in cfgs:
                fh.write(f" - {c}\n")
        if constants_dir:
            fh.write(f"\nIncluded tokenizer constants from: {constants_dir}\n")

    # If using HTTP upload method, upload folder contents via the Hub API and return
    if upload_method == "http":
        if HfApi is None:
            raise RuntimeError("huggingface_hub.HfApi is required for HTTP uploads. Install huggingface_hub>=0.14.0")

        api = HfApi()
        print(f"Uploading folder {tmpdir} to {repo_id} via HTTP API...")
        try:
            # prefer upload_folder if available
            if hasattr(api, "upload_folder"):
                api.upload_folder(
                    folder_path=tmpdir,
                    path_in_repo="",
                    repo_id=repo_id,
                    token=hf_token,
                    commit_message="Upload model checkpoint and constants (http)",
                )
            else:
                # fallback: upload files individually
                for root, dirs, files in os.walk(tmpdir):
                    for f in files:
                        local_path = os.path.join(root, f)
                        repo_path = os.path.relpath(local_path, tmpdir)
                        api.upload_file(
                            path_or_fileobj=local_path,
                            path_in_repo=repo_path,
                            repo_id=repo_id,
                            token=hf_token,
                            commit_message="Upload model checkpoint and constants (http)",
                        )
        except Exception as e:
            raise RuntimeError(f"HTTP upload to {repo_id} failed: {e}")

        print(f"Uploaded to {repo_id} via HTTP API (tmpdir {tmpdir})")
        return tmpdir

    # commit and push (wrapped for better diagnostics)
    try:
        repo.git_add(pattern="*")
    except Exception as e:
        print(f"repo.git_add failed: {e}")

    try:
        # show git status after add
        try:
            import subprocess
            status2 = subprocess.run(["git", "status", "--porcelain"], cwd=tmpdir, capture_output=True, text=True)
            print("\nGit status (after add):")
            print(status2.stdout.strip() or "(clean)")
        except Exception:
            pass

        repo.git_commit("Upload model checkpoint and constants")
    except Exception as e:
        print(f"repo.git_commit failed: {e}")

    try:
        repo.push_to_hub()
    except Exception as e:
        print(f"repo.push_to_hub failed: {e}")
        # surface git logs to help debug
        try:
            import subprocess
            out = subprocess.run(["git", "log", "-n", "5", "--oneline"], cwd=tmpdir, capture_output=True, text=True)
            print("Recent git log in tmpdir:")
            print(out.stdout.strip() or "(no commits)")
        except Exception:
            pass
        raise

    print(f"Pushed to {repo_id} (tmpdir {tmpdir})")
    return tmpdir


def main():
    parser = argparse.ArgumentParser(description="Push trained models and tokenizer constants to the Hugging Face Hub.")
    parser.add_argument("--roots", nargs="+", default=["outputs/results", "results-final", "results"], help="Root folders to search for checkpoints")
    parser.add_argument("--namespace", required=False, help="HF username/namespace to push repos under (e.g. myuser). If omitted, pass full repo ids via --full-repo-ids or set --dry-run to inspect.")
    parser.add_argument("--full-repo-ids", nargs="*", help="Optional list of full repo ids to map to discovered checkpoints in order. If provided, length must match discovered checkpoints.")
    parser.add_argument("--token", required=False, help="Hugging Face token. Falls back to HF_TOKEN env var.")
    parser.add_argument("--include-constants", action="store_true", help="Include nearby 'constants/' folder when found")
    parser.add_argument("--dry-run", action="store_true", help="Don't push; just print planned actions")
    parser.add_argument("--upload-method", choices=["git", "http"], default="git", help="Upload method: 'git' (default) uses git repo clone and push; 'http' uses the HF HTTP API to upload files directly")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of checkpoints to process (0 = all)")
    args = parser.parse_args()

    hf_token = args.token or os.environ.get("HF_TOKEN")

    roots = [r for r in args.roots if os.path.exists(r)]
    if not roots:
        print("No search roots found. Exiting.")
        return

    checkpoints = find_checkpoints(roots)
    if not checkpoints:
        print("No checkpoints found under:", roots)
        return

    if args.limit > 0:
        checkpoints = checkpoints[: args.limit]

    # prepare repo ids if user passed full ids
    full_ids = args.full_repo_ids or []

    for i, ck in enumerate(checkpoints):
        # determine repo id
        if i < len(full_ids):
            repo_id = full_ids[i]
        elif args.namespace:
            # create a repo name from relative path to nearest provided root
            chosen_root = None
            for r in roots:
                try:
                    rel = os.path.relpath(ck, r)
                    if not rel.startswith(".."):
                        chosen_root = r
                        break
                except Exception:
                    continue
            rel_base = os.path.relpath(ck.parent, chosen_root) if chosen_root else str(ck.parent)
            model_dataset = get_model_dataset(ck)
            repo_name = sanitize_repo_name(model_dataset)
            repo_id = f"{args.namespace}/{repo_name}"
        else:
            # fallback: use directory name as repo id (user must set namespace)
            model_dataset = get_model_dataset(ck)
            repo_name = sanitize_repo_name(model_dataset)
            repo_id = repo_name
        try:
            push_single_checkpoint(
                checkpoint=ck,
                repo_id=repo_id,
                hf_token=hf_token,
            include_constants=args.include_constants,
            dry_run=args.dry_run,
            upload_method=args.upload_method,
            )
        except Exception as e:
            print(f"Failed to push {ck}: {e}")


if __name__ == "__main__":
    main()

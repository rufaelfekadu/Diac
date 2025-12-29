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
import json
import os
import re
import shutil
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

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
    # "**/checkpoints/*.pt",
    # "**/checkpoints/*.pth",
    # "**/*.ckpt",
    # "**/*.pt",
    # "**/*.pth",
    # "**/pytorch_model.bin",
    # "**/model.pt",
    # "**/model.pth",
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
    name = (
        rel_path.replace(os.sep, "-")
        .replace("/", "-")
        .replace(" ", "-")
        .replace("+", "-")
    )
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


def find_eval_logs(checkpoint_path: Path) -> List[Path]:
    """Find evaluation log files near the checkpoint directory.
    
    Looks for logs in: <checkpoint_dir>/../logs/eval-*.log
    Also checks parent directories up to 6 levels.
    """
    logs = []
    # Check checkpoint parent and ancestors for logs directory
    cur = checkpoint_path.parent
    for _ in range(6):
        logs_dir = cur / "logs"
        if logs_dir.exists() and logs_dir.is_dir():
            for log_file in logs_dir.glob("eval-*.log"):
                logs.append(log_file)
        if cur.parent == cur:
            break
        cur = cur.parent
    return logs


def parse_metric_line(line: str) -> Optional[List[float]]:
    """Parse a metric table row line to extract 4 float values.
    
    Expected format: |   %   |    6.43    |    4.95    |    7.89    |    6.03    |
    """
    # Match lines like: |   %   | ... | ... | ... | ... |
    m = re.match(r"^\|\s*%\s*\|(.+?)\|\s*$", line)
    if not m:
        return None
    inner = m.group(1)
    parts = [p.strip() for p in inner.split("|")]
    nums: List[float] = []
    for p in parts:
        p2 = p.replace(",", ".")
        if re.search(r"\d", p2):
            try:
                nums.append(float(p2))
            except ValueError:
                continue
    if len(nums) >= 4:
        return nums[:4]
    return None


def parse_eval_log(log_path: Path) -> Optional[Dict]:
    """Parse a single evaluation log file and extract metrics and metadata.
    
    Returns a dict with:
    - model: model name
    - model_type: model type (e.g., text-only, text-asr)
    - dataset: training dataset
    - eval_set: evaluation dataset
    - der: [4 values] - DER metrics
    - wer: [4 values] - WER metrics  
    - ser: [4 values] - SER metrics
    """
    model = model_type = training_dataset = eval_set = None
    der_values = wer_values = ser_values = None
    expecting_der = expecting_wer = expecting_ser = False
    
    MODEL_RE = re.compile(r"Model:\s*(\S+)")
    MODEL_TYPE_RE = re.compile(r"Model Type:\s*(\S+)")
    DATASET_RE = re.compile(r"Dataset:\s*(\S+)")
    TEST_FILE_RE = re.compile(r"Test file:\s*(\S+)")
    
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_stripped = line.strip()
                
                # Extract metadata
                if "Model:" in line_stripped and model is None:
                    mm = MODEL_RE.search(line_stripped)
                    if mm:
                        model = mm.group(1)
                elif "Model Type:" in line_stripped and model_type is None:
                    mt = MODEL_TYPE_RE.search(line_stripped)
                    if mt:
                        model_type = mt.group(1)
                elif "Dataset:" in line_stripped and training_dataset is None:
                    ds = DATASET_RE.search(line_stripped)
                    if ds:
                        training_dataset = ds.group(1)
                elif "Test file:" in line_stripped and eval_set is None:
                    tf = TEST_FILE_RE.search(line_stripped)
                    if tf:
                        parts = Path(tf.group(1)).parts
                        if "data" in parts:
                            idx = parts.index("data")
                            if idx + 1 < len(parts):
                                eval_set = parts[idx + 1]
                
                # Identify metric sections
                if re.search(r"\|\s+DER\s+\|", line):
                    expecting_der = True
                    expecting_wer = False
                    expecting_ser = False
                elif re.search(r"\|\s+WER\s+\|", line):
                    expecting_wer = True
                    expecting_der = False
                    expecting_ser = False
                elif re.search(r"\|\s+SER\s+\|", line):
                    expecting_ser = True
                    expecting_der = False
                    expecting_wer = False
                
                # Parse metric values
                if "|   %" in line:
                    vals = parse_metric_line(line)
                    if vals:
                        if expecting_der and der_values is None:
                            der_values = vals
                            expecting_der = False
                        elif expecting_wer and wer_values is None:
                            wer_values = vals
                            expecting_wer = False
                        elif expecting_ser and ser_values is None:
                            ser_values = vals
                            expecting_ser = False
                
                # Reset if we hit a separator line
                if line.strip().startswith("+") and (expecting_der or expecting_wer or expecting_ser):
                    expecting_der = expecting_wer = expecting_ser = False
                    
    except Exception as e:
        print(f"Warning: failed to parse {log_path}: {e}")
        return None
    
    # Fallback: try to extract from path
    if (model is None or model_type is None or training_dataset is None) and "results" in log_path.parts:
        try:
            idx = log_path.parts.index("results")
            if idx + 2 < len(log_path.parts):
                model_full_dir = log_path.parts[idx + 1]
                path_training_dataset = log_path.parts[idx + 2]
                if model is None and model_type is None and "-" in model_full_dir:
                    tokens = model_full_dir.split("-")
                    model = tokens[0]
                    model_type = "-".join(tokens[1:])
                else:
                    model = model or model_full_dir
                    model_type = model_type or "unknown"
                training_dataset = training_dataset or path_training_dataset
        except Exception:
            pass
    
    if model is None:
        return None
    
    result = {
        "model": model or "unknown",
        "model_type": model_type or "unknown",
        "dataset": training_dataset or "unknown",
        "eval_set": eval_set or "unknown",
        "der": der_values,
        "wer": wer_values,
        "ser": ser_values,
    }
    return result


def extract_model_metadata(checkpoint: Path, configs: List[Path]) -> Dict:
    """Extract model metadata for YAML frontmatter.
    
    Returns a dict with model information extracted from checkpoint path and configs.
    """
    metadata = {
        "language": ["ar"],
        "tags": ["diacritization", "nlp", "arabic"],
        "metrics": ["DER", "WER", "SER"],
    }
    
    # Try to extract model type from checkpoint path
    # Pattern: results/<model>-<type>/<dataset>/tensorboard/version_*/checkpoints/...
    path_parts = checkpoint.parts
    if "results" in path_parts:
        try:
            idx = path_parts.index("results")
            if idx + 1 < len(path_parts):
                model_full = path_parts[idx + 1]
                if "-" in model_full:
                    parts = model_full.split("-")
                    model_name = parts[0]
                    model_type = "-".join(parts[1:])
                    metadata["model_type"] = model_type.lower()
                    if model_type.lower() in ["transformer", "lstm"]:
                        metadata["tags"].append(model_type.lower())
                else:
                    metadata["model_type"] = model_full.lower()
        except Exception:
            pass
    
    # Try to extract dataset from path
    if "results" in path_parts:
        try:
            idx = path_parts.index("results")
            if idx + 2 < len(path_parts):
                dataset = path_parts[idx + 2]
                metadata["datasets"] = [dataset]
        except Exception:
            pass
    
    # Try to read config files for additional metadata
    for config_path in configs:
        try:
            if config_path.suffix in [".yaml", ".yml"]:
                try:
                    import yaml
                except ImportError:
                    # yaml not available, skip config parsing
                    continue
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)
                    if isinstance(config_data, dict):
                        if "MODEL" in config_data and "TYPE" in config_data["MODEL"]:
                            model_type = config_data["MODEL"]["TYPE"].lower()
                            metadata["model_type"] = model_type
                            if model_type not in metadata["tags"]:
                                metadata["tags"].append(model_type)
        except Exception:
            pass
    
    return metadata


def generate_config_json(
    checkpoint: Path,
    configs: List[Path],
    metadata: Dict,
) -> Dict:
    """Generate a config.json file for Hugging Face Hub download tracking.
    
    Args:
        checkpoint: Path to the checkpoint file
        configs: List of config file paths
        metadata: Model metadata dict from extract_model_metadata
    
    Returns:
        Dictionary representing the config.json content
    """
    config_data = {
        "model_type": metadata.get("model_type", "diacritization"),
        "architecture": metadata.get("model_type", "transformer").title(),
    }
    
    # Try to extract additional config from YAML files
    for config_path in configs:
        try:
            if config_path.suffix in [".yaml", ".yml"]:
                try:
                    import yaml
                except ImportError:
                    continue
                with open(config_path, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f)
                    if isinstance(yaml_data, dict):
                        # Extract model configuration
                        if "MODEL" in yaml_data:
                            model_cfg = yaml_data["MODEL"]
                            if "TYPE" in model_cfg:
                                config_data["model_type"] = model_cfg["TYPE"].lower()
                                config_data["architecture"] = model_cfg["TYPE"]
                            if "D_MODEL" in model_cfg:
                                config_data["d_model"] = model_cfg["D_MODEL"]
                            if "NUM_HEADS" in model_cfg:
                                config_data["num_heads"] = model_cfg["NUM_HEADS"]
                            if "NUM_BLOCKS" in model_cfg:
                                config_data["num_blocks"] = model_cfg["NUM_BLOCKS"]
                            if "DFF" in model_cfg:
                                config_data["dff"] = model_cfg["DFF"]
                            if "USE_ASR" in model_cfg:
                                config_data["use_asr"] = model_cfg["USE_ASR"]
                        break  # Use first valid config
        except Exception:
            continue
    
    # Add dataset info if available
    if "datasets" in metadata:
        config_data["datasets"] = metadata["datasets"]
    
    return config_data


def format_metric_table(metric_name: str, values: Optional[List[float]]) -> str:
    """Format metric values as a Markdown table.
    
    Args:
        metric_name: Name of the metric (DER, WER, SER)
        values: List of 4 float values [w_case_incl, wo_case_incl, w_case_excl, wo_case_excl]
    
    Returns:
        Markdown table string
    """
    if not values or len(values) < 4:
        return f"### {metric_name}\n\n*No evaluation results available.*\n\n"
    
    # Format as a clear table with all 4 metric variants
    # Values: [w_case_incl, wo_case_incl, w_case_excl, wo_case_excl]
    table = f"""{metric_name}

| Configuration | With case ending | Without case ending |
|---|---|---|
| **Including no diacritic** | {values[0]:.2f}% | {values[1]:.2f}% |
| **Excluding no diacritic** | {values[2]:.2f}% | {values[3]:.2f}% |

"""
    return table


def generate_readme(
    checkpoint: Path,
    configs: List[Path],
    eval_results: List[Dict],
    repo_id: str,
) -> str:
    """Generate a comprehensive README.md with YAML frontmatter, instructions, and evaluation results.
    
    Args:
        checkpoint: Path to the checkpoint file
        configs: List of config file paths
        eval_results: List of parsed evaluation result dicts
        repo_id: Hugging Face repository ID
    
    Returns:
        Complete README content as string
    """
    # Extract metadata for YAML
    metadata = extract_model_metadata(checkpoint, configs)
    
    # Build YAML frontmatter
    yaml_lines = ["---"]
    yaml_lines.append(f"language: {metadata['language']}")
    yaml_lines.append(f"tags:")
    for tag in metadata["tags"]:
        yaml_lines.append(f"  - {tag}")
    yaml_lines.append(f"metrics:")
    for metric in metadata["metrics"]:
        yaml_lines.append(f"  - {metric}")
    # if "model_type" in metadata:
    #     yaml_lines.append(f"model_type: {metadata['model_type']}")
    # if "datasets" in metadata:
    #     yaml_lines.append(f"datasets:")
    #     for ds in metadata["datasets"]:
    #         yaml_lines.append(f"  - {ds}")
    yaml_lines.append("---")
    yaml_frontmatter = "\n".join(yaml_lines)
    
    # Build README content
    readme_parts = [yaml_frontmatter, ""]
    readme_parts.append("# Automatic Restoration of Diacritics for Speech Data Sets")
    readme_parts.append("")
    
    # Model description
    readme_parts.append(f"This is a transformer-baed model for Arabic text diacritization as described [here](https://github.com/rufaelfekadu/Diac.git).")
    readme_parts.append("")
    
    
    # Evaluation results
    if eval_results:
        readme_parts.append("## Evaluation Results")
        readme_parts.append("")
        
        # Group results by eval_set if multiple
        for result in eval_results:
            if result.get("eval_set") and result["eval_set"] != "unknown":
                readme_parts.append(f"### Evaluation on {result['eval_set']}")
                readme_parts.append("")
            
            if result.get("der"):
                readme_parts.append(format_metric_table("DER (Diacritic Error Rate)", result["der"]))
            if result.get("wer"):
                readme_parts.append(format_metric_table("WER (Word Error Rate)", result["wer"]))
            # if result.get("ser"):
            #     readme_parts.append(format_metric_table("SER (Sentence Error Rate)", result["ser"]))
    else:
        readme_parts.append("## Evaluation Results")
        readme_parts.append("")
        readme_parts.append("*No evaluation results found in log files.*")
        readme_parts.append("")
    
    # Usage instructions
    readme_parts.append("## How to Use")
    readme_parts.append("")
    readme_parts.append("### Installation")
    readme_parts.append("")
    readme_parts.append("```bash")
    readme_parts.append("git clone https://github.com/rufaelfekadu/diac.git")
    readme_parts.append("cd diac")
    readme_parts.append("pip install -e .")
    readme_parts.append("```")
    readme_parts.append("")
    
    readme_parts.append("### Loading the Model")
    readme_parts.append("")
    readme_parts.append("```python")
    readme_parts.append("from diac.models import DiacritizationModule")
    readme_parts.append("")
    readme_parts.append(f"model = DiacritizationModule.from_pretrained(")
    readme_parts.append(f'    "{repo_id}",')
    readme_parts.append('    tokenizer_constants_path="constants/"  # Path to constants directory')
    readme_parts.append(")")
    readme_parts.append("```")
    readme_parts.append("")
    
    readme_parts.append("### Running Inference")
    readme_parts.append("")
    readme_parts.append("```python")
    readme_parts.append("# Predict diacritization for a text file")
    readme_parts.append("model.predict_file(")
    readme_parts.append('    input_file="path/to/input.txt",')
    readme_parts.append('    output_file="path/to/output.txt"')
    readme_parts.append(")")
    readme_parts.append("")
    readme_parts.append("# Or predict for a single text string")
    readme_parts.append("diacritized_text = model.predict_text(\"مرحبا بك\")")
    readme_parts.append("```")
    readme_parts.append("")
    
    readme_parts.append("### Running Evaluation")
    readme_parts.append("")
    readme_parts.append("To evaluate the model on your own test set:")
    readme_parts.append("")
    readme_parts.append("1. **Run inference** to generate predictions:")
    readme_parts.append("")
    readme_parts.append("```bash")
    readme_parts.append("python inference.py \\")
    readme_parts.append("    --config configs/<model>.yml \\")
    readme_parts.append("    --opts \\")
    readme_parts.append("    DATA.TEST_PATH path/to/test.txt \\")
    readme_parts.append(f"    INFERENCE.MODEL_PATH <path_to_checkpoint> \\")
    readme_parts.append("    INFERENCE.OUTPUT_PATH path/to/predictions.txt")
    readme_parts.append("```")
    readme_parts.append("")
    
    readme_parts.append("2. **Prepare reference file** (if needed):")
    readme_parts.append("")
    readme_parts.append("```bash")
    readme_parts.append("python src/diac/utils/prep_ref.py \\")
    readme_parts.append("    --input_file path/to/test.txt \\")
    readme_parts.append("    -o path/to/output_dir")
    readme_parts.append("```")
    readme_parts.append("")
    
    readme_parts.append("3. **Calculate metrics** (DER, WER, SER):")
    readme_parts.append("")
    readme_parts.append("```bash")
    readme_parts.append("python src/diac/utils/eval.py \\")
    readme_parts.append("    -ofp path/to/predictions.txt \\")
    readme_parts.append("    -tfp path/to/reference.txt \\")
    readme_parts.append("    --style Fadel")
    readme_parts.append("```")
    readme_parts.append("")
    
    readme_parts.append("The evaluation script will output DER, WER, and SER metrics with different configurations:")
    readme_parts.append("- With/without case ending")
    readme_parts.append("- Including/excluding no diacritic")
    readme_parts.append("")
    
    return "\n".join(readme_parts)


def find_config_files(start: Path) -> List[Path]:
    # look for common config files in ancestor or same dirs
    paths = []
    cand_names = [
        "config.json",
        "hparams.yaml",
        "hparams.yml",
        "config.yml",
        "config.yaml",
    ]
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
    include_constants: bool = False,
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


    planned = {
        "repo_id": repo_id,
        "checkpoint": str(checkpoint),
        "configs": [str(p) for p in cfgs],
    }

    if dry_run:
        print("Planned push:")
        print(
            textwrap.indent(
                textwrap.dedent(
                    f"""
            repo: {planned['repo_id']}
            checkpoint: {planned['checkpoint']}
            configs: {planned['configs']}
        """
                ),
                "  ",
            )
        )
        return None

    if create_repo is None or Repository is None:
        raise RuntimeError(
            "huggingface_hub is required to push models. Install it with `pip install huggingface_hub`."
        )

    # create repo (id may include namespace)
    if hf_token is None:
        print(
            "Warning: No Hugging Face token provided (use --token or set HF_TOKEN).\n"
            "Creating repositories or pushing to private repos will likely fail without a token."
        )

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

        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=tmpdir, capture_output=True, text=True
        )
        print("\nGit status (before add/commit):")
        print(status.stdout.strip() or "(clean)")
    except Exception as e:
        print("Could not run git status:", e)


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

    # Find and parse evaluation logs
    eval_logs = find_eval_logs(checkpoint)
    eval_results = []
    for log_path in eval_logs:
        result = parse_eval_log(log_path)
        if result:
            eval_results.append(result)
    
    # Generate comprehensive README
    readme_content = generate_readme(
        checkpoint=checkpoint,
        configs=cfgs,
        eval_results=eval_results,
        repo_id=repo_id,
    )
    
    # write README
    readme = os.path.join(tmpdir, "README.md")
    with open(readme, "w", encoding="utf-8") as fh:
        fh.write(readme_content)
    
    # Generate config.json for Hugging Face Hub download tracking
    # HF Hub tracks downloads through specific query files like config.json
    metadata = extract_model_metadata(checkpoint, cfgs)
    config_json_data = generate_config_json(checkpoint, cfgs, metadata)
    config_json_path = os.path.join(tmpdir, "config.json")
    with open(config_json_path, "w", encoding="utf-8") as fh:
        json.dump(config_json_data, fh, indent=2, ensure_ascii=False)
    print(f"Generated config.json for download tracking: {config_json_path}")

    # If using HTTP upload method, upload folder contents via the Hub API and return
    if upload_method == "http":
        if HfApi is None:
            raise RuntimeError(
                "huggingface_hub.HfApi is required for HTTP uploads. Install huggingface_hub>=0.14.0"
            )

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

            status2 = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )
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

            out = subprocess.run(
                ["git", "log", "-n", "5", "--oneline"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )
            print("Recent git log in tmpdir:")
            print(out.stdout.strip() or "(no commits)")
        except Exception:
            pass
        raise

    print(f"Pushed to {repo_id} (tmpdir {tmpdir})")
    return tmpdir


def main():
    parser = argparse.ArgumentParser(
        description="Push trained models and tokenizer constants to the Hugging Face Hub."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["results/to_push"],
        help="Root folders to search for checkpoints",
    )
    parser.add_argument(
        "--namespace",
        required=False,
        help="HF username/namespace to push repos under (e.g. myuser). If omitted, pass full repo ids via --full-repo-ids or set --dry-run to inspect.",
    )
    parser.add_argument(
        "--full-repo-ids",
        nargs="*",
        help="Optional list of full repo ids to map to discovered checkpoints in order. If provided, length must match discovered checkpoints.",
    )
    parser.add_argument(
        "--token",
        required=False,
        help="Hugging Face token. Falls back to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--include-constants",
        action="store_true",
        help="Include nearby 'constants/' folder when found",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't push; just print planned actions"
    )
    parser.add_argument(
        "--upload-method",
        choices=["git", "http"],
        default="git",
        help="Upload method: 'git' (default) uses git repo clone and push; 'http' uses the HF HTTP API to upload files directly",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of checkpoints to process (0 = all)",
    )
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
            rel_base = (
                os.path.relpath(ck.parent, chosen_root)
                if chosen_root
                else str(ck.parent)
            )
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

from sys import exception
import torch
import torch.nn as nn
import lightning as L
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
from tqdm import tqdm
from pyarabic import araby
import os
import json
import tempfile
from types import SimpleNamespace
import yaml


from diac.models.asr_model import AsrModel
from diac.config import _C as default_config
from diac.tokenizer import ArabicDiacritizationTokenizer
from diac.models import LSTMModel, TransformerModel
from diac.utils.text import remove_diacritics

# Optional huggingface-hub helpers. If not installed, hub-related helpers will raise
# a clear error when used.
try:
    from huggingface_hub import (
        hf_hub_download,
        snapshot_download,
        create_repo,
        Repository,
    )
except Exception:
    hf_hub_download = None
    snapshot_download = None
    create_repo = None
    Repository = None

AVAILABLE_MODELS = {"Transformer": TransformerModel, "LSTM": LSTMModel}


class DiacritizationModule(L.LightningModule):
    """PyTorch Lightning module for diacritization models."""

    @classmethod
    def from_pretrained(
        cls,
        repo_or_dir,
        config=None,
        tokenizer=None,
        tokenizer_constants_path=None,
        map_location="cpu",
        device=None,
        hf_token=None,
    ):
        """Instantiate DiacritizationModule from a local directory or Hugging Face repo.

        This implementation uses PyTorch Lightning's `load_from_checkpoint` to load a
        Lightning checkpoint (e.g., best_model.ckpt). It will download checkpoint/config/constants
        from a local path or HF hub if needed, then delegate loading to Lightning.
        """

        def _download_file(fname):
            local_candidate = os.path.join(repo_or_dir, fname)
            if os.path.exists(local_candidate):
                return local_candidate
            if hf_hub_download is None:
                return None
            try:
                return hf_hub_download(
                    repo_id=repo_or_dir, filename=fname, use_auth_token=hf_token
                )
            except Exception:
                return None

        # find checkpoint
        candidates = [
            "pytorch_model.bin",
            "model.pt",
            "model.pth",
            "state_dict.pt",
            "final_model.pt",
            "best_model.ckpt",
        ]
        ckpt_path = None
        for c in candidates:
            p = _download_file(c)
            if p is not None:
                ckpt_path = p
                break

        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint found in '{repo_or_dir}'. Tried: {candidates}"
            )

        # load or build config
        if config is None:
            cfg_path = _download_file("config.yml")
            config = default_config.clone()
            if cfg_path:
                try:
                    config.merge_from_file(cfg_path)
                except Exception:
                    print(
                        "warning: failed to load config from file, using default config"
                    )
        # build or load tokenizer
        # download constants
        if tokenizer is None and tokenizer_constants_path is None:
            try:
                constants_file = _download_file("constants.json")
                if constants_file:
                    tokenizer = ArabicDiacritizationTokenizer(constants_file)
            except Exception:
                raise RuntimeError(
                    "Failed to load tokenizer from constants.json in the repo."
                )

        elif tokenizer_constants_path is not None and tokenizer is None:
            tokenizer = ArabicDiacritizationTokenizer(tokenizer_constants_path)

        elif isinstance(tokenizer, ArabicDiacritizationTokenizer):
            pass

        else:
            raise ValueError("Error loading tokenizer.")

        if tokenizer is None:
            raise ValueError(
                "Tokenizer instance not provided and constants directory not found in the repo."
            )

        if config is None:
            raise ValueError(
                "Config not provided and config.json not found in the repo."
            )

        # Use Lightning's load_from_checkpoint to construct the module
        try:
            # pass config and tokenizer to the constructor via load_from_checkpoint
            module = cls.load_from_checkpoint(
                ckpt_path, config=config, tokenizer=tokenizer, map_location=map_location
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint with Lightning's load_from_checkpoint: {e}"
            )

        if device is not None:
            module.to(device)

        # Note: text_branch_only and strict handling (for custom partial loads) are not implemented here;
        # if needed, load a plain state_dict and apply selective mapping manually.
        return module

    def push_to_hub(
        self,
        repo_id,
        hf_token=None,
        files_to_include=None,
        commit_message="upload model checkpoint",
    ):
        """Save the module's model weights and push to Hugging Face Hub.

        This saves the internal `self.model` state dict as 'pytorch_model.bin' and
        writes a lightweight 'config.json' (if possible). The tokenizer constants
        directory is not uploaded automatically — include it via files_to_include
        or push it separately.
        """
        if Repository is None or create_repo is None:
            raise RuntimeError(
                "huggingface_hub is required for push_to_hub; please pip install huggingface_hub"
            )

        tmpdir = tempfile.mkdtemp()

        # create repo if necessary
        try:
            create_repo(repo_id=repo_id, exist_ok=True, token=hf_token)
        except Exception:
            pass

        repo = Repository(tmpdir, clone_from=repo_id, use_auth_token=hf_token)

        # save model weights
        model_path = os.path.join(tmpdir, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)

        # save the tokenizer constants.json
        tokenizer_path = os.path.join(tmpdir, "constants.json")
        with open(tokenizer_path, "w", encoding="utf-8") as fh:
            json.dump(self.tokenizer.constants, fh, ensure_ascii=False, indent=2)

        # try to dump a minimal config.json if the config is simple
        try:

            def _ns_to_dict(ns):
                if isinstance(ns, dict):
                    return {k: _ns_to_dict(v) for k, v in ns.items()}
                if isinstance(ns, SimpleNamespace):
                    return {k: _ns_to_dict(v) for k, v in ns.__dict__.items()}
                if hasattr(ns, "__dict__"):
                    return {k: _ns_to_dict(v) for k, v in ns.__dict__.items()}
                return ns

            cfg_dict = _ns_to_dict(self.config)
            with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as fh:
                json.dump(cfg_dict, fh, ensure_ascii=False, indent=2)
        except Exception:
            # non-fatal
            pass

        # include extra files
        if files_to_include:
            for fname, content in files_to_include.items():
                target = os.path.join(tmpdir, fname)
                mode = "wb" if isinstance(content, (bytes, bytearray)) else "w"
                with open(
                    target, mode, encoding="utf-8" if mode == "w" else None
                ) as fh:
                    fh.write(content)

        repo.git_add(pattern="*")
        repo.git_commit(commit_message)
        repo.push_to_hub()
        return tmpdir

    def __init__(self, config, tokenizer: ArabicDiacritizationTokenizer):
        super().__init__()

        self.save_hyperparameters()
        self.config = config
        self.tokenizer = tokenizer

        if config.MODEL.TYPE not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {config.MODEL.TYPE}")

        model_class = AVAILABLE_MODELS[config.MODEL.TYPE]

        self.model = model_class.from_config(config)

        # Load pretrained weights if specified
        if hasattr(config.MODEL, "PRETRAINED_PATH") and config.MODEL.PRETRAINED_PATH:
            self.model.load_pretrained(
                config.MODEL.PRETRAINED_PATH,
                text_branch_only=getattr(config.MODEL, "LOAD_TEXT_BRANCH_ONLY", False),
            )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        # Metrics
        self.train_accuracy = []
        self.val_accuracy = []
        # Gradient clipping configuration (optional) — read from config.TRAIN.GRAD_CLIP_NORM if present
        try:
            # allow configs that may not have TRAIN or GRAD_CLIP_NORM
            self.grad_clip_norm = getattr(self.config.TRAIN, "GRAD_CLIP_NORM", None)
        except Exception:
            self.grad_clip_norm = None

    def forward(self, inputs, inputs_asr=None):
        return self.model(inputs, inputs_asr=inputs_asr)

    def training_step(self, batch, batch_idx):
        inputs, inputs_asr, targets = batch

        # Forward pass
        outputs = self.forward(inputs, inputs_asr=inputs_asr)

        # Calculate loss
        loss = self.criterion(outputs.permute(0, 2, 1), targets)

        # Calculate accuracy
        pred = outputs.argmax(dim=-1)
        correct = (pred == targets).sum().item()
        total = targets.numel()
        accuracy = correct / total

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, inputs_asr, targets = batch

        # Forward pass
        outputs = self.forward(inputs, inputs_asr=inputs_asr)

        # Calculate loss
        loss = self.criterion(outputs.permute(0, 2, 1), targets)

        # Calculate accuracy
        pred = outputs.argmax(dim=-1)
        correct = (pred == targets).sum().item()
        total = targets.numel()
        accuracy = correct / total

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": loss, "val_acc": accuracy}

    def test_step(self, batch, batch_idx):
        inputs, inputs_asr, targets = batch

        # Forward pass
        outputs = self.forward(inputs, inputs_asr=inputs_asr)

        # Calculate loss
        loss = self.criterion(outputs.permute(0, 2, 1), targets)

        # Calculate accuracy
        pred = outputs.argmax(dim=-1)
        correct = (pred == targets).sum().item()
        total = targets.numel()
        accuracy = correct / total

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return {"test_loss": loss, "test_acc": accuracy}

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(), lr=getattr(self.config.TRAIN, "LEARNING_RATE", 1e-4)
        )

        # Optional: Add learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.TRAIN.LR_SCHEDULER_FACTOR,
            patience=self.config.TRAIN.LR_SCHEDULER_PATIENCE,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_after_backward(self):
        """Compute gradient norm after backward, log it, and optionally clip gradients.

        - Logs `grad_norm` (L2 norm of gradients) on each step.
        - If `self.grad_clip_norm` is set (not None/0), applies
          torch.nn.utils.clip_grad_norm_ and logs `grad_norm_clipped`.
        This hook runs after loss.backward() and before the optimizer step.
        """
        # collect parameters with gradients
        params = [p for p in self.parameters() if p.grad is not None]
        if not params:
            return

        # compute total norm robustly (handle sparse grads)
        total_norm_sq = 0.0
        for p in params:
            grad = p.grad.detach()
            if grad.is_sparse:
                # sparse grads: use coalesced values
                param_norm = grad.coalesce().values().norm(2)
            else:
                param_norm = grad.norm(2)
            total_norm_sq += param_norm.item() ** 2

        total_norm = total_norm_sq**0.5

        # log the raw gradient norm (per-step)
        try:
            # prefer richer signature when available
            self.log(
                "grad_norm", total_norm, on_step=True, on_epoch=False, prog_bar=False
            )
        except Exception:
            # fallback
            self.log("grad_norm", total_norm)

        # Clip gradients if requested via config
        if self.grad_clip_norm:
            try:
                clipped_norm = torch.nn.utils.clip_grad_norm_(
                    params, self.grad_clip_norm
                )
            except Exception:
                # defensive: if clipping failed, skip
                clipped_norm = None

            if clipped_norm is not None:
                try:
                    self.log(
                        "grad_norm_clipped",
                        clipped_norm,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=False,
                    )
                except Exception:
                    self.log("grad_norm_clipped", clipped_norm)

    def predict_step(self, batch, batch_idx):
        inputs, inputs_asr, _ = batch
        outputs = self.forward(inputs, inputs_asr=inputs_asr)
        predictions = outputs.argmax(dim=-1)
        return predictions

    def predict_text(self, text, asr_text=[]):
        self.model.eval()

        if isinstance(text, str):
            text = [text]

        if isinstance(asr_text, str):
            asr_text = [asr_text]

        encoded_text, encoded_asr, _ = self.tokenizer.encode_batch(
            text, asr_text, padding=True
        )
        encoded_text = encoded_text.to(self.device)
        encoded_asr = (
            encoded_asr.to(self.device) if self.config.INFERENCE.USE_ASR else None
        )

        with torch.no_grad():
            try:
                outputs = self.model(encoded_text, inputs_asr=encoded_asr)
                predictions = outputs.argmax(dim=-1).cpu().tolist()
            except Exception as e:
                print(f"Error during prediction: {e}")
                return [""] * len(text)

        decoded_texts = self.tokenizer.decode_batch(predictions, text)

        return decoded_texts

    def predict_sliding_window(self, text, asr_text=None):
        self.model.eval()
        original_text = text

        text = remove_diacritics(text).strip()

        _len = len(text)

        if _len == 0:
            return original_text

        r = len(asr_text) / _len if asr_text else 1

        # Sliding window
        if len(text) <= self.config.INFERENCE.MAX_LENGTH:
            output = self.predict_text(text, asr_text=asr_text)
        else:

            window_size = self.config.INFERENCE.WINDOW_SIZE
            buffer_size = getattr(self.config.INFERENCE, "BUFFER_SIZE", 25)
            start_idx = 0
            end_idx = window_size
            output = ""

            while end_idx < len(text):
                start = max(0, start_idx - buffer_size)
                end = min(len(text), end_idx + window_size + buffer_size)
                end_idx = min(len(text), start_idx + window_size)

                chunk = text[start:end]
                chunk_asr = asr_text[int(start * r) : int(end * r)] if asr_text else []
                encoded_chunk, encoded_asr_chunk, _ = self.tokenizer.encode(
                    chunk, chunk_asr, return_tensor=True
                )
                encoded_chunk = encoded_chunk.to(self.device)
                encoded_asr_chunk = (
                    encoded_asr_chunk.to(self.device) if chunk_asr else None
                )

                with torch.no_grad():
                    outputs = self.model(
                        encoded_chunk, inputs_asr=encoded_asr_chunk
                    ).squeeze(0)
                    predictions = (
                        outputs.argmax(dim=-1).cpu().tolist()
                    )  # remove <sos> and <eos>

                # if end_idx > len(text) - buffer_size:
                #     decoded_chunk += self.tokenizer.decode(predictions[end_idx:], chunk[end_idx:])
                #     output += decoded_chunk
                #     break
                # else:

                decoded_chunk = self.tokenizer.decode(
                    predictions[start_idx - start : end_idx - start],
                    chunk[start_idx - start : end_idx - start],
                )

                output += decoded_chunk
                start_idx = end_idx
            output = [output]
        return output

    @staticmethod
    def is_audio(path: str) -> bool:
        audio_extensions = [".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a"]
        return any(path.lower().endswith(ext) for ext in audio_extensions)

    def predict_file(self, input_file, output_file):

        # clear output file if exists
        open(output_file, "w").close()

        # determine the structure of the input file| if first col is audio paths or text
        with open(input_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            is_audio_file = self.is_audio(first_line.split("\t")[0])
            is_asr_text = len(first_line.split("\t")) > 1 and not is_audio_file

        if is_audio_file and self.config.INFERENCE.USE_ASR:
            # input file contains audio paths
            asr_model = AsrModel(
                model_name=self.config.INFERENCE.ASR_MODEL_NAME,
                device=self.config.INFERENCE.DEVICE,
                forced_ids=None,
            )
            print("using audio files...")
            with open(input_file, "r", encoding="utf-8") as f_in, open(
                output_file, "w", encoding="utf-8"
            ) as f_out:
                reader = csv.reader(f_in, delimiter="\t")
                for line in reader:
                    if not line:
                        continue
                    audio_path = line[0]
                    asr_text = asr_model.transcribe(audio_path) if asr_model else None
                    diacritized_line = self.predict_sliding_window(
                        line[1], text_asr=asr_text
                    )
                    f_out.write(diacritized_line + "\n")
            return

        elif is_asr_text and self.config.INFERENCE.USE_ASR:
            # input file contains ASR text in second column
            print("using ASR text...")
            with open(input_file, "r", encoding="utf-8") as f_in, open(
                output_file, "w", encoding="utf-8"
            ) as f_out:
                reader = csv.reader(f_in, delimiter="\t")
                for line in tqdm(reader, desc="Processing lines"):
                    if not line:
                        continue
                    diacritized_line = self.predict_sliding_window(
                        line[0], text_asr=line[1]
                    )[0]
                    f_out.write(diacritized_line + "\n")
            return

        else:
            print("without ASR text...")
            with open(input_file, "r", encoding="utf-8") as f_in, open(
                output_file, "w", encoding="utf-8"
            ) as f_out:
                reader = csv.reader(f_in, delimiter="\t")
                for line in tqdm(reader, desc="Processing lines"):
                    if not line:
                        continue
                    diacritized_line = self.predict_sliding_window(line[0])[0]
                    f_out.write(diacritized_line + "\n")
            return

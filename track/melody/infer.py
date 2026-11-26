from __future__ import annotations

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional

# -------------------------------------------------------------
# MuPT local model config (update if your path changes)
# -------------------------------------------------------------
MODEL_DIR: str = os.getenv("MUSICAGENT_MUPT_MODEL_DIR", "")
DEVICE: str = os.getenv("MUSICAGENT_MUPT_DEVICE", "").strip() or (
    "cuda" if torch.cuda.is_available() else "cpu"
)

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None
_loaded_model_key: Optional[tuple] = None


def _ensure_prefix_one_line(prefix: str) -> str:
    """Enforce the MuPT prefix to be exactly one physical line.
    - Remove real newlines and surrounding spaces.
    - Do NOT modify the literal '<n>' placeholders.
    """
    return str(prefix).replace("\r\n", "\n").replace("\n", "<n>").strip()


def load_model(
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    policy_checkpoint: str = "",
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load tokenizer and model once and cache them globally.
    The model is moved to DEVICE and set to eval().
    """
    global _tokenizer, _model, _loaded_model_key
    from model_registry import resolve_model
    resolved_model_dir = resolve_model("mupt", model_dir or MODEL_DIR)
    resolved_device = device or DEVICE
    from pathlib import Path
    resolved_checkpoint = str(Path(policy_checkpoint).resolve()) if policy_checkpoint else ""
    model_key = (resolved_model_dir, resolved_device, resolved_checkpoint)
    if _tokenizer is not None and _model is not None and _loaded_model_key == model_key:
        return _tokenizer, _model

    # Tokenizer
    from pathlib import Path
    tokenizer_source = Path(resolved_model_dir) / "tokenization_mupt.py"
    if tokenizer_source.is_file():
        # transformers 4.36 treats dots in local checkpoint directory names as
        # package separators. Load the existing local tokenizer under a stable
        # Python identifier; no model files are renamed or downloaded.
        import hashlib, importlib.util, sys
        module_name = "_coma_mupt_tokenizer_" + hashlib.sha256(str(tokenizer_source).encode()).hexdigest()[:12]
        if module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(module_name, tokenizer_source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        _tokenizer = sys.modules[module_name].MuPTTokenizer.from_pretrained(resolved_model_dir, local_files_only=True)
    else:
        _tokenizer = AutoTokenizer.from_pretrained(resolved_model_dir,trust_remote_code=True,use_fast=False,local_files_only=True)

    # Model
    _model = AutoModelForCausalLM.from_pretrained(
        resolved_model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16 if resolved_device.startswith("cuda") else torch.float32,
    )

    if resolved_device.startswith("cuda"):
        _model = _model.to(dtype=torch.float16, device=resolved_device)
    else:
        _model = _model.to(dtype=torch.float32, device=resolved_device)

    if resolved_checkpoint:
        state = torch.load(resolved_checkpoint, map_location="cpu", weights_only=True)
        metadata = state["metadata"]
        if metadata.get("family") != "symbolic" or metadata.get("base_model", {}).get("path") != resolved_model_dir:
            raise ValueError("symbolic policy checkpoint does not match the selected MuPT base")
        ranks = {v.shape[0] for n, v in state["parameters"].items() if n.endswith(".a.weight")}
        if len(ranks) != 1:
            raise ValueError("symbolic checkpoint needs a consistent LoRA rank")
        from training.policies import SymbolicTokenPolicy
        policy = SymbolicTokenPolicy(_model, rank=ranks.pop())
        expected = {n for n,p in policy.named_parameters() if p.requires_grad}
        if set(state["parameters"]) != expected:
            raise ValueError("symbolic checkpoint parameter names do not match this implementation")
        current = dict(policy.named_parameters())
        with torch.no_grad():
            for name, value in state["parameters"].items():
                if current[name].shape != value.shape: raise ValueError("symbolic checkpoint shape mismatch")
                current[name].copy_(value)
        _model.requires_grad_(False)
    _model.eval()
    _loaded_model_key = model_key
    return _tokenizer, _model


@torch.no_grad()
def generate(
    prefix: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.95,
    do_sample: bool = True,
    seed: Optional[int] = None,
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    policy_checkpoint: str = "",
) -> str:
    """Continue ABC text from a single-line MuPT prefix.

    Args:
        prefix: A *single-line* string joined with literal '<n>' between ABC headers,
                e.g. 'X:1<n>L:1/8<n>Q:128<n>M:4/4<n>K:Gmin<n>|:"Gm" BGdB'
        max_new_tokens: Number of new tokens to sample.
        temperature: Sampling temperature (ignored if do_sample=False).
        top_p: Nucleus sampling probability mass (ignored if do_sample=False).
        do_sample: Whether to sample (True) or do greedy decoding (False).
        seed: Optional RNG seed for reproducibility.
        model_dir: Custom model directory (defaults to MODEL_DIR).

    Returns:
        Full decoded text: the original prefix + model continuation.
    """
    tokenizer, model = load_model(model_dir=model_dir, device=device, policy_checkpoint=policy_checkpoint)
    # Loading a checkpoint initializes temporary modules and consumes RNG.
    # Seed after loading so warm-cache and cold-cache calls sample identically.
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    one_line_prefix = _ensure_prefix_one_line(prefix)
    inputs = tokenizer(one_line_prefix, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )

    if do_sample:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
    else:
        gen_kwargs.update(dict(do_sample=False))

    outputs = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


if __name__ == "__main__":
    # Minimal manual test: ensure your prefix is EXACTLY one line with '<n>' placeholders
    example_prefix = 'X:1<n>L:1/8<n>Q:128<n>M:4/4<n>K:Gmin<n>|:"Gm" BGdB'
    result = generate(example_prefix, max_new_tokens=256, temperature=0.8, top_p=0.95)
    print(result)

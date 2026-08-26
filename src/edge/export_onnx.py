"""Merge the LoRA adapter into the base model and export ONNX / INT8 builds.

The adapter is merged first so the exported graph is a single self-contained
model with no PEFT dependency at inference time, which is what an actual edge
deployment would ship.

    python src/edge/export_onnx.py
    python src/edge/export_onnx.py --adapter experiments/smollm2-lora-boundary

Outputs land under experiments/edge/ (gitignored, they are large):

    merged/                fp32 PyTorch, adapter folded into the base weights
    onnx-fp32/             ONNX export of the merged model
    onnx-int8/             dynamic INT8, per-tensor weight scales
    onnx-int8-perchannel/  dynamic INT8, per-output-channel weight scales

Both INT8 variants are produced because they do not behave the same on a model
this small; see REPORT.md for the measured accuracy of each.
"""

import argparse
import shutil
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEFAULT_ADAPTER = Path("experiments/smollm2-lora-policy")
OUTPUT_ROOT = Path("experiments/edge")


def directory_size_mb(path):
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return round(total / 1024**2, 1)


def merge_adapter(adapter, destination):
    print(f"Merging {adapter} into {BASE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Export runs on CPU, so keep the merge in fp32; fp16 ONNX export of a
    # causal LM to a CPU provider is not reliably supported.
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()

    destination.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    print(f"  merged -> {destination} ({directory_size_mb(destination)} MB)")
    return destination


def export_onnx(merged_dir, destination):
    from optimum.onnxruntime import ORTModelForCausalLM

    print("Exporting to ONNX ...")
    started = time.perf_counter()
    model = ORTModelForCausalLM.from_pretrained(merged_dir, export=True)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    AutoTokenizer.from_pretrained(merged_dir).save_pretrained(destination)
    print(
        f"  onnx -> {destination} ({directory_size_mb(destination)} MB) "
        f"in {time.perf_counter() - started:.1f}s"
    )
    return destination


# The output projection maps the hidden state onto a ~49k vocabulary, and the
# classifier reads its logits for exactly four label tokens. Quantisation error
# there lands directly on the decision, so it is worth measuring with this node
# left in fp32.
LM_HEAD_NODE = "/lm_head/MatMul"


def quantize_int8(onnx_dir, destination, per_channel, exclude_lm_head=False):
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    mode = "per-channel" if per_channel else "per-tensor"
    if exclude_lm_head:
        mode += ", fp32 lm_head"
    print(f"Quantising to dynamic INT8 ({mode}) ...")
    started = time.perf_counter()
    destination.mkdir(parents=True, exist_ok=True)

    # Dynamic quantisation needs no calibration set, which suits a classifier
    # whose activations vary widely across document types. Per-tensor scaling is
    # the cheaper default; per-channel keeps a separate scale for each output
    # channel and usually survives better on small models whose weight ranges
    # differ sharply between channels.
    config = AutoQuantizationConfig.avx512_vnni(
        is_static=False,
        per_channel=per_channel,
        nodes_to_exclude=[LM_HEAD_NODE] if exclude_lm_head else None,
    )

    for model_file in sorted(Path(onnx_dir).glob("*.onnx")):
        quantizer = ORTQuantizer.from_pretrained(onnx_dir, file_name=model_file.name)
        quantizer.quantize(save_dir=destination, quantization_config=config)

    for extra in Path(onnx_dir).glob("*.json"):
        shutil.copy2(extra, destination / extra.name)
    AutoTokenizer.from_pretrained(onnx_dir).save_pretrained(destination)

    print(
        f"  int8 ({mode}) -> {destination} "
        f"({directory_size_mb(destination)} MB) "
        f"in {time.perf_counter() - started:.1f}s"
    )
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    merged_dir = args.output / "merged"
    onnx_dir = args.output / "onnx-fp32"
    int8_dir = args.output / "onnx-int8"
    int8_channel_dir = args.output / "onnx-int8-perchannel"

    if not args.skip_merge or not merged_dir.exists():
        merge_adapter(args.adapter, merged_dir)

    if not args.skip_export or not onnx_dir.exists():
        export_onnx(merged_dir, onnx_dir)

    int8_safehead_dir = args.output / "onnx-int8-safehead"

    quantize_int8(onnx_dir, int8_dir, per_channel=False)
    quantize_int8(onnx_dir, int8_channel_dir, per_channel=True)
    quantize_int8(
        onnx_dir, int8_safehead_dir, per_channel=False, exclude_lm_head=True
    )

    print("\nArtifact sizes")
    for path in (merged_dir, onnx_dir, int8_dir, int8_channel_dir, int8_safehead_dir):
        if path.exists():
            print(f"  {path.name:22s} {directory_size_mb(path):8.1f} MB")


if __name__ == "__main__":
    main()

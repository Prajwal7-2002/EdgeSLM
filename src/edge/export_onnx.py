"""Merge the LoRA adapter into the base model and export ONNX builds.

The adapter is merged first so the exported graph is a single self-contained
model with no PEFT dependency at inference time, which is what an actual edge
deployment would ship. The classifier scores labels from one forward pass, so
the graph is exported without a KV cache.

    python src/edge/export_onnx.py
    python src/edge/export_onnx.py --adapter experiments/smollm2-lora-policy --output experiments/edge-v1

Outputs land under experiments/edge/ (gitignored, they are large):

    merged/                  fp32 PyTorch, adapter folded into the base weights
    onnx-fp32/               ONNX export of the merged model
    onnx-int8-u8s8/          dynamic INT8, signed weights, full range
    onnx-int8-u8u8/          dynamic INT8, unsigned weights
    onnx-int8-weight-only/   8-bit weights (MatMulNBits), fp32 activations
    onnx-int4-weight-only/   4-bit weights (MatMulNBits), fp32 activations
    onnx-int4-int8-compute/  4-bit weights, activations quantised per block of 32

Why both dynamic INT8 layouts are built: u8s8 (unsigned activations, signed
weights) is what optimum's avx512_vnni preset produces and what v1 shipped. On
x86 CPUs without VNNI it can saturate 16-bit intermediate sums, so u8u8 was
added to test whether that explained v1's collapse. It did not (see
results/int8_diagnosis.json): both layouts fail. What they share is that
activations are quantised too, per tensor, at run time. The weight-only builds
never quantise activations, which isolates that.
"""

import argparse
import shutil
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_ADAPTER = Path("experiments/smollm2-v2")
OUTPUT_ROOT = Path("experiments/edge")

VARIANTS = (
    "onnx-int8-u8s8",
    "onnx-int8-u8u8",
    "onnx-int8-weight-only",
    "onnx-int4-weight-only",
    "onnx-int4-int8-compute",
)


def directory_size_mb(path):
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return round(total / 1024**2, 1)


def merge_adapter(adapter, destination):
    import json

    config = json.loads((Path(adapter) / "adapter_config.json").read_text(encoding="utf-8"))
    base_model = config["base_model_name_or_path"]
    print(f"Merging {adapter} into {base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Export runs on CPU, so keep the merge in fp32; fp16 ONNX export of a
    # causal LM to a CPU provider is not reliably supported.
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()

    # SmolLM2 ties the output projection to the token embedding. Exported tied,
    # the graph feeds lm_head a Transpose of the embedding, which no quantiser
    # can touch, so the 180 MB fp32 table survives every quantised build.
    # Separate copies cost 180 MB in the fp32 build and let both be quantised.
    if getattr(merged.config, "tie_word_embeddings", False):
        merged.lm_head.weight = torch.nn.Parameter(merged.get_input_embeddings().weight.detach().clone())
        merged.config.tie_word_embeddings = False

    destination.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    print(f"  merged -> {destination} ({directory_size_mb(destination)} MB)")


def export_onnx(merged_dir, destination):
    from optimum.onnxruntime import ORTModelForCausalLM

    print("Exporting to ONNX (no KV cache) ...")
    started = time.perf_counter()
    model = ORTModelForCausalLM.from_pretrained(merged_dir, export=True, use_cache=False)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    AutoTokenizer.from_pretrained(merged_dir).save_pretrained(destination)
    print(
        f"  onnx -> {destination} ({directory_size_mb(destination)} MB) "
        f"in {time.perf_counter() - started:.1f}s"
    )


def copy_sidecar_files(source, destination):
    for path in source.iterdir():
        if path.is_file() and path.suffix != ".onnx" and not path.name.endswith(".onnx_data"):
            shutil.copy2(path, destination / path.name)


def quantize_dynamic(onnx_dir, destination, unsigned_weights):
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    preset = AutoQuantizationConfig.avx2 if unsigned_weights else AutoQuantizationConfig.avx512_vnni
    config = preset(is_static=False, per_channel=False)
    ORTQuantizer.from_pretrained(onnx_dir, file_name="model.onnx").quantize(
        save_dir=destination, quantization_config=config
    )


def quantize_weight_only(onnx_dir, destination, bits, accuracy_level=None):
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    model = onnx.load(str(onnx_dir / "model.onnx"))
    # The token embedding is a Gather, not a MatMul. Left alone it stays fp32
    # and is about half of the 4-bit artifact, so it is quantised too.
    quantizer = MatMulNBitsQuantizer(
        model,
        bits=bits,
        block_size=32,
        is_symmetric=True,
        accuracy_level=accuracy_level,
        op_types_to_quantize=("MatMul", "Gather"),
        quant_axes=(("MatMul", 0), ("Gather", 1)),
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(destination / "model.onnx"), use_external_data_format=False)


def build_variant(name, onnx_dir, destination):
    print(f"Quantising {name} ...")
    started = time.perf_counter()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    if name == "onnx-int8-u8s8":
        quantize_dynamic(onnx_dir, destination, unsigned_weights=False)
    elif name == "onnx-int8-u8u8":
        quantize_dynamic(onnx_dir, destination, unsigned_weights=True)
    elif name == "onnx-int8-weight-only":
        quantize_weight_only(onnx_dir, destination, bits=8)
    elif name == "onnx-int4-weight-only":
        quantize_weight_only(onnx_dir, destination, bits=4)
    elif name == "onnx-int4-int8-compute":
        # accuracy_level=4 lets MatMulNBits quantise activations to int8 per
        # block of 32 values for the multiply, instead of one scale per tensor.
        quantize_weight_only(onnx_dir, destination, bits=4, accuracy_level=4)
    else:
        raise ValueError(name)

    copy_sidecar_files(onnx_dir, destination)
    print(
        f"  {name} -> {destination} ({directory_size_mb(destination)} MB) "
        f"in {time.perf_counter() - started:.1f}s"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    merged_dir = args.output / "merged"
    onnx_dir = args.output / "onnx-fp32"

    if not args.skip_merge or not merged_dir.exists():
        merge_adapter(args.adapter, merged_dir)
    if not args.skip_export or not onnx_dir.exists():
        export_onnx(merged_dir, onnx_dir)

    for name in args.variants:
        build_variant(name, onnx_dir, args.output / name)

    print("\nArtifact sizes")
    for path in [merged_dir, onnx_dir] + [args.output / name for name in VARIANTS]:
        if path.exists():
            print(f"  {path.name:24s} {directory_size_mb(path):8.1f} MB")


if __name__ == "__main__":
    main()

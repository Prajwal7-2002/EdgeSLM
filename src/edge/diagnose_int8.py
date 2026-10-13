"""Why did v1's INT8 build collapse? Re-quantise the v1 export two ways and score both.

v1 quantised with optimum's avx512_vnni preset: dynamic INT8 with unsigned
activations and signed weights (u8s8), full range. ONNX Runtime documents that
u8s8 can saturate on x86 CPUs without VNNI, because the AVX2 instruction it
uses (vpmaddubsw) adds pairs of products into a 16-bit register. This laptop's
Ryzen 5 6600H has AVX2 but no VNNI.

If saturation is the cause, the same weights quantised as u8u8 (the avx2
preset) should keep most of the fp32 accuracy. Both builds are scored with the
v1 decode path on the same sets as the v1 report.

    python src/edge/export_onnx.py --adapter experiments/smollm2-lora-policy --output experiments/edge-v1 --variants
    python src/edge/diagnose_int8.py
"""

import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge.export_onnx import build_variant
from evaluation.datasets import load_dataset
from evaluation.run_ablation import score
from evaluation.systems import OnnxSystem


V1_EXPORT = Path("experiments/edge-v1")
OUTPUT = Path("results/int8_diagnosis.json")
DATASETS = ("reviewed-heldout", "realistic", "adversarial")


def cpu_flags():
    try:
        import cpuinfo

        flags = set(cpuinfo.get_cpu_info().get("flags", []))
        return {"avx2": "avx2" in flags, "avx512_vnni": "avx512_vnni" in flags, "avx_vnni": "avx_vnni" in flags}
    except ImportError:
        return None


def main():
    onnx_dir = V1_EXPORT / "onnx-fp32"
    if not (onnx_dir / "model.onnx").exists():
        raise SystemExit(f"{onnx_dir} missing; export the v1 adapter first (see module docstring)")

    builds = {"fp32": onnx_dir}
    for name in ("onnx-int8-u8s8", "onnx-int8-u8u8"):
        destination = V1_EXPORT / name
        if not (destination / "model.onnx").exists():
            build_variant(name, onnx_dir, destination)
        builds[name] = destination

    rows = {name: load_dataset(name) for name in DATASETS}
    results = {
        "meta": {
            "adapter": "experiments/smollm2-lora-policy (v1)",
            "decode": "generate (as v1 shipped)",
            "processor": platform.processor(),
            "cpu_flags": cpu_flags(),
        },
        "builds": {},
    }
    for build, directory in builds.items():
        system = OnnxSystem(directory, decode="generate")
        results["builds"][build] = {}
        for name, dataset in rows.items():
            predictions = [system.predict(row["text"])[0] for row in dataset]
            summary = score(dataset, predictions)
            results["builds"][build][name] = {
                key: summary[key]
                for key in ("accuracy", "restricted_recall", "restricted_under_classified", "non_answer_rate")
            }
            print(
                f"{build:16s} {name:18s} acc={summary['accuracy'] * 100:5.1f}%  "
                f"restricted_recall={summary['restricted_recall']}  "
                f"non_answers={summary['non_answer_rate']}",
                flush=True,
            )
        system.close()

    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

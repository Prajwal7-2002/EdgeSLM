# ONNX 4-Bit Edge Quantization 🗜️

To run edge inference on consumer laptops without discrete GPUs, EdgeSLM exports trained LoRA adapters to **ONNX Runtime** and evaluates multiple quantization strategies.

---

## The Edge Quantization Dilemma: INT8 vs. 4-Bit

In early experiments with SmolLM2-360M, standard dynamic INT8 quantization caused catastrophic accuracy degradation on contrastive sets.

### The INT8 Diagnostic
Profiling revealed that dynamic INT8 quantizes **both weights and activations**:
- In small language models ($\le 500\text{M}$ parameters), activation tensors exhibit high dynamic range outliers.
- Quantizing activations to 8-bit integers distorted the logit distributions of single label tokens.
- **Solution**: Switch to **weight-only quantization**. Keeping activations in floating point (FP32/FP16) while compressing weights preserves numerical fidelity.

---

## 4-Bit Quantization on Qwen2.5-0.5B

Using ONNX block-wise 4-bit weight quantization:

| Build Variant | Model Size | RAM Usage | Laptop CPU Latency (Ryzen 5) | Contrast Accuracy (95% Wilson) |
|---|---|---|---|---|
| **PyTorch FP32 Baseline** | 1,890 MB | 2,450 MB | 1,420 ms | 88.8% [80–94] |
| **ONNX FP32** | 1,870 MB | 2,120 MB | 1,180 ms | 88.8% [80–94] |
| **ONNX 4-Bit Weight-Only** | **392 MB** | **1,040 MB** | **520 ms** | **88.8% [80–94]** |

### Statistical Verification: Zero Accuracy Loss
Paired **exact McNemar tests** confirm that the 4-bit ONNX edge build matches full precision FP32 across all evaluation sets ($p \ge 0.5$). 

4-bit compression reduced model size by **79.2%** and cut CPU inference latency by **2.7x** with zero statistical loss in accuracy.

---

## Exporting the Model

To export and quantize your own fine-tuned adapter:

```powershell
uv run python src/edge/export_onnx.py \
  --adapter experiments/qwen0.5b-v2 \
  --output experiments/edge-qwen \
  --variants onnx-int4-weight-only
```

# Statistical Evaluation & McNemar Tests 📐

In small-scale machine learning benchmarks ($N = 40$ to $150$), reporting raw percentage gaps (e.g. "88% vs 81%") without uncertainty bounds is bad science. A 7-point gap on 40 samples is frequently pure random noise.

EdgeSLM reports **95% Wilson Score Confidence Intervals** and **Paired McNemar Significance Tests**.

---

## 1. Wilson Score Intervals

Standard normal approximation intervals ($\hat{p} \pm 1.96 \sqrt{\hat{p}(1-\hat{p})/n}$) fail when accuracy approaches $100\%$ or sample size is small.

EdgeSLM uses Wilson intervals:

$$w = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

---

## 2. Paired McNemar Significance Testing

To verify whether Model A is genuinely better than Model B, EdgeSLM constructs a $2 \times 2$ contingency matrix on identical test items:

| | Model B Correct | Model B Incorrect |
|---|---|---|
| **Model A Correct** | $a$ (Both Right) | $b$ (Only A Right) |
| **Model A Incorrect** | $c$ (Only B Right) | $d$ (Both Wrong) |

The exact two-tailed McNemar test evaluates whether the discordant pairs ($b$ vs $c$) differ from a binomial distribution with $p = 0.5$:

$$\text{Exact } p\text{-value} = 2 \sum_{k=b}^{n} \binom{n}{k} 0.5^n \quad (\text{where } n = b + c)$$

---

## Complete Ablation Benchmark (`contrast` & `reviewed-heldout`)

| System Architecture | `contrast` Accuracy | Pairs (Both Right) | `reviewed-heldout` | RESTRICTED Recall (`contrast`) |
|---|---|---|---|---|
| **TF-IDF + Logistic Regression (v1 data)** | 47.5% [37–58] | 8/40 | 91.7% [82–96] | 50% |
| **SmolLM2-360M LoRA v1** | 36.2% [27–47] | 5/40 | 58.3% [46–70] | 25% |
| **TF-IDF + Logistic Regression (v2 data)** | 75.0% [65–83] | 21/40 | **95.0%** [86–98] | 85% |
| **SmolLM2-360M LoRA v2** | 81.2% [71–88] | 27/40 | 80.0% [68–88] | 85% |
| **Qwen2.5-0.5B LoRA v2** | **88.8%** [80–94] | **31/40** | 90.0% [80–95] | 90% |
| **Qwen2.5-0.5B LoRA v3** | 87.5% [78–93] | 30/40 | **93.3%** [84–97] | 90% |
| **Qwen2.5-0.5B ONNX 4-Bit (Edge)** | **88.8%** [80–94] | **31/40** | 86.7% [76–93] | 95% |

### Key Statistical Conclusions
1. **Beating the Bag-of-Words Baseline**: On the `contrast` set (where vocabulary is identical between pairs), Qwen2.5-0.5B significantly outperforms TF-IDF trained on the same data ($p = 0.043$).
2. **Quantization Invariance**: The 4-bit ONNX edge model yields no significant difference compared to full precision FP32 ($p \ge 0.5$).

# Modern Large Language Model Architectures & Distributed Systems: The Complete Technical Compendium

**A Rigorous Mathematical, Systems-Level, and Architectural Synthesis of Quantization, State Space Models (Mamba), Distributed Parallelism (3D/4D & ZeRO), Inference Economics & PagedAttention, and Novel Transformer Topologies**

---

## Table of Contents

- [1. Foundational Systems Perspective: The Compute & Memory Wall](#1-foundational-systems-perspective-the-compute--memory-wall)
  - [1.1 The Hardware Execution Model: SRAM vs. HBM Hierarchy](#11-the-hardware-execution-model-sram-vs-hbm-hierarchy)
  - [1.2 The Roofline Model: Arithmetic Intensity & Bottlenecks](#12-the-roofline-model-arithmetic-intensity--bottlenecks)
- [2. Module I: Numerical Precision, Arithmetic & Model Quantization](#2-module-i-numerical-precision-arithmetic--model-quantization)
  - [2.1 Data Types & Binary Encodings](#21-data-types--binary-encodings)
  - [2.2 Linear Quantization Mechanics: Symmetric vs. Asymmetric](#22-linear-quantization-mechanics-symmetric-vs-asymmetric)
    - [2.2.1 Symmetric Quantization (Absmax)](#221-symmetric-quantization-absmax)
    - [2.2.2 Asymmetric Quantization (Zero-Point)](#222-asymmetric-quantization-zero-point)
  - [2.3 Outliers, Dynamic Range Clipping & Calibration Strategies](#23-outliers-dynamic-range-clipping--calibration-strategies)
  - [2.4 Weight vs. Activation Quantization (Static vs. Dynamic Calibration)](#24-weight-vs-activation-quantization-static-vs-dynamic-calibration)
  - [2.5 Advanced 4-Bit Post-Training Quantization (PTQ)](#25-advanced-4-bit-post-training-quantization-ptq)
    - [2.5.1 GPTQ (Generalized Post-Training Quantization)](#251-gptq-generalized-post-training-quantization)
    - [2.5.2 GGUF & K-Quants (Block-Wise Hierarchical Quantization)](#252-gguf--k-quants-block-wise-hierarchical-quantization)
  - [2.6 Quantization-Aware Training (QAT) & Loss Landscapes](#26-quantization-aware-training-qat--loss-landscapes)
  - [2.7 Sub-2-Bit and Extreme Low-Bit Paradigms](#27-sub-2-bit-and-extreme-low-bit-paradigms)
    - [2.7.1 BitNet 1-Bit (BitLinear)](#271-bitnet-1-bit-bitlinear)
    - [2.7.2 BitNet 1.58b (Ternary Quantization)](#272-bitnet-158b-ternary-quantization)
  - [2.8 Quantization Comparison Matrix](#28-quantization-comparison-matrix)
- [3. Module II: Sequence Modeling Paradigms — Transformers, RNNs & State Space Models (Mamba)](#3-module-ii-sequence-modeling-paradigms--transformers-rnns--state-space-models-mamba)
  - [3.1 The Sequence Modeling Trilemma](#31-the-sequence-modeling-trilemma)
  - [3.2 Continuous-Time State Space Models (SSMs)](#32-continuous-time-state-space-models-ssms)
  - [3.3 Discretization via Zero-Order Hold (ZOH)](#33-discretization-via-zero-order-hold-zoh)
  - [3.4 The Dual Representation of Linear Time-Invariant (LTI) SSMs](#34-the-dual-representation-of-linear-time-invariant-lti-ssms)
    - [3.4.1 Recurrent Representation (Fast Linear Inference)](#341-recurrent-representation-fast-linear-inference)
    - [3.4.2 Convolutional Representation (Parallel Training via FFT)](#342-convolutional-representation-parallel-training-via-fft)
  - [3.5 Long-Range Memory & The HiPPO Matrix](#35-long-range-memory--the-hippo-matrix)
  - [3.6 The Fundamental Failure Mode of LTI SSMs: Content Invariance](#36-the-fundamental-failure-mode-of-lti-ssms-content-invariance)
  - [3.7 Mamba: Selective State Spaces (S6)](#37-mamba-selective-state-spaces-s6)
    - [3.7.1 Mathematical Formulation of Selective Parameterization](#371-mathematical-formulation-of-selective-parameterization)
    - [3.7.2 Physical Intuition of Step-Size Delta](#372-physical-intuition-of-step-size-delta)
  - [3.8 Systems Innovations for Selective SSMs](#38-systems-innovations-for-selective-ssms)
    - [3.8.1 The Parallel Associative Scan](#381-the-parallel-associative-scan)
    - [3.8.2 Hardware-Aware Memory Hierarchy & Kernel Fusion](#382-hardware-aware-memory-hierarchy--kernel-fusion)
    - [3.8.3 Activation Recomputation in the Backward Pass](#383-activation-recomputation-in-the-backward-pass)
  - [3.9 The Mamba Block Architecture vs. Transformer Decoder](#39-the-mamba-block-architecture-vs-transformer-decoder)
- [4. Module III: Distributed Training, Memory Layouts & Systems Parallelism](#4-module-iii-distributed-training-memory-layouts--systems-parallelism)
  - [4.1 Anatomy of GPU Memory in Large Language Models](#41-anatomy-of-gpu-memory-in-large-language-models)
    - [4.1.1 The 16 Bytes per Parameter Rule in Mixed-Precision Adam](#411-the-16-bytes-per-parameter-rule-in-mixed-precision-adam)
    - [4.1.2 The Mathematical Necessity of FP32 Master Weights & Optimizer States](#412-the-mathematical-necessity-of-fp32-master-weights--optimizer-states)
  - [4.2 Data Parallelism (DDP) & Collective All-Reduce](#42-data-parallelism-ddp--collective-all-reduce)
  - [4.3 Tensor Parallelism (Megatron-LM Intra-Layer Slicing)](#43-tensor-parallelism-megatron-lm-intra-layer-slicing)
    - [4.3.1 Slicing the Multi-Layer Perceptron (MLP) Block](#431-slicing-the-multi-layer-perceptron-mlp-block)
    - [4.3.2 Slicing Multi-Head Self-Attention (MHSA)](#432-slicing-multi-head-self-attention-mhsa)
  - [4.4 Pipeline Parallelism (PP) & Inter-Layer Scheduling](#44-pipeline-parallelism-pp--inter-layer-scheduling)
    - [4.4.1 The Pipeline Bubble & Mathematical Formulation](#441-the-pipeline-bubble--mathematical-formulation)
    - [4.4.2 Mathematical Equivalence of Micro-Batch Gradient Accumulation](#442-mathematical-equivalence-of-micro-batch-gradient-accumulation)
    - [4.4.3 GPipe vs. 1F1B Schedule Activation Footprints](#443-gpipe-vs-1f1b-schedule-activation-footprints)
  - [4.5 Context Parallelism (CP) & Ring Attention](#45-context-parallelism-cp--ring-attention)
    - [4.5.1 The Ring Communication Topology](#451-the-ring-communication-topology)
    - [4.5.2 Online Numerically Stable Softmax Accumulation](#452-online-numerically-stable-softmax-accumulation)
  - [4.6 ZeRO (Zero Redundancy Optimizer) & FSDP Mechanics](#46-zero-zero-redundancy-optimizer--fsdp-mechanics)
    - [4.6.1 ZeRO-Stage 1: Optimizer State Sharding](#461-zero-stage-1-optimizer-state-sharding)
    - [4.6.2 ZeRO-Stage 2: Gradient Sharding](#462-zero-stage-2-gradient-sharding)
    - [4.6.3 ZeRO-Stage 3: Parameter Sharding (Fully Sharded Data Parallel)](#463-zero-stage-3-parameter-sharding-fully-sharded-data-parallel)
    - [4.6.4 Communication vs. Memory Trade-Off Across Stages](#464-communication-vs-memory-trade-off-across-stages)
  - [4.7 Operator & Kernel Fusion (FlashAttention Deep Dive)](#47-operator--kernel-fusion-flashattention-deep-dive)
  - [4.8 Unified Distributed Parallelism Matrix](#48-unified-distributed-parallelism-matrix)
- [5. Module IV: LLM Inference Economics, Memory Architecture & PagedAttention](#5-module-iv-llm-inference-economics-memory-architecture--pagedattention)
  - [5.1 The Economic Calculus of LLM Serving: TCO & Unit Cost per Token](#51-the-economic-calculus-of-llm-serving-tco--unit-cost-per-token)
  - [5.2 The Two Execution Phases: Prefill vs. Decoding](#52-the-two-execution-phases-prefill-vs-decoding)
  - [5.3 First-Principles FLOP & Parameter Counting (Case Study: LLaMA 3.3 70B)](#53-first-principles-flop--parameter-counting-case-study-llama-33-70b)
    - [5.3.1 Layer-by-Layer FLOP Derivations](#531-layer-by-layer-flop-derivations)
    - [5.3.2 Total Parameter Calculation](#532-total-parameter-calculation)
  - [5.4 The KV Cache: Eliminating Redundant Attention Computations](#54-the-kv-cache-eliminating-redundant-attention-computations)
    - [5.4.1 Mathematical Memory Footprint Formulation](#541-mathematical-memory-footprint-formulation)
    - [5.4.2 The KV Cache Memory Explosion at Scale](#542-the-kv-cache-memory-explosion-at-scale)
  - [5.5 Batching Economics & The Law of Diminishing Latency](#55-batching-economics--the-law-of-diminishing-latency)
    - [5.5.1 Amortizing Memory Bandwidth Across Batch Dimension B](#551-amortizing-memory-bandwidth-across-batch-dimension-b)
    - [5.5.2 Throughput vs. Per-User Latency (TPOT) Trade-Off](#552-throughput-vs-per-user-latency-tpot-trade-off)
    - [5.5.3 Model Bandwidth Utilization (MBU) vs. Model FLOPs Utilization (MFU)](#553-model-bandwidth-utilization-mbu-vs-model-flops-utilization-mfu)
  - [5.6 PagedAttention: OS-Inspired Virtual Memory for KV Cache (vLLM)](#56-pagedattention-os-inspired-virtual-memory-for-kv-cache-vllm)
    - [5.6.1 The Memory Fragmentation Crisis](#561-the-memory-fragmentation-crisis)
    - [5.6.2 Block Tables & Virtual Paging Architecture](#562-block-tables--virtual-paging-architecture)
    - [5.6.3 Copy-on-Write (CoW) Forking & Parallel Sampling](#563-copy-on-write-cow-forking--parallel-sampling)
    - [5.6.4 Prefix Caching (Prompt Caching)](#564-prefix-caching-prompt-caching)
  - [5.7 Continuous (Iteration-Level) Batching vs. Static Batching](#57-continuous-iteration-level-batching-vs-static-batching)
- [6. Module V: Novel Transformer Topologies — Per-Layer Embeddings (PLE) in Gemma 4](#6-module-v-novel-transformer-topologies--per-layer-embeddings-ple-in-gemma-4)
  - [6.1 Motivation: Decoupling Capacity from FLOPs & Eliminating Identity Dilution](#61-motivation-decoupling-capacity-from-flops--eliminating-identity-dilution)
  - [6.2 Mathematical Formulation & Forward Pass](#62-mathematical-formulation--forward-pass)
    - [6.2.1 Pre-Layer Representation](#621-pre-layer-representation)
    - [6.2.2 State-Dependent Gating & Integration](#622-state-dependent-gating--integration)
  - [6.3 Deep Dive: Why the Hadamard Product Is Structurally Mandatory](#63-deep-dive-why-the-hadamard-product-is-structurally-mandatory)
  - [6.4 Information Retention & Semantic Disambiguation Dynamics](#64-information-retention--semantic-disambiguation-dynamics)
  - [6.5 Structural Comparison: Standard Transformer vs. Gemma 4 PLE](#65-structural-comparison-standard-transformer-vs-gemma-4-ple)
- [7. Module VI: Cross-Disciplinary Synthesis & Engineering Playbook](#7-module-vi-cross-disciplinary-synthesis--engineering-playbook)
  - [7.1 Full Model Lifecycle Workflow](#71-full-model-lifecycle-workflow)
  - [7.2 Systems Engineering Decision Flowchart](#72-systems-engineering-decision-flowchart)
  - [7.3 Comprehensive Glossary of Symbols & Notation](#73-comprehensive-glossary-of-symbols--notation)

---

## 1. Foundational Systems Perspective: The Compute & Memory Wall

Modern machine learning systems are fundamentally constrained by computer architecture. To master deep learning at scale, one must analyze algorithmic complexity not only through asymptotic FLOPs ($O(N)$), but through memory bandwidth, cache hierarchy utilization, and inter-accelerator interconnects.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             NVIDIA GPU HIERARCHY                            │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Streaming Multiprocessor (SM)                                         │  │
│  │  ┌────────────────────────┐  ┌─────────────────────────────────────┐  │  │
│  │  │ Registers (~64 KB / SM)│  │ SRAM / L1 Data Cache (~128-256 KB)  │  │  │
│  │  │ Bandwidth: ~20-30 TB/s │  │ Bandwidth: ~15-19 TB/s              │  │  │
│  │  └────────────────────────┘  └─────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │ Tensor Cores (Dense Matrix Multiply Units: FP16/BF16/FP8/INT8)   │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ L2 Cache (Shared on-chip, ~40-60 MB) — Bandwidth: ~5-7 TB/s           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ High Bandwidth Memory (HBM3 / DRAM, 80-141 GB)                        │  │
│  │ Bandwidth: ~2.0 - 4.8 TB/s                                            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼ Interconnect                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ NVLink / NVSwitch: ~900 GB/s bidirectional per GPU                    │  │
│  │ PCIe Gen5: ~64 GB/s | InfiniBand NDR: ~50 GB/s (400 Gbps)             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 The Hardware Execution Model: SRAM vs. HBM Hierarchy

1. **Registers & SRAM (L1 Cache):** Ultra-fast, on-chip storage directly adjacent to execution units. Has massive bandwidth ($>15\text{ TB/s}$) but strictly limited capacity ($<256\text{ KB}$ per Streaming Multiprocessor).
2. **High Bandwidth Memory (HBM / DRAM):** Off-chip bulk memory ($80\text{ GB}-141\text{ GB}$). High capacity, but bandwidth is orders of magnitude slower than SRAM ($2-4.8\text{ TB/s}$).
3. **Interconnects (NVLink vs. Ethernet/InfiniBand):** Intra-node NVLink provides massive bandwidth ($900\text{ GB/s}$), while inter-node networks (InfiniBand/RoCE) drop down to $25-50\text{ GB/s}$.

### 1.2 The Roofline Model: Arithmetic Intensity & Bottlenecks

The performance of an operational kernel is governed by its **Arithmetic Intensity** ($I$), defined as:

$$
I = \frac{\text{Floating Point Operations (FLOPs)}}{\text{Memory Access (Bytes transferred from HBM)}}
$$

$$
\text{Attainable Performance } P = \min\left(P_{\text{peak}}, \; I \times \text{Bandwidth}_{\text{HBM}}\right)
$$

```text
Performance (TFLOPs)
   ^
   |                      /---------------- Peak Compute Limit (P_peak)
   |                     /
   |                    /
   |                   /  Compute-Bound Regime (e.g., Prefill Phase, Large GEMMs)
   |                  /
   |                 /
   |                / Memory-Bound Regime (e.g., Autoregressive Decoding, Softmax, RMSNorm)
   |               /
   |              /
   |             /
   |------------/-------------------------------------------------------->
   0           I_knee                               Arithmetic Intensity (FLOPs/Byte)
```

- **Memory-Bound Operations ($I < I_{\text{knee}}$):** Normalization layers (RMSNorm, LayerNorm), element-wise activations (GeLU, SiLU), Point-wise additions, Softmax, Autoregressive decoding with batch size $1$. The GPU cores sit idle waiting for memory loads from HBM.
- **Compute-Bound Operations ($I > I_{\text{knee}}$):** Large Matrix Multiplications ($Y = XW$) in Linear and Attention projection layers during the prompt prefill phase or decoding with large batched requests ($B \gg 1$).

---

## 2. Module I: Numerical Precision, Arithmetic & Model Quantization

Model quantization compresses high-precision tensors (typically 32-bit or 16-bit floating point) into low-bit representations (8-bit, 4-bit, 2-bit, or ternary), drastically reducing memory footprint and memory bandwidth pressure while accelerating compute on specialized tensor hardware.

### 2.1 Data Types & Binary Encodings

Under the IEEE-754 standard and modern deep learning formats, numerical values are partitioned into three bitfields: Sign ($s$), Exponent ($e$), and Mantissa/Fraction ($m$).

$$
\text{Value} = (-1)^s \times 2^{e - \text{bias}} \times \left(1 + \sum_{i=1}^{M} m_i 2^{-i}\right)
$$

```text
FP32 (Single Precision, 32-bit):
 1   8               23
┌─┬────────┬───────────────────────────────┐
│s│exponent│           mantissa            │  Range: ~1.4e-45 to ~3.4e38
└─┴────────┴───────────────────────────────┘

FP16 (Half Precision, 16-bit):
 1   5         10
┌─┬──────┬──────────┐
│s│ exp  │ mantissa │                         Range: ~5.96e-8 to 65,504
└─┴──────┴──────────┘

BF16 (Brain Floating Point, 16-bit):
 1   8        7
┌─┬────────┬───────┐
│s│exponent│mantiss│                         Range: Matches FP32 (~3.4e38), lower precision
└─┴────────┴───────┘

FP8 (E4M3 - High Precision for Weights/Activations, 8-bit):
 1   4     3
┌─┬────┬───────┐
│s│exp │mantiss│                             Range: [-448, 448], higher precision
└─┴────┴───────┘

FP8 (E5M2 - Large Dynamic Range for Gradients, 8-bit):
 1   5    2
┌─┬─────┬──┐
│s│ exp │man│                                Range: Matches FP16 dynamic range (~57,344)
└─┴─────┴──┘

INT8 (Signed Integer, 8-bit):
 1         7
┌─┬────────────────┐
│s│ magnitude      │                         Range: [-128, 127], uniform step size
└─┴────────────────┘
```

| Format | Total Bits | Exponent Bits | Mantissa Bits | Dynamic Range | Relative Precision (Epsilon) | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** | 32 | 8 | 23 | $\approx 10^{\pm 38}$ | $2^{-23} \approx 1.19 \times 10^{-7}$ | Master weights, optimizer states |
| **FP16** | 16 | 5 | 10 | $\approx [-6.5 \times 10^4, 6.5 \times 10^4]$ | $2^{-10} \approx 9.77 \times 10^{-4}$ | Legacy mixed-precision forward/backward |
| **BF16** | 16 | 8 | 7 | $\approx 10^{\pm 38}$ | $2^{-7} \approx 7.81 \times 10^{-3}$ | Modern LLM standard training |
| **FP8 (E4M3)** | 8 | 4 | 3 | $\approx [-448, 448]$ | $2^{-3} = 0.125$ | Forward pass weights/activations |
| **FP8 (E5M2)** | 8 | 5 | 2 | $\approx [-5.7 \times 10^4, 5.7 \times 10^4]$ | $2^{-2} = 0.25$ | Backward pass gradients |
| **INT8** | 8 | 0 | 7 | $[-128, 127]$ | Constant step size $\Delta = 1$ | Fast integer inference, activations |
| **INT4** | 4 | 0 | 3 | $[-8, 7]$ | Constant step size $\Delta = 1$ | PTQ weight storage (GPTQ, AWQ) |
| **Ternary** | 1.58 | 0 | 0 | $\{-1, 0, +1\}$ | Discrete ternary state | BitNet 1.58b addition-only inference |

---

### 2.2 Linear Quantization Mechanics: Symmetric vs. Asymmetric

Linear quantization maps a continuous, real-valued interval $[\alpha, \beta] \subset \mathbb{R}$ to a discrete grid of integer values $[q_{\min}, q_{\max}] \subset \mathbb{Z}$.

```text
Continuous FP32 Space:
  β (Min) ─────────────────────── 0 ───────────────────────────── α (Max)
     │                            │                                  │
     │ Linear Scale Factor s      │ Zero-Point Shift z               │
     ▼                            ▼                                  ▼
Quantized INT8 Grid:
   -128 ───────────────────────── z ─────────────────────────────── +127
```

#### 2.2.1 Symmetric Quantization (Absmax)

Symmetric quantization forces the floating-point zero to map exactly to the integer zero ($z = 0$). The dynamic range is symmetric around zero: $[-\alpha, \alpha]$ where $\alpha = \max(|x|)$.

1. **Scale Factor Calculation:**
   $$
   s = \frac{2^{b-1} - 1}{\max(|x|)} = \frac{q_{\max}}{\alpha}
   $$

2. **Quantization Formula (Floating Point $\to$ Integer):**
   $$
   x_q = \text{clamp}\left(\lfloor \text{round}(s \cdot x) \rceil, \; -q_{\max}, \; q_{\max}\right)
   $$

3. **Dequantization Formula (Integer $\to$ Floating Point):**
   $$
   \hat{x} = \frac{x_q}{s}
   $$

4. **Quantization Error:**
   $$
   \epsilon = |x - \hat{x}| \le \frac{1}{2s}
   $$

#### 2.2.2 Asymmetric Quantization (Zero-Point)

When data distributions are strictly skewed (e.g., post-ReLU or post-GeLU activations where $x \ge 0$), symmetric quantization wastes half the integer representation domain. Asymmetric quantization maps $[\min(x), \max(x)] \to [q_{\min}, q_{\max}]$ by introducing an integer **Zero-Point** $z$.

1. **Scale Factor Calculation:**
   $$
   s = \frac{q_{\max} - q_{\min}}{\max(x) - \min(x)} = \frac{2^b - 1}{\alpha - \beta}
   $$

2. **Zero-Point Calculation:**
   $$
   z = \text{round}\left(- \beta \cdot s\right) + q_{\min} = \text{clamp}\left(\left\lfloor \text{round}\left(- \min(x) \cdot s + q_{\min}\right) \right\rceil, \; q_{\min}, \; q_{\max}\right)
   $$

3. **Quantization Mapping:**
   $$
   x_q = \text{clamp}\left(\lfloor \text{round}(s \cdot x) \rceil + z, \; q_{\min}, \; q_{\max}\right)
   $$

4. **Dequantization Mapping:**
   $$
   \hat{x} = \frac{x_q - z}{s}
   $$

```text
Symmetric vs. Asymmetric Numerical Walkthrough (FP32 -> INT8):
Input Array: x = [-6.0, -2.0, 0.0, 1.5, 4.0]

[Symmetric Absmax]:
  alpha = max(|x|) = 6.0
  s = 127 / 6.0 = 21.1667
  x_q = round(x * s) = [-127, -42, 0, 32, 85]
  x_hat = x_q / s = [-6.000, -1.984, 0.000, 1.512, 4.016]

[Asymmetric Zero-Point]:
  beta = -6.0, alpha = 4.0, range = 10.0
  s = (127 - (-128)) / (4.0 - (-6.0)) = 255 / 10.0 = 25.5
  z = round(-(-6.0) * 25.5) + (-128) = round(153) - 128 = 25
  x_q = round(x * s) + z = [-153+25, -51+25, 0+25, 38+25, 102+25] = [-128, -26, 25, 63, 127]
  x_hat = (x_q - z) / s = [-6.000, -2.000, 0.000, 1.490, 4.000]
```

---

### 2.3 Outliers, Dynamic Range Clipping & Calibration Strategies

In LLMs (notably at scales $>6.7\text{B}$ parameters), activation distributions exhibit emergent **systematic channel outliers**—isolated features with magnitudes $20-100\times$ larger than normal activations. If the scale factor $s$ is computed strictly from the maximum value $\alpha = \max(|x|)$, the non-outlier values will be squeezed into a tiny band of integer buckets, destroying representation capacity.

```text
Without Clipping (Full Range Mapping):
  Outlier Value: 120.0, Normal Values: [-2.0, 1.0, 0.5, -0.8]
  Scale s = 127 / 120 = 1.058
  Quantized Normal Values: [-2, 1, 1, -1]  <-- Extreme precision loss / collapse!

With Calibration / Clipping at Dynamic Range [-5.0, 5.0]:
  Outlier Value 120.0 clipped to +5.0 -> Quantized to +127 (High Outlier Error)
  Scale s = 127 / 5.0 = 25.4
  Quantized Normal Values: [-51, 25, 13, -20] <-- High granularity preserved!
```

```text
                        CALIBRATION STRATEGIES
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
  Percentile Clipping        Mean Squared Error        Entropy Minimization
  (e.g., 99.99th percentile) (MSE Optimization)        (KL-Divergence / TensorRT)
  - Drops top 0.01%         - Minimizes ||W - W_hat|| - Minimizes information loss
  - Simple, fast            - Optimal for weights     - Optimal for activations
```

1. **Mean Squared Error (MSE) Optimization:**
   $$
   \alpha^* = \arg\min_{\alpha} \sum_{i} \left( x_i - \text{Dequant}(\text{Quant}(x_i; \alpha)) \right)^2
   $$
2. **Kullback-Leibler (KL) Divergence Calibration:**
   Measures the relative entropy between the continuous probability distribution $P$ of the original tensor and the discretized distribution $Q$ constructed by expanding the quantized histogram back to the original bucket count:
   $$
   D_{\text{KL}}(P \parallel Q) = \sum_{k=1}^{N} P(k) \log\left(\frac{P(k)}{Q(k)}\right)
   $$

---

### 2.4 Weight vs. Activation Quantization (Static vs. Dynamic Calibration)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DYNAMIC VS. STATIC ACTIVATION QUANTIZATION              │
│                                                                             │
│  [Dynamic Quantization]                                                     │
│   Input X ──► [Compute s, z on-the-fly] ──► [Quantize X_q] ──► [MatMul]     │
│   - Scale computed at runtime per token/layer.                              │
│   - Higher accuracy, but adds kernel launch & reduction overhead.           │
│                                                                             │
│  [Static Quantization]                                                      │
│   Calibration Data ──► [Offline Profiling] ──► [Precomputed s, z in Graph]  │
│   Input X ──► [Quantize with Precomputed s, z] ──► [MatMul]                │
│   - Scale is frozen; zero runtime calibration latency.                      │
│   - Lower compute overhead, but sensitive to out-of-distribution inputs.     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 2.5 Advanced 4-Bit Post-Training Quantization (PTQ)

Going below 8 bits (e.g., INT4) using naive linear rounding causes catastrophic perplexity degradation. Modern 4-bit PTQ algorithms solve this through second-order optimization and hierarchical block-wise scaling.

#### 2.5.1 GPTQ (Generalized Post-Training Quantization)

GPTQ quantizes weight matrices column-by-column while continuously compensating unquantized weights for the introduced quantization error, utilizing a second-order Taylor expansion of the layer output error.

1. **Objective Function:**
   $$
   \arg\min_{\widehat{W}} \| W X - \widehat{W} X \|_2^2
   $$

2. **Inverse Hessian Formulation:**
   The Hessian of the squared error with respect to the weights is $H = 2 X X^T$. Let $H^{-1}$ denote its inverse. GPTQ precomputes $H^{-1}$ via Cholesky decomposition:
   $$
   H^{-1} = L \cdot L^T
   $$

3. **Step-by-Step Weight Update & Error Redistribution:**
   When the $q$-th column of the weight matrix $W_{:, q}$ is quantized to $\widehat{W}_{:, q} = \text{Quant}(W_{:, q})$, the error vector is:
   $$
   E_q = W_{:, q} - \widehat{W}_{:, q}
   $$
   This error is compensated across all remaining unquantized columns $j > q$ using:
   $$
   W_{:, j} \leftarrow W_{:, j} - \frac{E_q \cdot [H^{-1}]_{q, j}}{[H^{-1}]_{q, q}}
   $$

```text
GPTQ Execution Trace Across Weight Matrix Columns:
Weight Matrix W: [w_1 | w_2 | w_3 | ... | w_N]
Hessian Inverse: H^-1 = (2 X X^T + lambda*I)^-1

Step 1: Quantize column 1 -> w_hat_1 = Quant(w_1)
        Quantization Error: e_1 = w_1 - w_hat_1
Step 2: Update all remaining columns:
        w_2 <- w_2 - e_1 * (H^-1)_{1,2} / (H^-1)_{1,1}
        w_3 <- w_3 - e_1 * (H^-1)_{1,3} / (H^-1)_{1,1}
Step 3: Quantize updated column 2 -> w_hat_2 = Quant(w_2)
        Redistribute e_2 to columns 3...N via H^-1
... Repeat until all N columns are quantized.
```

> [!NOTE]
> **GPTQ Engineering Optimizations:**
> 1. **Damping Factor:** Adds $\lambda \cdot I$ ($\lambda \approx 1\%\text{ of avg diagonal}$) to ensure $H$ is positive-definite and numerically invertible.
> 2. **Lazy Batching:** Quantizes blocks of columns ($128$ columns at a time) using local updates, followed by a single fused GEMM to update the remaining matrix.
> 3. **Cholesky Precomputation:** Avoids recomputing $H^{-1}$ dynamically at each step.

#### 2.5.2 GGUF & K-Quants (Block-Wise Hierarchical Quantization)

GGUF (used heavily in `llama.cpp`) utilizes **Hierarchical Block Quantization (K-quants)**. Weights are grouped into **super-blocks** (e.g., 256 weights), which are subdivided into **sub-blocks** (e.g., 16 or 32 weights).

```text
Super-Block (256 weights)
┌───────────────────────────────────────────────────────────────────────┐
│ Super-Scale S (8-bit)  |  Super-Min M (8-bit)                         │
├───────────────────────────────────────────────────────────────────────┤
│ Sub-Block 0 (32 weights)  ──► Scale s_0 (quantized using S)           │
│ Sub-Block 1 (32 weights)  ──► Scale s_1 (quantized using S)           │
│ ...                                                                   │
│ Sub-Block 7 (32 weights)  ──► Scale s_7 (quantized using S)           │
└───────────────────────────────────────────────────────────────────────┘
```

- **Dual-Level Compression:** Sub-block scales $s_i$ are themselves quantized using the super-block scale $S$.
- **Unified CPU/GPU Hybrid Offloading:** Allows splitting layers between CPU RAM and GPU VRAM with customized memory-mapped I/O (`mmap`).

---

### 2.6 Quantization-Aware Training (QAT) & Loss Landscapes

Post-Training Quantization fits rounding to a model trained in continuous space. In contrast, Quantization-Aware Training (QAT) models the discretization error directly during gradient descent.

```text
                  QAT FORWARD & BACKWARD PASS
                  
Forward Pass:
  FP32 Master Weight W ──► [ Fake Quantization ] ──► W_quant (INT4/INT8)
                                    │
                                    ▼
                         MatMul with Activations
                                    │
                                    ▼
                               Compute Loss L
                                    │
Backward Pass (STE):                ▼
  FP32 Master Weight W ◄── [ dL/dW = dL/dW_quant ] ◄── Backpropagation
```

1. **Straight-Through Estimator (STE):**
   The derivative of the rounding function $\lfloor x \rceil$ is zero almost everywhere and undefined at integers. STE substitutes the rounding derivative with the identity function during backpropagation:
   $$
   \text{Forward: } x_q = \text{round}(x), \qquad \text{Backward: } \frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial x_q} \cdot \mathbf{1}_{|x| \le \text{clip}}
   $$

2. **Loss Landscape Minima Geometry:**
   - **PTQ:** Frequently converges to sharp, narrow local minima in FP32 space. When rounded, the parameters jump outside the basin, creating massive loss spikes.
   - **QAT:** Forces optimizer trajectories toward wide, flat minima. Even when perturbed by discrete rounding, the loss remains near-optimal.

```text
Loss
 ^              Narrow Minimum (PTQ Trajectory)      Wide Minimum (QAT Trajectory)
 |
 |                     /\                                  \                /
 |                    /  \   <- Large loss jump             \              /
 |                   / \/ \     after rounding               \   Rounded  /
 |                  /      \                                  \---[x]----/
 |                 /   x    \                                      ^
 |                / Rounded  \                                     | Minimal loss
 |               /     |      \                                    | perturbation
 |              /      v       \                                   |
 └─────────────/──────[x]───────\──────────────────────────────────v──────────────>
                                                                Weight Parameter w
```

---

### 2.7 Sub-2-Bit and Extreme Low-Bit Paradigms

#### 2.7.1 BitNet 1-Bit (BitLinear)

BitNet replaces standard `nn.Linear` layers with `BitLinear`, constraining weights to 1-bit binary values $\{-1, +1\}$ while maintaining INT8 activations.

```text
BitNet 1-Bit Weight Quantization Pipeline:
Input Weights W ──► [ Center around Mean: W - E[W] ] ──► [ Signum Function: sgn(W) ] ──► W_bin in {-1, +1}
                                                                   │
                                      Track Beta = 1/N * sum(|W|)  │
                                                                   ▼
Act X (FP16) ──► [ Absmax Quant to INT8 ] ──► X_quant ──► [ MatMul: X_quant * W_bin ] ──► Dequant with (alpha * beta)
```

1. **Weight Binarization (Signum):**
   $$
   W_{\text{bin}} = \text{Sign}(W - \bar{W}), \qquad \text{where } \text{Sign}(x) = \begin{cases} +1, & x \ge 0 \\ -1, & x < 0 \end{cases}
   $$
   $$
   \beta = \frac{1}{nm} \|W\|_1 = \frac{1}{nm}\sum_{i,j} |W_{i,j}|
   $$
2. **Activation Quantization (Absmax):**
   $$
   X_{\text{quant}} = \text{clamp}\left(\lfloor \text{round}\left( \frac{127}{\alpha} X \right) \rceil, \; -128, \; 127\right), \qquad \alpha = \|X\|_{\infty} = \max(|X|)
   $$
3. **Dequantization of Matrix Multiplication:**
   $$
   Y = \left( X_{\text{quant}} \times W_{\text{bin}} \right) \times \frac{\alpha \cdot \beta}{127}
   $$

#### 2.7.2 BitNet 1.58b (Ternary Quantization)

BitNet 1.58b introduces zero ($0$) into the weight representation, yielding a ternary alphabet $\{-1, 0, +1\}$. Because $\log_2(3) \approx 1.585\text{ bits}$, this architecture represents every parameter in just 1.58 bits.

```text
Matrix Multiplication Transformation:
Standard MatMul:  y_i = sum_j ( W_ij * x_j )    --> Requires Floating Point Multiply + Accumulate (MAC)
BitNet 1.58b:     W_ij in {-1, 0, +1}
                  If W_ij = +1:  Add x_j
                  If W_ij = -1:  Subtract x_j
                  If W_ij =  0:  Ignore x_j (Skip)
                  --> ZERO Floating Point Multiplications! Addition/Subtraction ONLY.
```

1. **Weight Quantization (`absmean`):**
   $$
   W_{\text{ternary}} = \text{clamp}\left( \left\lfloor \text{round}\left( \frac{W}{\gamma + \epsilon} \right) \right\rceil, \; -1, \; +1 \right), \qquad \gamma = \frac{1}{nm} \sum_{i,j} |W_{i,j}|
   $$
2. **Activation Quantization:**
   $$
   X_{\text{quant}} = \text{clamp}\left( \left\lfloor \text{round}\left( \frac{X \cdot 127}{\|X\|_{\infty}} \right) \right\rceil, \; -128, \; 127 \right)
   $$
3. **Structural Advantages of the Zero State:**
   - **Feature Filtering / Sparsity:** Setting a weight to $0$ explicitly mutes irrelevant context.
   - **Energy Efficiency:** Eliminates hardware multipliers entirely, reducing dynamic matrix multiplication energy by up to $10-20\times$ on silicon.

---

### 2.8 Quantization Comparison Matrix

| Method | Weight Bits | Activation Bits | Multiplier Operations | Accuracy Retention ($>7\text{B}$) | Primary Bottleneck | Key Application |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP16 / BF16** | 16 | 16 | FP16 Tensor MAC | 100% (Baseline) | Memory capacity & memory bandwidth | Pre-training, high-precision serving |
| **FP8 (E4M3)** | 8 | 8 | FP8 Tensor MAC | $>99.5\%$ | Dynamic range calibration | High-throughput H100 pre-training/serving |
| **INT8 (PTQ)** | 8 | 8 | INT8 Tensor MAC | $>99.0\%$ | Activation outlier preservation | Enterprise server inference |
| **GPTQ (PTQ)** | 4 | 16 | FP16/INT4 Dequant GEMM | $>98.0\%$ | Calibration dataset alignment | GPU-only memory constrained serving |
| **GGUF (K-Quants)** | 2-6 | 16 | Dequantized GEMV/GEMM | $>95.0-99\%$ | CPU compute bandwidth | Consumer hardware, CPU/GPU hybrid offload |
| **BitNet 1.0** | 1 | 8 | INT8 Add/Sub | Medium ($<30\text{B}$ degradation) | High-entropy capacity limits | Ultra-low power edge devices |
| **BitNet 1.58b** | 1.58 | 8 | INT8 Addition Only | Parity with FP16 at scale | Specialized ternary ASIC availability | Next-gen energy-efficient LLM inference |

---

## 3. Module II: Sequence Modeling Paradigms — Transformers, RNNs & State Space Models (Mamba)

### 3.1 The Sequence Modeling Trilemma

Sequence models in machine learning must balance three competing properties:
1. **$O(1)$ Step-Wise Inference Cost:** Generating token $t+1$ requires constant compute and memory, independent of context length $L$.
2. **Parallelizable Training:** Processing training sequences of length $L$ scales concurrently as $O(1)$ or $O(L \log L)$ depth on GPUs.
3. **Uncompressed Contextual Recall:** The architecture can model complex interactions across long contexts without catastrophic forgetting.

```text
                      THE SEQUENCE MODELING TRILEMMA
                                    
                                [Transformers]
                                 /          \
                     Parallel Training    Uncompressed Recall
                               /              \
                              /   [Mamba / S6] \
                             /    (Best of All) \
                            /                    \
              [Linear Attention / S4]         [Recurrent NNs]
                           \                     /
                       O(1) Step-Wise Inference
```

| Architecture | Training Parallelism | Inference Step Complexity ($L$-th Token) | Context Footprint During Generation | Long-Context Modeling Fidelity |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Transformer** | $O(1)$ Depth (Parallel via Attention) | $O(L)$ Compute, $O(L)$ Memory Reads | $O(L)$ KV Cache (Grows linearly) | Exceptional (Attention over raw history) |
| **Classic RNN / LSTM** | $O(L)$ Sequential (Cannot parallelize) | $O(1)$ Compute, $O(1)$ Memory Reads | $O(1)$ Hidden Vector $h_t$ | Poor (Information bottleneck / forgetting) |
| **LTI SSMs (S4 / LSSL)** | $O(L \log L)$ (Parallel via FFT Conv) | $O(1)$ Compute, $O(1)$ Memory Reads | $O(1)$ Hidden Vector $h_t$ | Moderate (Fails dynamic selection) |
| **Selective SSM (Mamba)** | $O(L)$ (Parallel via Associative Scan) | $O(1)$ Compute, $O(1)$ Memory Reads | $O(1)$ Hidden Vector $h_t$ | Exceptional (Matches Transformers at scale) |

---

### 3.2 Continuous-Time State Space Models (SSMs)

Originating in classical control theory, continuous-time state space models map a 1-dimensional continuous input signal $x(t) \in \mathbb{R}$ to an output signal $y(t) \in \mathbb{R}$ through an $N$-dimensional latent state variable $h(t) \in \mathbb{R}^N$.

$$
\frac{d h(t)}{dt} = \dot{h}(t) = \mathbf{A} h(t) + \mathbf{B} x(t)
$$

$$
y(t) = \mathbf{C} h(t) + \mathbf{D} x(t)
$$

```text
Continuous SSM Block Diagram:
                 x(t) ──────────────────────────[ D ]─────────────────────────(+)──► y(t)
                   │                                                           ^
                   ▼                                                           │
                  [ B ]                                                       [ C ]
                   │                                                           │
                   ▼           ┌─────────────┐                                 │
                  (+)─────────►│ Integrator  ├────────┬────────────────────────┤
                   ^           │   ∫ dt      │        │ h(t)
                   │           └─────────────┘        │
                   │                                  │
                   └─────────────────[ A ]────────────┘
```

- **Transition Matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$:** Controls how the internal hidden state evolves over time in the absence of external inputs.
- **Input Matrix $\mathbf{B} \in \mathbb{R}^{N \times 1}$:** Maps the continuous scalar input $x(t)$ into the $N$-dimensional state dynamics.
- **Output Matrix $\mathbf{C} \in \mathbb{R}^{1 \times N}$:** Projects the latent state $h(t)$ back to the observable output space.
- **Feedthrough Matrix $\mathbf{D} \in \mathbb{R}^{1 \times 1}$:** Direct skip connection from input to output (often omitted, setting $\mathbf{D} = 0$).

---

### 3.3 Discretization via Zero-Order Hold (ZOH)

Digital computers process discrete tokens $[x_0, x_1, \dots, x_L]$ rather than continuous signals $x(t)$. To apply SSMs to discrete sequences, continuous differential equations are converted into discrete recurrence relations using a sample step size parameter $\Delta \in \mathbb{R}^+$.

Under the **Zero-Order Hold (ZOH)** assumption, the input $x(t)$ is assumed to be constant over the sample interval $[k\Delta, (k+1)\Delta)$:

$$
x(t) = x_k \quad \forall t \in [k\Delta, (k+1)\Delta)
$$

Integrating the continuous state equation over the interval $\Delta$ yields:

$$
h_k = h(k\Delta) = \exp(\Delta \mathbf{A}) h_{k-1} + \left( \int_{0}^{\Delta} \exp(\tau \mathbf{A}) d\tau \cdot \mathbf{B} \right) x_k
$$

Defining the discretized matrices $\mathbf{\bar{A}}$ and $\mathbf{\bar{B}}$:

$$
\mathbf{\bar{A}} = \exp(\Delta \mathbf{A})
$$

$$
\mathbf{\bar{B}} = (\Delta \mathbf{A})^{-1} \left( \exp(\Delta \mathbf{A}) - \mathbf{I} \right) \cdot (\Delta \mathbf{B}) \approx \Delta \mathbf{B} \quad (\text{via first-order Taylor expansion})
$$

$$
\mathbf{\bar{C}} = \mathbf{C}
$$

$$
\text{Discrete SSM Equations: } \begin{cases} h_k = \mathbf{\bar{A}} h_{k-1} + \mathbf{\bar{B}} x_k \\ y_k = \mathbf{\bar{C}} h_k \end{cases}
$$

---

### 3.4 The Dual Representation of Linear Time-Invariant (LTI) SSMs

When matrices $\mathbf{A}, \mathbf{B}, \mathbf{C}, \Delta$ are **Linear Time-Invariant (LTI)** (i.e., constant across all timesteps $k$), the system exhibits a mathematical duality: it can be executed as a **Recurrent Network** during inference and as a **1D Convolution** during training.

#### 3.4.1 Recurrent Representation (Fast Linear Inference)

$$
h_0 = \mathbf{\bar{B}} x_0
$$

$$
h_1 = \mathbf{\bar{A}} h_0 + \mathbf{\bar{B}} x_1 = \mathbf{\bar{A}} \mathbf{\bar{B}} x_0 + \mathbf{\bar{B}} x_1
$$

$$
h_k = \mathbf{\bar{A}} h_{k-1} + \mathbf{\bar{B}} x_k
$$

$$
y_k = \mathbf{\bar{C}} h_k
$$

- **Inference Complexity:** Requires only storing the fixed-size vector $h_k \in \mathbb{R}^N$. Generating the next token is strictly $O(1)$ time and $O(1)$ memory.

#### 3.4.2 Convolutional Representation (Parallel Training via FFT)

Expanding the discrete recurrence explicitly across all timesteps from $k = 0$ to $L$:

$$
y_0 = \mathbf{\bar{C}} \mathbf{\bar{B}} x_0
$$

$$
y_1 = \mathbf{\bar{C}} \mathbf{\bar{A}} \mathbf{\bar{B}} x_0 + \mathbf{\bar{C}} \mathbf{\bar{B}} x_1
$$

$$
y_2 = \mathbf{\bar{C}} \mathbf{\bar{A}}^2 \mathbf{\bar{B}} x_0 + \mathbf{\bar{C}} \mathbf{\bar{A}} \mathbf{\bar{B}} x_1 + \mathbf{\bar{C}} \mathbf{\bar{B}} x_2
$$

$$
y_k = \sum_{j=0}^{k} \mathbf{\bar{C}} \mathbf{\bar{A}}^{k-j} \mathbf{\bar{B}} x_j
$$

This operation is a discrete convolution between the input sequence $\mathbf{x}$ and a non-causal/causal SSM convolution kernel $\mathbf{\bar{K}} \in \mathbb{R}^L$:

$$
\mathbf{\bar{K}} = \left( \mathbf{\bar{C}}\mathbf{\bar{B}}, \; \mathbf{\bar{C}}\mathbf{\bar{A}}\mathbf{\bar{B}}, \; \mathbf{\bar{C}}\mathbf{\bar{A}}^2\mathbf{\bar{B}}, \; \dots, \; \mathbf{\bar{C}}\mathbf{\bar{A}}^{L-1}\mathbf{\bar{B}} \right)
$$

$$
y = x * \mathbf{\bar{K}}
$$

```text
Convolutional Computation via Fast Fourier Transform (FFT):
  Input Sequence x ────────► FFT ────┐
                                     ▼
  Precomputed Kernel K_bar ──► FFT ──► Elementwise Multiply ──► IFFT ──► Output y
```

By the Convolution Theorem, computing $y = x * \mathbf{\bar{K}}$ via FFT requires only $O(L \log L)$ time and can be computed entirely in parallel on GPUs during pre-training.

---

### 3.5 Long-Range Memory & The HiPPO Matrix

If matrix $\mathbf{A}$ is initialized randomly, the recurrent hidden state $h_k = \mathbf{\bar{A}}^k h_0$ suffers from vanishing or exploding gradients over long sequences.

**HiPPO (High-order Polynomial Projection Operators)** solves this by designing $\mathbf{A}$ such that the state vector $h(t)$ maintains an optimal online compression of the continuous history $x(\le t)$ projected onto an orthogonal basis of shifted Legendre polynomials:

$$
\mathbf{A}_{\text{HiPPO}}(n, k) = \begin{cases} (2n + 1)^{1/2} (2k + 1)^{1/2}, & \text{if } n > k \\ n + 1, & \text{if } n = k \\ -(2n + 1)^{1/2} (2k + 1)^{1/2}, & \text{if } n < k \end{cases}
$$

$$
\mathbf{B}_{\text{HiPPO}}(n) = (2n + 1)^{1/2}
$$

```text
HiPPO State Reconstruction Over Time:
Continuous Signal x(t) ──► [ HiPPO Matrix A ] ──► Vector of Legendre Coefficients h(t)
                                                           │
                                                           ▼ Inverse Legendre Transform
Reconstructed Historical Signal x_reconstructed(tau <= t) ◄─┘
```

The HiPPO initialization allows structured SSMs (S4) to solve long-range benchmarks (e.g., Path-X with $16\text{k}$ token contexts) that break standard RNNs and un-augmented Transformers.

---

### 3.6 The Fundamental Failure Mode of LTI SSMs: Content Invariance

Despite their computational efficiency, LTI SSMs fail on fundamental linguistic reasoning tasks:
1. **Selective Copying:** Memorizing a subset of key tokens while ignoring padding/stop-words.
2. **Induction Heads:** Associating patterns like $[A][B] \dots [A] \to [B]$ (associative in-context recall).

```text
Task 1: Selective Copying
Input:   [BOS]  dog  [PAD]  [PAD]  cat  [PAD]  [EOS]
LTI SSM: Memory state weights every token EQUALLY via static A, B.
         Cannot selectively drop [PAD] without altering state dynamics!

Task 2: Induction Heads
Input:   Harry ... Potter ... Harry ──► Target: [Potter]
LTI SSM: Cannot dynamically trigger recall conditioned on token identity.
```

**Root Cause:** Because $\mathbf{A}, \mathbf{B}, \mathbf{C}, \Delta$ are static and independent of the input $x_k$, the model cannot modulate information flow based on sequence content.

---

### 3.7 Mamba: Selective State Spaces (S6)

Mamba resolves the LTI dilemma by introducing **Selective State Spaces (S6)**, making matrices $\mathbf{B}, \mathbf{C}$ and the discretization step size $\Delta$ dynamic functions of the current input token $x$.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LTI SSM vs. MAMBA SELECTIVE SSM                        │
│                                                                             │
│  [LTI Structured SSM (S4)]:                                                 │
│   Input x_k ──────► [ Constant B ] ──► State Update (Constant A, Delta)     │
│   - Static mappings. Zero content-dependent filtering.                      │
│                                                                             │
│  [Mamba Selective SSM (S6)]:                                                │
│   Input x_k ──┬───► [ B_k = Linear_B(x_k) ]       (Dynamic Projection)      │
│               ├───► [ C_k = Linear_C(x_k) ]       (Dynamic Output Readout)  │
│               └───► [ Delta_k = Softplus(Linear_Delta(x_k)) ] (Dynamic Gate)│
│               │                                                             │
│               ▼                                                             │
│   Continuous state dynamically filtered based on token importance!          │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 3.7.1 Mathematical Formulation of Selective Parameterization

For an input batch of hidden representations $x \in \mathbb{R}^{B \times L \times D}$ (where $B$ is batch size, $L$ is sequence length, $D$ is model dimension, and $N$ is state expansion dimension):

$$
\mathbf{B}_k = \text{Linear}_{N}(x_k) \in \mathbb{R}^{B \times L \times N}
$$

$$
\mathbf{C}_k = \text{Linear}_{N}(x_k) \in \mathbb{R}^{B \times L \times N}
$$

$$
\Delta_k = \text{Softplus}\left( \text{Parameter}_{\Delta} + \text{Linear}_{D}(x_k) \right) \in \mathbb{R}^{B \times L \times D}
$$

$$
\mathbf{\bar{A}}_k = \exp\left( \Delta_k \mathbf{A} \right) \in \mathbb{R}^{B \times L \times D \times N}
$$

$$
\mathbf{\bar{B}}_k = \Delta_k \mathbf{B}_k \in \mathbb{R}^{B \times L \times D \times N}
$$

$$
\text{Selective Recurrence: } h_k = \mathbf{\bar{A}}_k h_{k-1} + \mathbf{\bar{B}}_k x_k
$$

$$
y_k = \mathbf{C}_k h_k
$$

#### 3.7.2 Physical Intuition of Step-Size Delta

The input-dependent step size $\Delta_k$ acts as a continuous generalized gating mechanism:
- **Large $\Delta_k \to \infty$:** $\mathbf{\bar{A}}_k = \exp(\Delta_k \mathbf{A}) \to 0$, while $\mathbf{\bar{B}}_k \to \text{large}$. The model **resets** the historical state $h_{k-1}$ and writes the current token $x_k$ into memory.
- **Small $\Delta_k \to 0$:** $\mathbf{\bar{A}}_k = \exp(\Delta_k \mathbf{A}) \to \mathbf{I}$, while $\mathbf{\bar{B}}_k \to 0$. The model **ignores** the current token $x_k$ (e.g., punctuation, padding) and preserves historical context $h_{k-1}$ unchanged.

---

### 3.8 Systems Innovations for Selective SSMs

Making $\mathbf{B}, \mathbf{C}, \Delta$ time-varying breaks the convolutional representation ($y = x * \mathbf{\bar{K}}$ no longer holds because the kernel changes at every token). Without FFTs, naive recurrence would force sequential $O(L)$ execution during training, abandoning GPU parallelization.

Mamba solves this through three hardware-centric algorithmic innovations:

#### 3.8.1 The Parallel Associative Scan

The discrete recurrence $h_k = \mathbf{\bar{A}}_k h_{k-1} + \mathbf{\bar{B}}_k x_k$ can be formulated as an **associative binary operator** $\circ$:

$$
(a_j, b_j) \circ (a_i, b_i) = (a_j \cdot a_i, \; a_j \cdot b_i + b_j)
$$

```text
Blelloch Parallel Associative Scan Tree (Sequence Length L = 4):
Step 1: Up-Sweep (Reduction Phase)
        [h0]       [h1]       [h2]       [h3]
          \         /           \         /
           ▼       ▼             ▼       ▼
         [ (h0 o h1) ]         [ (h2 o h3) ]
               \                     /
                └─────────┬─────────┘
                          ▼
                  [ (h0 o h1 o h2 o h3) ]

Step 2: Down-Sweep (Distribution Phase)
All intermediate prefix sums materialized concurrently!
Total Compute Work: O(L) | Parallel Execution Span / Depth: O(log L)
```

Using parallel prefix sum algorithms (Blelloch scan), the sequence of hidden states is computed in $O(\log L)$ parallel depth across GPU thread blocks.

#### 3.8.2 Hardware-Aware Memory Hierarchy & Kernel Fusion

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MAMBA FUSED HARDWARE-AWARE KERNEL                        │
│                                                                             │
│  High Bandwidth Memory (HBM):                                               │
│   Loads: x, Delta, A, B, C ────────────────────────────────────────┐        │
│                                                                    │        │
│  On-Chip SRAM (Fused CUDA / Triton Kernel):                         ▼        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 1. Discretization in SRAM:  A_bar = exp(Delta * A),  B_bar = Delta * B│  │
│  │ 2. Parallel Scan in SRAM:   h_k = A_bar * h_{k-1} + B_bar * x         │  │
│  │ 3. Output Projection:       y_k = C * h_k                             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│  High Bandwidth Memory (HBM):       ▼                                       │
│   Stores ONLY Final Output y ◄──────┘                                       │
│   (Never materializes intermediate expanded states [B, L, D, N] in HBM!)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **Memory Traffic Reduction:** Instead of reading/writing the giant intermediate state tensor of shape $(B, L, D, N)$ to slow HBM, Mamba fuses Discretization, Scan, and Readout into a single GPU kernel executed entirely in fast on-chip SRAM.

#### 3.8.3 Activation Recomputation in the Backward Pass

During backpropagation, computing gradients requires the intermediate states $h_k$. Instead of saving all $h_k$ in HBM during the forward pass (which would consume gigabytes of memory), Mamba **discards** $h_k$ and recomputes it dynamically in SRAM during the backward pass.

$$
\text{Memory Overhead: } O(L \cdot D) \quad \text{instead of} \quad O(L \cdot D \cdot N)
$$

---

### 3.9 The Mamba Block Architecture vs. Transformer Decoder

```text
┌──────────────────────────────────┐      ┌──────────────────────────────────┐
│    TRANSFORMER DECODER BLOCK     │      │        MAMBA BLOCK (S6)          │
├──────────────────────────────────┤      ├──────────────────────────────────┤
│ Input Hidden State x             │      │ Input Hidden State x             │
│        │                         │      │        │                         │
│        ▼                         │      │        ├──► [Linear Expand (2D)] │
│   [ RMSNorm ]                    │      │        │           │             │
│        │                         │      │        │           ▼             │
│        ▼                         │      │        │     [ 1D Conv (d_conv) ]│
│ [ Multi-Head Self-Attention ]    │      │        │           │             │
│ (Requires KV Cache in Gen)       │      │        │           ▼             │
│        │                         │      │        │      [ SiLU Act ]       │
│        ▼                         │      │        │           │             │
│       (+) Residual               │      │        │           ▼             │
│        │                         │      │        │     [ Selective SSM S6 ]│
│        ▼                         │      │        │           │             │
│   [ RMSNorm ]                    │      │        │           ▼             │
│        │                         │      │        │     ( Multiplication ) ◄┼── [ Linear Gating Branch ]
│        ▼                         │      │        │           │             │       │
│ [ Multi-Layer Perceptron (FFN) ] │      │        │           ▼             │       ▼
│        │                         │      │        │   [ Linear Down (D) ]   │   [ SiLU Gate ]
│        ▼                         │      │        │           │             │
│       (+) Residual               │      │        ▼           ▼             │
│        │                         │      │       (+) ◄──────────────────────┘
│        ▼                         │      │        │                         │
│ Output State                     │      │ Output State                     │
└──────────────────────────────────┘      └──────────────────────────────────┘
```

---

## 4. Module III: Distributed Training, Memory Layouts & Systems Parallelism

### 4.1 Anatomy of GPU Memory in Large Language Models

To train multi-billion parameter models, memory consumption must be decomposed into its constituent elements.

#### 4.1.1 The 16 Bytes per Parameter Rule in Mixed-Precision Adam

In standard mixed-precision training (e.g., FP16/BF16 forward/backward with FP32 Adam optimizer), **each model parameter requires 16 bytes of static GPU memory**:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│             STATIC MEMORY BREAKDOWN FOR A 7B PARAMETER MODEL                │
│                                                                             │
│  Component               Precision    Bytes/Param    7B Model Memory        │
│ ─────────────────────────────────────────────────────────────────────────── │
│  Model Parameters        FP16/BF16      2 bytes          14.0 GB            │
│  Gradients               FP16/BF16      2 bytes          14.0 GB            │
│  FP32 Master Weights     FP32           4 bytes          28.0 GB            │
│  Adam Momentum (m)       FP32           4 bytes          28.0 GB            │
│  Adam Variance (v)       FP32           4 bytes          28.0 GB            │
│ ─────────────────────────────────────────────────────────────────────────── │
│  TOTAL STATIC FOOTPRINT:               16 bytes         112.0 GB            │
│  (Exceeds a single 80 GB A100/H100 GPU before even allocating activations!) │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 4.1.2 The Mathematical Necessity of FP32 Master Weights & Optimizer States

Why can we not simply train entirely in FP16?

1. **Underflow of Gradient Updates:**
   In FP16, the minimum positive subnormal value is $\approx 5.96 \times 10^{-8}$, and the minimum normal value is $\approx 6.10 \times 10^{-5}$. A standard weight update is:
   $$
   \Delta w = \eta \cdot g_t = 10^{-4} \times 10^{-3} = 10^{-7}
   $$
   If added directly to an FP16 weight $w = 1.5000$, the machine epsilon $\epsilon_{\text{FP16}} \approx 9.77 \times 10^{-4}$. The update $\Delta w$ is rounded down to $0$, causing optimization to stall completely.
2. **Compounding Variance in Momentum/Variance Accumulators:**
   Adam tracks running exponential moving averages:
   $$
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
   $$
   $$
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   $$
   Over $100,000$ iterations, FP16 rounding errors compound non-linearly, causing numerical divergence.

---

### 4.2 Data Parallelism (DDP) & Collective All-Reduce

In Distributed Data Parallelism (DDP), the model is replicated across $N$ GPUs. Each GPU processes an independent mini-batch of data of size $B_{\text{local}} = B_{\text{global}} / N$.

```text
GPU 0 (Local Batch 0) ──► Forward ──► Backward ──► Local Grads ∇L_0 ──┐
                                                                      │
GPU 1 (Local Batch 1) ──► Forward ──► Backward ──► Local Grads ∇L_1 ──┼──► [ Ring All-Reduce ]
                                                                      │    ∇L = 1/N * sum(∇L_i)
GPU 2 (Local Batch 2) ──► Forward ──► Backward ──► Local Grads ∇L_2 ──┤           │
                                                                      │           ▼
GPU 3 (Local Batch 3) ──► Forward ──► Backward ──► Local Grads ∇L_3 ──┘    All GPUs update with
                                                                           identical averaged ∇L
```

- **Ring All-Reduce Communication:** Communicates $2 \times \frac{N-1}{N} \times |W|$ bytes per step in two phases: **Reduce-Scatter** followed by **All-Gather**.

---

### 4.3 Tensor Parallelism (Megatron-LM Intra-Layer Slicing)

When a single layer is too large to fit in GPU memory, **Tensor Parallelism (TP)** shards weight matrices across GPUs along rows or columns within the same node over NVLink.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                MEGATRON-LM TENSOR PARALLEL MLP ARCHITECTURE                 │
│                                                                             │
│  Input X ──┬──► GPU 0: Column Linear W_1,0 ──► GeLU ──► Row Linear W_2,0 ──┐│
│  [B, d]    │                                                               ││
│            └──► GPU 1: Column Linear W_1,1 ──► GeLU ──► Row Linear W_2,1 ──┼┼──► [ All-Reduce (Sum) ] ──► Output Y
│                                                                            ││
│  [Non-Commutative Activation]: No communication between W_1 and W_2!       ││
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 4.3.1 Slicing the Multi-Layer Perceptron (MLP) Block

The Transformer MLP consists of an up-projection $W_1 \in \mathbb{R}^{d \times 4d}$ followed by a down-projection $W_2 \in \mathbb{R}^{4d \times d}$:

$$
\text{MLP}(X) = \text{GeLU}(X W_1) W_2
$$

1. **Column-Parallel Linear 1:**
   Split $W_1$ along columns into $N$ shards: $W_1 = [W_{1,0} \mid W_{1,1} \mid \dots \mid W_{1,N-1}]$, where $W_{1,i} \in \mathbb{R}^{d \times \frac{4d}{N}}$.
   $$
   Y_{1,i} = X W_{1,i}
   $$
   Because $\text{GeLU}(\cdot)$ is an element-wise operator:
   $$
   Z_i = \text{GeLU}(Y_{1,i}) = \text{GeLU}(X W_{1,i})
   $$
   **Zero inter-GPU communication is required after Linear 1!**
2. **Row-Parallel Linear 2:**
   Split $W_2$ along rows into $N$ shards matching the output of $Z$:
   $$
   W_2 = \begin{bmatrix} W_{2,0} \\ W_{2,1} \\ \vdots \\ W_{2,N-1} \end{bmatrix}, \quad \text{where } W_{2,i} \in \mathbb{R}^{\frac{4d}{N} \times d}
   $$
   Each GPU computes local matrix multiplication:
   $$
   \text{Partial}_i = Z_i W_{2,i} \in \mathbb{R}^{B \times d}
   $$
   By block matrix multiplication:
   $$
   Y = Z W_2 = \sum_{i=0}^{N-1} Z_i W_{2,i} = \sum_{i=0}^{N-1} \text{Partial}_i
   $$
   A single **All-Reduce (Sum)** collective combines the partial sums across GPUs, returning the identical mathematical output $Y$ to all devices.

#### 4.3.2 Slicing Multi-Head Self-Attention (MHSA)

- **$W_Q, W_K, W_V$ Projections:** Column-parallel (split across attention heads).
- **$W_O$ Output Projection:** Row-parallel (combines head outputs via a single All-Reduce).
- **Communication Cost:** Exactly $2 \times \text{All-Reduce}$ per Transformer layer in the forward pass, and $2 \times \text{All-Reduce}$ in the backward pass.

---

### 4.4 Pipeline Parallelism (PP) & Inter-Layer Scheduling

Pipeline Parallelism partitions the sequential layers of a network across $P$ pipeline stages (GPUs).

#### 4.4.1 The Pipeline Bubble & Mathematical Formulation

If a batch is executed as a monolithic unit, downstream GPUs sit idle waiting for upstream activations, creating a **pipeline bubble**:

$$
F_{\text{bubble}} = \frac{t_{\text{idle}}}{t_{\text{total}}} = \frac{p - 1}{p - 1 + m}
$$

Where $p$ is the number of pipeline stages, and $m$ is the number of micro-batches ($m \gg p$).

```text
GPipe Schedule (p = 4 stages, m = 4 micro-batches):
Time ──►  t1   t2   t3   t4   t5   t6   t7   t8   t9
GPU 3:   [  ] [  ] [  ] [F1] [F2] [F3] [F4] [B4] [B3] ...
GPU 2:   [  ] [  ] [F1] [F2] [F3] [F4] [B4] [B3] [B2] ...
GPU 1:   [  ] [F1] [F2] [F3] [F4] [B4] [B3] [B2] [B1] ...
GPU 0:   [F1] [F2] [F3] [F4] [B4] [B3] [B2] [B1] [  ] ...
         |<-- Bubble -->|                   |<-- Bubble -->|
Peak Activation Memory on GPU 0 = O(m) micro-batches!
```

#### 4.4.2 Mathematical Equivalence of Micro-Batch Gradient Accumulation

Splitting a mini-batch of size $B$ into $m$ micro-batches of size $b = B/m$ computes mathematically identical gradients:

$$
\mathcal{L}_{\text{full}}(W) = \frac{1}{B} \sum_{i=1}^{B} \ell(x_i; W) = \frac{1}{m} \sum_{k=1}^{m} \left( \frac{1}{b} \sum_{j=1}^{b} \ell(x_{k,j}; W) \right) = \frac{1}{m} \sum_{k=1}^{m} \mathcal{L}_k(W)
$$

$$
\nabla_W \mathcal{L}_{\text{full}}(W) = \frac{1}{m} \sum_{k=1}^{m} \nabla_W \mathcal{L}_k(W)
$$

Accumulating micro-batch gradients in place (`grad_buffer += dL_k/dW`) is bit-for-bit identical to full batch processing.

#### 4.4.3 GPipe vs. 1F1B Schedule Activation Footprints

```text
1F1B (One-Forward-One-Backward) Schedule on GPU 0:
Time ──►  t1   t2   t3   t4   t5   t6   t7   t8
GPU 0:   [F1] [F2] [F3] [F4] [B1] [F5] [B2] [F6] ...
Act Mem:  +1   +2   +3   +4   -1   +4   -1   +4   (Bounded at p = 4!)
```

- **GPipe:** Executes all $m$ forward passes before any backward pass. Peak memory scales with micro-batches $O(m)$.
- **1F1B (Megatron/DeepSpeed):** After a warmup phase of $p$ steps, alternates 1 Forward with 1 Backward. Each backward pass frees the activations of that micro-batch, bounding peak activation memory to $O(p)$, independent of $m$.

---

### 4.5 Context Parallelism (CP) & Ring Attention

For extreme context windows ($128\text{k}-1\text{M}$ tokens), self-attention memory becomes the dominant bottleneck. Context Parallelism shards the sequence length dimension $L$ across $N$ GPUs.

#### 4.5.1 The Ring Communication Topology

```text
Ring Attention Execution Over 4 GPUs (Sequence Length L sharded into 4 chunks):
Step 1: Compute Local Attention with own Q_i, K_i, V_i
        Simultaneously Send K_i, V_i to (i+1) mod N

Step 2: Receive K_{i-1}, V_{i-1} from upstream GPU
        Compute Cross-Attention with local Q_i and received KV
        Simultaneously Send received KV to next GPU in Ring

Step 3...N: Repeat until all N KV blocks traverse the ring.
All-to-all attention computed with zero memory footprint for the full L x L matrix!
```

#### 4.5.2 Online Numerically Stable Softmax Accumulation

Because softmax requires the global row-wise maximum $\max(S)$, computing attention across sharded blocks incrementally requires **Online Softmax rescaling** (Milakov & Gimelshein, 2018):

```text
Online Softmax Update Step:
Given Old Block Stats: (Max_old, Sum_old, Output_old)
Given New Block Stats: (Max_new, Sum_new, Output_new)

1. Compute Global Max:
   Max_global = max(Max_old, Max_new)

2. Rescale Correction Factors:
   alpha_old = exp(Max_old - Max_global)
   alpha_new = exp(Max_new - Max_global)

3. Update Normalization Denominator:
   Sum_total = Sum_old * alpha_old + Sum_new * alpha_new

4. Accumulate Output Representation:
   Output_final = (Output_old * Sum_old * alpha_old + Output_new * Sum_new * alpha_new) / Sum_total
```

This formulation allows Ring Attention to compute numerically exact full self-attention without ever materializing the global attention matrix.

---

### 4.6 ZeRO (Zero Redundancy Optimizer) & FSDP Mechanics

ZeRO eliminates the memory redundancy inherent in standard data parallelism by sharding optimizer states, gradients, and parameters across the data-parallel world.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ZeRO MEMORY SHARDING STAGES                        │
│                                                                             │
│  [Standard DDP (ZeRO-0)]: 112 GB / GPU                                      │
│  ┌──────────────────┬──────────────────┬─────────────────────────────────┐  │
│  │ Params (14 GB)   │ Gradients (14 GB)│ Adam Optimizer States (84 GB)   │  │
│  └──────────────────┴──────────────────┴─────────────────────────────────┘  │
│                                                                             │
│  [ZeRO-Stage 1 (Optimizer State Sharding)]: 49 GB / GPU                     │
│  ┌──────────────────┬──────────────────┬──────────────┐                     │
│  │ Params (14 GB)   │ Gradients (14 GB)│ Optim (21 GB)│  (4x Optim Sharded) │
│  └──────────────────┴──────────────────┴──────────────┘                     │
│                                                                             │
│  [ZeRO-Stage 2 (Gradient Sharding)]: 38.5 GB / GPU                          │
│  ┌──────────────────┬──────────────┬──────────────┐                         │
│  │ Params (14 GB)   │ Grads(3.5 GB)│ Optim (21 GB)│  (Reduce-Scatter Grads) │
│  └──────────────────┴──────────────┴──────────────┘                         │
│                                                                             │
│  [ZeRO-Stage 3 / FSDP (Full Parameter Sharding)]: 28 GB / GPU               │
│  ┌──────────────┬──────────────┬──────────────┐                             │
│  │Params(3.5 GB)│ Grads(3.5 GB)│ Optim (21 GB)│  (On-Demand All-Gather)     │
│  └──────────────┴──────────────┴──────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 4.6.1 ZeRO-Stage 1: Optimizer State Sharding
- **Mechanism:** Partitions FP32 Adam states ($m, v$, master weights) into $N_{\text{data}}$ shards.
- **Workflow:** Standard forward/backward $\to$ Gradient All-Reduce $\to$ Each GPU updates only its local $1/N$ parameter shard $\to$ Post-step **All-Gather** to synchronize updated parameters.
- **Memory Saved:** Reduces static optimizer memory from $12 \times \Phi$ to $\frac{12 \times \Phi}{N}$.

#### 4.6.2 ZeRO-Stage 2: Gradient Sharding
- **Mechanism:** Instead of All-Reduce (which sends all gradients to all GPUs), runs **Reduce-Scatter**.
- **Workflow:** Each GPU receives and retains averaged gradients *only* for its assigned parameter shard. All other gradients are discarded immediately during backpropagation.
- **Memory Saved:** Reduces gradient buffer memory from $2 \times \Phi$ to $\frac{2 \times \Phi}{N}$.

#### 4.6.3 ZeRO-Stage 3: Parameter Sharding (Fully Sharded Data Parallel)
- **Mechanism:** Parameters themselves are partitioned across all GPUs.
- **Workflow:**
  1. **Before Layer $L$ Forward:** All GPUs run an **All-Gather** to assemble Layer $L$'s weights $\to$ Compute forward pass $\to$ Non-owning GPUs **immediately discard** Layer $L$'s parameters from memory.
  2. **Before Layer $L$ Backward:** All GPUs run an **All-Gather** to re-materialize Layer $L$'s weights $\to$ Compute gradients $\to$ Run **Reduce-Scatter** for gradients $\to$ Discard parameters again.

#### 4.6.4 Communication vs. Memory Trade-Off Across Stages

| Configuration | Parameters / GPU | Gradients / GPU | Optimizer / GPU | Total Static Memory (7B on 4 GPUs) | Total Extra Communication Volume |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DDP (ZeRO-0)** | $2\Phi$ ($14\text{ GB}$) | $2\Phi$ ($14\text{ GB}$) | $12\Phi$ ($84\text{ GB}$) | $112.0\text{ GB}$ | Baseline ($2\Phi$ All-Reduce) |
| **ZeRO-1** | $2\Phi$ ($14\text{ GB}$) | $2\Phi$ ($14\text{ GB}$) | $\frac{12\Phi}{N}$ ($21\text{ GB}$) | $49.0\text{ GB}$ | Baseline $+ \Phi$ All-Gather |
| **ZeRO-2** | $2\Phi$ ($14\text{ GB}$) | $\frac{2\Phi}{N}$ ($3.5\text{ GB}$) | $\frac{12\Phi}{N}$ ($21\text{ GB}$) | $38.5\text{ GB}$ | Baseline (Reduce-Scatter replaces All-Reduce) |
| **ZeRO-3 / FSDP** | $\frac{2\Phi}{N}$ ($3.5\text{ GB}$) | $\frac{2\Phi}{N}$ ($3.5\text{ GB}$) | $\frac{12\Phi}{N}$ ($21\text{ GB}$) | $28.0\text{ GB}$ | Baseline $+ 1.5\times$ Forward/Backward All-Gather |

---

### 4.7 Operator & Kernel Fusion (FlashAttention Deep Dive)

Standard attention materializes an $L \times L$ attention score matrix in HBM, requiring $O(L^2)$ memory reads and writes:

```text
Standard Attention (Memory Wall Bottleneck):
  1. S = Q K^T          ──► Write S [B, H, L, L] to HBM
  2. P = softmax(S)     ──► Read S from HBM, Write P [B, H, L, L] to HBM
  3. O = P V            ──► Read P from HBM, Write O [B, H, L, d] to HBM
  Total HBM Transfers: O(L^2) bytes (Massive memory-bound bottleneck!)

FlashAttention (Tiled On-Chip SRAM Fusion):
  ┌──────────────────────────────────────────────────────────────────┐
  │ For each Block of Q (in SRAM):                                  │
  │   For each Block of K, V (in SRAM):                             │
  │     1. Compute S_tile = Q_tile * K_tile^T                        │
  │     2. Compute Online Softmax on S_tile (Update running max/sum) │
  │     3. Accumulate O_tile += P_tile * V_tile in SRAM              │
  │ Write ONLY final output O to HBM!                                │
  │ Memory Complexity: O(L) in HBM | Speedup: 2x - 4x                │
  └──────────────────────────────────────────────────────────────────┘
```

---

### 4.8 Unified Distributed Parallelism Matrix

```text
                      DISTRIBUTED PARALLELISM DECISION TREE
                                        │
                    Is Model Size > Single GPU Memory?
                                   /  \
                                 NO    YES
                                 /      \
               [DDP / ZeRO-1]             Is Interconnect NVLink (Intra-Node)?
                                                /  \
                                              YES   NO
                                              /      \
                             [Tensor Parallel (TP)]   [ZeRO-3 / FSDP or Pipeline (PP)]
                                       │                              │
                        Is Context Length > 32k?            Is Context Length > 32k?
                                /     \                             /    \
                              YES      NO                         YES     NO
                              /         \                         /        \
                     [TP + CP]         [TP]                [FSDP + CP]   [FSDP]
```

---

## 5. Module IV: LLM Inference Economics, Memory Architecture & PagedAttention

### 5.1 The Economic Calculus of LLM Serving: TCO & Unit Cost per Token

In commercial API deployments (e.g., OpenAI, Anthropic, DeepSeek, OpenRouter), access to models is priced per token. The underlying profitability is strictly determined by **hardware unit economics**:

$$
\text{Cost per Token} = \frac{\text{Hourly GPU Server Cost}}{\text{Tokens Generated per Hour}}
$$

$$
\text{Tokens per Hour} = \text{Throughput (Tokens/s)} \times 3600\text{ s}
$$

- **Hardware Amortization & TCO:** An 8x H100 node costs $\approx \$300,000$ to purchase or $\$20 - \$30/\text{hr}$ to rent on cloud providers.
- **The Core Objective:** Maximize the aggregate token generation rate ($\text{Tokens/s}$) across the cluster to minimize the amortized cost per token.

---

### 5.2 The Two Execution Phases: Prefill vs. Decoding

LLM inference operates in two fundamentally distinct regimes with opposite computational profiles:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PREFILL VS. DECODING PHASES                           │
│                                                                             │
│  [Phase 1: Prefill / Prompt Processing]                                     │
│   Input: Full prompt of S tokens in parallel ──► Matrix-Matrix GEMM (BLAS3) │
│   - Compute-Bound (Very High Arithmetic Intensity: I >> I_knee)             │
│   - Generates the first token; determines Time To First Token (TTFT).       │
│   - Efficiently saturates Tensor Core TFLOPs.                               │
│                                                                             │
│  [Phase 2: Autoregressive Decoding]                                         │
│   Input: 1 token at a time ──► Matrix-Vector GEMV (BLAS2)                   │
│   - Memory-Bound (Very Low Arithmetic Intensity: I << I_knee)               │
│   - Must stream all model weights (2 * Phi bytes) from HBM for EVERY token! │
│   - Determines Time Per Output Token (TPOT) and generation speed (tps).     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 5.3 First-Principles FLOP & Parameter Counting (Case Study: LLaMA 3.3 70B)

Let us derive the exact parameters and FLOP requirements for a modern flagship architecture (LLaMA 3.3 70B: $d = 8192$, $L = 80$ layers, $V = 128,256$ vocabulary, Grouped-Query Attention with $n_q = 64$ heads, $n_{kv} = 8$ heads, and SwiGLU MLP intermediate size $d_{\text{ffn}} = 28,672$).

#### 5.3.1 Layer-by-Layer FLOP Derivations

Recall that multiplying an $m \times n$ matrix by an $n \times o$ matrix takes $2mno$ FLOPs ($n$ multiplications and $n-1 \approx n$ additions per element).

For a sequence of $S$ tokens:

1. **RMSNorm Operations ($4 \times S \times d$ FLOPs per norm):**
   $$
   \text{FLOPs}_{\text{RMSNorm}} = S \cdot d \text{ (square)} + S \cdot d \text{ (sum)} + S \cdot d \text{ (div)} + S \cdot d \text{ (scale)} \approx 4 S d
   $$
2. **Query Projection ($W_Q \in \mathbb{R}^{d \times d}$):**
   $$
   \text{FLOPs}_Q = 2 \cdot S \cdot d \cdot d = 2 S d^2
   $$
3. **Key & Value Projections ($W_K, W_V \in \mathbb{R}^{d \times \frac{d}{8}}$ for GQA with 8 KV heads):**
   $$
   \text{FLOPs}_{KV} = 2 \times \left( 2 \cdot S \cdot d \cdot \frac{d}{8} \right) = \frac{1}{2} S d^2
   $$
4. **Rotary Position Embedding (RoPE):**
   $$
   \text{FLOPs}_{\text{RoPE}} \approx 3 \cdot S \cdot d
   $$
5. **Attention Matrix Multiplications ($Q K^T$ and $A V$):**
   $$
   \text{FLOPs}_{QK^T} = 2 \cdot S \cdot S \cdot d = 2 S^2 d, \qquad \text{FLOPs}_{AV} = 2 \cdot S \cdot S \cdot d = 2 S^2 d
   $$
6. **Output Attention Projection ($W_O \in \mathbb{R}^{d \times d}$):**
   $$
   \text{FLOPs}_O = 2 \cdot S \cdot d \cdot d = 2 S d^2
   $$
7. **SwiGLU Feed-Forward Network (Gate $W_{\text{gate}}$, Up $W_{\text{up}}$, Down $W_{\text{down}}$ with $d_{\text{ffn}} = 3.5 d$):**
   $$
   \text{FLOPs}_{\text{FFN}} = 3 \times \left( 2 \cdot S \cdot d \cdot d_{\text{ffn}} \right) = 6 \cdot S \cdot d \cdot (3.5 d) = 21 S d^2
   $$

$$
\text{Total FLOPs per Layer} \approx \left( 2 + \frac{1}{2} + 2 + 21 \right) S d^2 + 4 S^2 d = 25.5 S d^2 + 4 S^2 d
$$

When generating a single token ($S = 1$), the attention quadratic term $4 S^2 d$ is negligible, yielding the universal rule of thumb:

$$
\text{Forward Pass Compute per Token} \approx 2 \times \Phi \text{ FLOPs}
$$

Where $\Phi$ is the total parameter count.

#### 5.3.2 Total Parameter Calculation

$$
\begin{aligned}
\text{Total Parameters} &= \underbrace{V \cdot d}_{\text{Embedding}} + L \cdot \left[ \underbrace{\left( 1 + \frac{1}{8} + \frac{1}{8} + 1 \right) d^2}_{\text{Attention: } Q, K, V, O} + \underbrace{3 \cdot d \cdot (3.5 d)}_{\text{SwiGLU: Gate, Up, Down}} + \underbrace{2 d}_{\text{RMSNorms}} \right] + \underbrace{d}_{\text{Final Norm}} + \underbrace{d \cdot V}_{\text{LM Head}} \\
&= 128256 \cdot 8192 + 80 \cdot \left( 12.75 \cdot 8192^2 + 2 \cdot 8192 \right) + 8192 + 8192 \cdot 128256 \\
&= 70,553,706,496 \text{ parameters} \approx \mathbf{70.55\text{ Billion}}
\end{aligned}
$$

At BF16 precision ($2\text{ bytes/param}$), storing model weights alone requires:

$$
\text{Weight Storage} = 70.55 \times 10^9 \times 2 = \mathbf{141.1\text{ GB}}
$$

---

### 5.4 The KV Cache: Eliminating Redundant Attention Computations

Without caching, computing the $t$-th generated token requires re-computing Key and Value projections for all preceding $t-1$ tokens, forcing $O(L^2)$ redundant matrix multiplications.

The **KV Cache** stores the calculated Key and Value activation vectors in GPU memory, converting generation to $O(1)$ compute per step and $O(L)$ total generation complexity.

```text
Without KV Cache (Catastrophic O(L^2) Redundancy):
Token 1: Compute QKV for [t0] ───────────────► 1 token compute
Token 2: Re-compute QKV for [t0, t1] ────────► 2 tokens compute
Token 3: Re-compute QKV for [t0, t1, t2] ────► 3 tokens compute
...
Token L: Re-compute QKV for [t0 ... tL-1] ───► L tokens compute
Total Operations: sum_{k=1}^L k = O(L^2)

With KV Cache (O(1) Step-Wise Generation):
Token 1: Compute & Save KV_0 in memory ──────► 1 token compute
Token 2: Load [KV_0], Compute & Append KV_1 ──► 1 token compute
Token 3: Load [KV_0, KV_1], Append KV_2 ─────► 1 token compute
Total Operations: O(L)
```

#### 5.4.1 Mathematical Memory Footprint Formulation

For a model with $L$ layers, $n_{kv}$ key-value heads, head dimension $d_{\text{head}}$, batch size $B$, and context length $S$, stored in 16-bit precision ($2\text{ bytes}$):

$$
\text{KV Cache Memory (Bytes)} = 2 \times \underbrace{2}_{\text{K and V}} \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times S \times B
$$

$$
\text{KV Cache per Token} = 4 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \text{ bytes}
$$

For LLaMA 3.3 70B ($L = 80, n_{kv} = 8, d_{\text{head}} = 128$):

$$
\text{KV Cache per Token} = 4 \times 80 \times 8 \times 128 = \mathbf{327,680\text{ bytes}} = \mathbf{0.32768\text{ MB/token}}
$$

#### 5.4.2 The KV Cache Memory Explosion at Scale

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│            KV CACHE FOOTPRINT SCALING (LLaMA 3.3 70B, BF16)                 │
│                                                                             │
│  Context Length (S)    Batch Size (B = 1)    Batch Size (B = 8)   B = 32    │
│ ─────────────────────────────────────────────────────────────────────────── │
│  2,048 tokens               0.67 GB                5.36 GB        21.47 GB  │
│  8,192 tokens               2.68 GB               21.47 GB        85.90 GB  │
│  32,768 tokens             10.74 GB               85.90 GB       343.60 GB  │
│  131,072 (128k)            42.95 GB              343.60 GB     1,374.39 GB  │
│                                                                             │
│  CRITICAL THRESHOLD:                                                        │
│  At B = 8, S = 128k context, KV Cache = 343.6 GB, far exceeding the model   │
│  weight footprint (141.1 GB)!                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 5.5 Batching Economics & The Law of Diminishing Latency

#### 5.5.1 Amortizing Memory Bandwidth Across Batch Dimension B

During single-request autoregressive generation ($B = 1$), the GPU must read all $141.1\text{ GB}$ of weights from HBM to produce just **one token**. On an 4x H100 server (aggregate memory bandwidth $\approx 4 \times 3.35\text{ TB/s} = 13.4\text{ TB/s}$), the theoretical generation time is:

$$
t_{\text{token}} \ge \frac{141.1\text{ GB}}{13,400\text{ GB/s}} \approx 10.5\text{ ms} \implies \approx 95\text{ tokens/s per stream}
$$

The arithmetic intensity is dismal:

$$
I(B=1) = \frac{2 \times \Phi}{2 \times \Phi} = 1\text{ FLOP / Byte} \ll I_{\text{knee}} \approx 150\text{ FLOPs / Byte}
$$

The Tensor Cores sit $99\%$ idle!

**The Power of Batching:** When processing a batch of $B$ requests simultaneously, the model weights are loaded **once** from HBM into on-chip cache and reused across all $B$ tokens:

$$
I(B) = \frac{2 \cdot B \cdot \Phi}{2 \cdot \Phi + \text{KV\_Cache}(B)} \approx B \text{ FLOPs / Byte}
$$

```text
Arithmetic Intensity vs. Batch Size:
Batch Size B = 1:   Arithmetic Intensity = 1 FLOP/Byte   (Heavily Memory-Bound)
Batch Size B = 16:  Arithmetic Intensity = 16 FLOPs/Byte (Still Memory-Bound, but 16x throughput!)
Batch Size B = 128: Arithmetic Intensity = 120 FLOPs/Byte (Approaching Compute-Bound Knee)
```

Batching amortizes the fixed memory read cost across multiple customers, scaling total throughput almost linearly with $B$ and driving down unit cost.

#### 5.5.2 Throughput vs. Per-User Latency (TPOT) Trade-Off

As batch size $B$ grows, total throughput increases, but per-user latency (**Time Per Output Token - TPOT**) eventually degrades due to:
1. Extra memory traffic to load the massive KV cache tensors of all $B$ requests.
2. Tensor Core compute saturation ($I \to I_{\text{knee}}$).
3. Inter-GPU communication sync barriers across Tensor Parallel ranks.

```text
Throughput & Latency Dynamics:
Throughput (Tokens/s)                               Per-User Latency (TPOT ms/token)
   ^                                                   ^
   |                 /------------ Max Roofline        |                          /
   |                /                                  |                         /
   |               /                                   |                        /
   |              /                                    |              /--------/
   |             /                                     |             /
   |------------/------------------------>             |------------/------------------------>
   0         Batch Size B                              0         Batch Size B
   [Throughput Scales Linearly with B]                 [Per-User Speed Slows Down at High B]
```

#### 5.5.3 Model Bandwidth Utilization (MBU) vs. Model FLOPs Utilization (MFU)

Theoretical throughput is never fully reached in real production servers. **Model Bandwidth Utilization (MBU)** measures how effectively a serving system utilizes the physical memory bandwidth:

$$
\text{MBU} = \frac{\text{Observed Throughput (tok/s)} \times \left( \text{Memory}_{\text{Weights}} + \text{Memory}_{\text{KV}} \right)}{\text{Peak GPU Hardware Bandwidth}}
$$

$$
\text{MFU} = \frac{\text{Observed Throughput (tok/s)} \times 2 \Phi}{\text{Peak Hardware TFLOPs}}
$$

In practice, production engines (vLLM, TensorRT-LLM) achieve **$40\% - 65\%$ MBU** due to non-GEMM kernel launch overheads, CUDA stream synchronization stalls, and warp memory bank conflicts.

---

### 5.6 PagedAttention: OS-Inspired Virtual Memory for KV Cache (vLLM)

#### 5.6.1 The Memory Fragmentation Crisis

In legacy serving frameworks (e.g., HuggingFace, FasterTransformer), memory for a request's KV cache was allocated as a single, contiguous chunk based on the request's maximum possible output length (`max_seq_len`, e.g., 4096 tokens).

```text
Legacy Contiguous Pre-Allocation (60% - 80% Memory Wasted):
Request 1 (Actual Length = 500 tokens):
┌────────────────────┬─────────────────────────────────────────────────┐
│ Used: 500 tokens   │ WASTED: 3,596 tokens (Internal Fragmentation)   │
└────────────────────┴─────────────────────────────────────────────────┘
Request 2 (Waiting in queue because no contiguous 4,096-token chunk exists!)
                      (External Fragmentation)
```

1. **Internal Fragmentation:** Memory reserved for tokens that are never generated (short requests).
2. **External Fragmentation:** Free memory scattered across non-contiguous fragments, preventing new requests from being scheduled.
3. **Over-allocation Waste:** $>60-80\%$ of GPU VRAM remained completely idle, capping batch sizes to small values.

#### 5.6.2 Block Tables & Virtual Paging Architecture

**PagedAttention** (Kwon et al., SOSP 2023 / vLLM) solves memory fragmentation by adapting the principles of **Operating System Virtual Memory with Paging** to the KV cache:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PAGEDATTENTION ARCHITECTURE                           │
│                                                                             │
│  Logical KV Blocks (Virtual Sequence):                                      │
│  ┌───────────────┬───────────────┬───────────────┐                          │
│  │ Block 0       │ Block 1       │ Block 2       │                          │
│  │ (tokens 0-15) │ (tokens 16-31)│ (tokens 32-47)│                          │
│  └───────┬───────┴───────┬───────┴───────┬───────┘                          │
│          │               │               │                                  │
│          ▼               ▼               ▼                                  │
│  Block Table: [ 0 ──► Page 7 ] [ 1 ──► Page 2 ] [ 2 ──► Page 11 ]           │
│                                                                             │
│  Physical Memory (Non-Contiguous GPU Pages in HBM):                         │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐        │
│  │ Page 2   │ Page 7   │ Page 11  │ Free     │ Page 4   │ Free     │        │
│  │ (Blk 1)  │ (Blk 0)  │ (Blk 2)  │          │ (Other)  │          │        │
│  └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Logical vs. Physical Partitioning:** The continuous sequence of KV vectors is split into fixed-size **Logical Blocks** (e.g., $16$ or $32$ tokens).
2. **Dynamic Page Allocation:** Physical blocks are allocated on-demand from a central pool as tokens are generated. Physical pages do *not* need to be contiguous in memory.
3. **Block Table Mapping:** The serving engine maintains a lookup table for each request mapping `Logical Block Index -> Physical Page Address`.
4. **Zero External Fragmentation & Minimal Internal Waste:** Memory is only allocated when a block fills up. Internal fragmentation is strictly bounded to the last partially-filled block ($<4\%$ total memory waste).

#### 5.6.3 Copy-on-Write (CoW) Forking & Parallel Sampling

When performing **Parallel Sampling** (e.g., generating $N$ independent candidate completions for a single prompt) or **Beam Search**, multiple output streams share the exact same prompt KV cache pages:

```text
Copy-on-Write (CoW) Memory Sharing:
Prompt Tokens (0-63) ──► Physical Pages [Page 1, Page 2, Page 3, Page 4]
                                ▲                          ▲
                                │                          │
               ┌────────────────┴────────────────┐         │
               │                                 │         │
Request Branch A (Child 1)        Request Branch B (Child 2)
Logical Table: [P1, P2, P3, P4]   Logical Table: [P1, P2, P3, P4]
               │                                 │
When Branch A generates new token:               │ When Branch B generates new token:
Allocates new private [Page 8]!                  │ Allocates new private [Page 9]!
```

- **Zero Memory Duplication:** Both branches reference identical prompt pages via reference counting.
- **Copy-on-Write:** A private physical page is allocated only when a branch writes new, diverging tokens. Reduces memory footprint by up to $55\%$ in multi-candidate generation.

#### 5.6.4 Prefix Caching (Prompt Caching)

In agentic workflows and multi-turn conversations, different user queries frequently share identical prompt prefixes (e.g., system instructions, few-shot examples, RAG documents).

**Prefix Caching** hashes blocks of tokens and stores their computed KV pages in a global LRU cache. When a new request arrives with a matching prefix hash, the engine **skips the prefill compute entirely** and binds the pre-existing KV physical pages directly to the request's Block Table.

$$
\text{TTFT Reduction: } 80\% - 95\% \quad \text{for cached multi-turn chats and agentic loops.}
$$

---

### 5.7 Continuous (Iteration-Level) Batching vs. Static Batching

Legacy static batching locks a batch of requests until *every single request* in the batch finishes generating:

```text
Static Batching (Tail Latency Bubble):
Req 1 (50 tokens):  [=== Decode ===] [--------- IDLE / STALLED WAITING ------------]
Req 2 (500 tokens): [======================= Decode ===============================]
Req 3 (100 tokens): [====== Decode ======] [--- IDLE / STALLED WAITING ------------]
                                                                                   ^
                                                              Batch finishes ONLY here!

Continuous / Iteration-Level Batching (Orca / vLLM):
Step t:   [Req 1 Step] [Req 2 Step] [Req 3 Step] ──► Req 1 Finishes & Leaves!
Step t+1: [Req 4 Prefill] [Req 2 Step] [Req 3 Step] ──► New Req 4 Immediately Injected!
Step t+2: [Req 4 Step]    [Req 2 Step] [Req 3 Step]
```

- **Dynamic Request Scheduling:** At every individual decoding step, finished sequences are evicted and returned to the client, while newly queued requests are seamlessly scheduled into the free execution slots.
- **Throughput Multiplier:** Continuous batching increases GPU utilization by $2\times - 4\times$ compared to static batching.

---

## 6. Module V: Novel Transformer Topologies — Per-Layer Embeddings (PLE) in Gemma 4

### 6.1 Motivation: Decoupling Capacity from FLOPs & Eliminating Identity Dilution

In standard Transformers, input tokens are mapped to continuous vectors *only once* at the input layer ($l = 0$). As activations propagate through deep networks ($L > 32$), two structural pathologies arise:
1. **Identity Dilution:** Continuous residual summation $h_L = h_0 + \sum_{l=0}^{L-1} f_l(h_l)$ dilutes the original lexical identity of the token.
2. **Compute-Memory Coupling:** Scaling model capacity (parameters) strictly forces an increase in matrix multiplication FLOPs per token ($O(d_{\text{model}}^2)$).

**Per-Layer Embeddings (PLE)** (introduced in Gemma 4 architectures) decouple total parameter capacity from per-token compute by providing every decoder layer with direct access to dedicated token embedding lookup tables.

---

### 6.2 Mathematical Formulation & Forward Pass

```text
                              [ Token ID: t ]
                                     │
          ┌──────────────────────────┴──────────────────────────┐
          │                                                     │
          ▼                                                     ▼
  [ Base Lookup e_0 ]                                [ Layer l PLE Lookup e_lookup^(l) ]
  (e_0 in R^{d_base})                                 (e_lookup^(l) in R^{d_ple})
          │                                                     │
  [ Projection W_proj ]                                         │
  (x_base^(l) in R^{d_ple})                                     │
          │                                                     │
          └──────────────────────────┬──────────────────────────┘
                                     │
                                     ▼
                          [ Combined Vector p_l ]
                                     │
  ──> [ Hidden State h_l ]           │
             │                       │
             ▼                       │
     [ RMSNorm + MHSA ] ──(+)        │
                          │          │
                          ▼          │
     [ RMSNorm + FFN  ] ──(+)        │
                          │          │
                          ▼          │
                      ( h_l'' ) ─────┼────────┐
                          │          │        │
                          │          ▼        ▼
                          │     [ Hadamard Product (g_l ⊙ p_l) ] <── [ Gate: σ(W_g h_l'') ]
                          │                   │
                          │                   ▼
                          │           [ Output Proj W_out ]
                          │                   │
                          │                   ▼
                          │            [ RMSNorm(u_l) ]
                          │                   │
                          └──────────┬────────┘
                                     ▼
                          [ Updated State h_(l+1) ]
```

#### 6.2.1 Pre-Layer Representation
For token index $t \in \mathcal{V}$, the architecture extracts two representations:
1. **Global Base Embedding:** $e_0 \in \mathbb{R}^{d_{\text{base}}}$ from global matrix $E_0 \in \mathbb{R}^{|\mathcal{V}| \times d_{\text{base}}}$.
2. **Layer-Specific Embedding:** $e_{\text{lookup}}^{(l)} \in \mathbb{R}^{d_{\text{ple}}}$ from layer $l$'s matrix $E_{\text{PLE}}^{(l)} \in \mathbb{R}^{|\mathcal{V}| \times d_{\text{ple}}}$.

$$
\mathbf{x}_{\text{base}}^{(l)} = W_{\text{proj}}^{(l)} e_0 \in \mathbb{R}^{d_{\text{ple}}}
$$

$$
p_l = \frac{1}{\sqrt{2}} \left( e_{\text{lookup}}^{(l)} + \mathbf{x}_{\text{base}}^{(l)} \right) \in \mathbb{R}^{d_{\text{ple}}}
$$

#### 6.2.2 State-Dependent Gating & Integration
Inside decoder layer $l$, the contextual hidden state $h_l'' = h_l + \text{Attn}(h_l) + \text{FFN}(h_l')$ modulates the PLE vector $p_l$ via a **State-Dependent Gate** $g_l$:

$$
g_l = \sigma(W_g h_l'') \in (0, 1)^{d_{\text{ple}}}
$$

$$
v_l = g_l \odot p_l \in \mathbb{R}^{d_{\text{ple}}}
$$

$$
u_l = W_{\text{out}} v_l \in \mathbb{R}^{d_{\text{model}}}
$$

$$
h_{l+1} = h_l'' + \text{RMSNorm}(u_l)
$$

---

### 6.3 Deep Dive: Why the Hadamard Product Is Structurally Mandatory

The Hadamard product (element-wise multiplication $[v_l]_i = [g_l]_i \cdot [p_l]_i$) provides three critical capabilities:

1. **Dimension-Wise Semantic Feature Selection:**
   Scalar gating ($g \in \mathbb{R}$) forces an all-or-nothing bottleneck. Individual embedding dimensions encode distinct latent factors (e.g., dim 14 = grammatical tense, dim 88 = domain, dim 302 = core entity). The Hadamard product gives the network independent continuous control over every single feature axis.
2. **Preservation of Latent Disentanglement:**
   Matrix multiplications ($W p_l$) blend orthogonal basis features across dimensions. The Hadamard product operates strictly along coordinate axes, allowing the gating network to filter noise without rotating the feature coordinate system.
3. **$O(d_{\text{ple}})$ Computational Complexity:**
   Performing a matrix multiplication would require $O(d_{\text{model}} \cdot d_{\text{ple}})$ FLOPs. The Hadamard product requires exactly $d_{\text{ple}}$ FLOPs, scaling parameter memory without adding compute overhead.

---

### 6.4 Information Retention & Semantic Disambiguation Dynamics

PLE balances new token lookups with historical context through state-conditioned gate saturation:

```text
Case Study: Disambiguating "The river bank overflowed"

Layer 2 (Syntax & Morphology):
  h_2'' is unresolved. Gate g_2 -> 1.0 (OPENS).
  Injects p_2 containing lexical syntax embeddings for "bank".

Layer 24 (Deep Contextual Reasoning):
  Attention has resolved "river" + "bank" -> [Geographical Shoreline].
  h_24'' is highly certain. Gate g_24 -> 0.0 (CLOSES).
  Suppresses p_24 lookup to prevent static dictionary definitions (e.g., financial bank)
  from corrupting the established semantic representation!
```

---

### 6.5 Structural Comparison: Standard Transformer vs. Gemma 4 PLE

| Dimension | Standard Transformer | Gemma 4 with Per-Layer Embeddings (PLE) |
| :--- | :--- | :--- |
| **Token Representation Ingestion** | Single lookup at input layer $l=0$ | Dynamic re-injection at every layer $l \in \{1, \dots, L\}$ |
| **Parameter vs. FLOP Scaling** | Strictly coupled ($\text{Params} \propto \text{FLOPs}$) | Decoupled ($\text{Capacity} \gg \text{Active Compute FLOPs}$) |
| **Deep Network Identity Retention** | Degrades as $L > 32$ due to residual dilution | Zero-FLOP identity re-anchoring at arbitrary depth |
| **Residual Structure** | Dual-Branch: $\text{Residual} + \text{Attn} + \text{FFN}$ | Triple-Branch: $\text{Residual} + \text{Attn} + \text{FFN} + \text{Gated PLE}$ |
| **Layer-Specific Specialization** | Universal embedding shared globally | Early layers learn syntax; deep layers learn abstract semantics |

---

## 7. Module VI: Cross-Disciplinary Synthesis & Engineering Playbook

### 7.1 Full Model Lifecycle Workflow

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                    END-TO-END LLM SYSTEMS LIFECYCLE                         │
│                                                                             │
│  [Phase 1: Architecture Selection]                                          │
│   - Pure Attention (Transformer) vs. Selective SSM (Mamba) vs. Hybrid (PLE) │
│                                                                             │
│  [Phase 2: Distributed Pre-Training]                                        │
│   - Precision: BF16 Mixed Precision with FP32 Master Weights                │
│   - 3D/4D Parallelism: Combine DDP/FSDP + TP (intra-node) + PP/CP           │
│   - Optimization: FlashAttention-3 + Fused Kernel Operators in SRAM         │
│                                                                             │
│  [Phase 3: Compression & Quantization]                                      │
│   - Moderate Compression: FP8 / INT8 PTQ with SmoothQuant / Outlier Clipping│
│   - High Compression: 4-Bit GPTQ with Cholesky Inverse Hessian              │
│   - Ultra-Low Bit / Edge: BitNet 1.58b QAT (Ternary Addition-Only Serving)  │
│                                                                             │
│  [Phase 4: High-Throughput Inference Serving]                               │
│   - Serving Engine: PagedAttention + Continuous Iteration-Level Batching    │
│   - Memory Sharing: Copy-on-Write (Parallel Sampling) + Prefix Caching      │
│   - Mamba Linear O(1) State vs. Transformer KV Cache Compression            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 7.2 Systems Engineering Decision Flowchart

```text
Target Deployment Constraints:
│
├── GPU Cluster Pre-Training (Large Scale > 50B)
│   ├── Intra-Node: Tensor Parallelism (TP = 8 over NVLink)
│   ├── Sequence > 32k: Enable Context Parallelism (Ring Attention)
│   ├── Inter-Node: ZeRO-3 / FSDP or Pipeline Parallelism (1F1B)
│   └── Numerical Format: BF16 / FP8 Mixed Precision
│
├── High-Concurrency Enterprise Serving (API Providers)
│   ├── Memory Management: PagedAttention (vLLM) with 16-token page size
│   ├── Scheduling: Continuous Iteration-Level Batching
│   ├── Redundant Prompts: Global Hash Prefix Caching (Prompt Caching)
│   └── Parallel Sampling: Copy-on-Write (CoW) page forking
│
├── Edge Device / Consumer Hardware Serving
│   ├── RAM < 8 GB: GGUF / K-Quants (4-bit or 2-bit offloading to CPU/Metal)
│   ├── Single GPU 24 GB: 4-Bit GPTQ or AWQ
│   └── Next-Gen Custom Silicon: BitNet 1.58b Ternary Kernels
│
└── Long-Context Streaming / Real-Time Generation
    ├── Unbounded Sequence Processing: Mamba (Selective SSM S6 Block)
    └── Standard Architecture with Zero Drift: Transformer + Gemma 4 PLE
```

---

### 7.3 Comprehensive Glossary of Symbols & Notation

| Symbol | Mathematical Definition | Domain / Architectural Role |
| :--- | :--- | :--- |
| $\mathbf{A}$ | State transition matrix in SSM ($\mathbb{R}^{N \times N}$) | Defines continuous-time internal state dynamics |
| $\mathbf{\bar{A}}, \mathbf{\bar{B}}$ | Discretized SSM state matrices | Computed via Zero-Order Hold (ZOH) with step size $\Delta$ |
| $\Delta$ | Discretization step size ($\mathbb{R}^+$) | Timescale resolution; in Mamba, acts as input-dependent gate |
| $\mathbf{\bar{K}}$ | SSM convolution kernel ($\mathbb{R}^L$) | Enables FFT-based parallel sequence training in LTI SSMs |
| $s$ | Quantization scale factor ($\mathbb{R}^+$) | Maps continuous interval to discrete integer grid |
| $z$ | Integer zero-point ($\mathbb{Z}$) | Offsets asymmetric quantized values |
| $H^{-1}$ | Inverse Hessian matrix ($2 X X^T + \lambda I)^{-1}$ | Second-order sensitivity matrix in GPTQ error redistribution |
| $\Phi$ | Total model parameter count | Base unit for distributed memory calculations |
| $p$ | Pipeline stages (GPUs in pipeline) | Determines pipeline bubble fraction $\frac{p-1}{p-1+m}$ |
| $m$ | Micro-batch count | Subdivisions of mini-batch for gradient accumulation |
| $e_0$ | Global base embedding ($\mathbb{R}^{d_{\text{base}}}$) | Universal token lookup vector in PLE |
| $e_{\text{lookup}}^{(l)}$ | Layer-specific embedding ($\mathbb{R}^{d_{\text{ple}}}$) | Depth-specialized lookup vector in PLE |
| $g_l$ | State-dependent gate vector ($\mathbb{R}^{d_{\text{ple}}}$) | Dimension-wise feature selector in PLE ($g_l = \sigma(W_g h_l'')$) |
| $\odot$ | Hadamard Product | Element-wise vector multiplication ($[u \odot v]_i = u_i \cdot v_i$) |
| $I$ | Arithmetic Intensity ($\text{FLOPs/Byte}$) | Governs compute-bound vs memory-bound hardware execution |
| $\text{TTFT}$ | Time to First Token (ms) | Latency of the prefill phase (compute-bound) |
| $\text{TPOT}$ | Time per Output Token (ms) | Latency of individual autoregressive decoding steps |
| $\text{MBU}$ | Model Bandwidth Utilization ($\%$) | Fraction of hardware theoretical memory bandwidth achieved |
| $\text{MFU}$ | Model FLOPs Utilization ($\%$) | Fraction of hardware peak theoretical compute achieved |
| $B_{\text{block}}$ | PagedAttention block size (tokens) | Unit of physical memory allocation in virtual KV paging |

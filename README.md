# Modern Large Language Model Architectures & Distributed Systems: The Complete Technical Compendium

**A Deep Mathematical, Systems-Level, and Architectural Synthesis of Quantization, State Space Models (Mamba), Distributed Parallelism (3D/4D & ZeRO), Inference Economics & PagedAttention, Rotary Position Embeddings (RoPE), and the Gemma 4 Frontier Architecture Suite**

---

## Table of Contents

- [1. Foundational Systems Perspective: The Compute & Memory Wall](#1-foundational-systems-perspective-the-compute--memory-wall)
  - [1.1 The Hardware Execution Model: SRAM vs. HBM Hierarchy](#11-the-hardware-execution-model-sram-vs-hbm-hierarchy)
  - [1.2 The Roofline Model: Mathematical Formulation of Arithmetic Intensity](#12-the-roofline-model-mathematical-formulation-of-arithmetic-intensity)
- [2. Module I: Numerical Precision, Arithmetic & Model Quantization](#2-module-i-numerical-precision-arithmetic--model-quantization)
  - [2.1 Data Types & IEEE-754 Binary Encodings](#21-data-types--ieee-754-binary-encodings)
  - [2.2 Linear Quantization Mechanics: Symmetric vs. Asymmetric](#22-linear-quantization-mechanics-symmetric-vs-asymmetric)
    - [2.2.1 Symmetric Quantization (Absmax)](#221-symmetric-quantization-absmax)
    - [2.2.2 Asymmetric Quantization (Zero-Point)](#222-asymmetric-quantization-zero-point)
  - [2.3 Outliers, Dynamic Range Clipping & Loss Optimization](#23-outliers-dynamic-range-clipping--loss-optimization)
    - [2.3.1 Mean Squared Error (MSE) Optimization](#231-mean-squared-error-mse-optimization)
    - [2.3.2 Kullback-Leibler (KL) Divergence Calibration](#232-kullback-leibler-kl-divergence-calibration)
  - [2.4 Weight vs. Activation Quantization (Static vs. Dynamic Calibration)](#24-weight-vs-activation-quantization-static-vs-dynamic-calibration)
  - [2.5 Advanced 4-Bit Post-Training Quantization (PTQ)](#25-advanced-4-bit-post-training-quantization-ptq)
    - [2.5.1 GPTQ: Second-Order Taylor Expansion & Inverse Hessian Compensation](#251-gptq-second-order-taylor-expansion--inverse-hessian-compensation)
    - [2.5.2 GGUF & K-Quants: Hierarchical Block-Wise Scaling](#252-gguf--k-quants-hierarchical-block-wise-scaling)
  - [2.6 Quantization-Aware Training (QAT) & Loss Landscapes](#26-quantization-aware-training-qat--loss-landscapes)
    - [2.6.1 Straight-Through Estimator (STE) Derivation](#261-straight-through-estimator-ste-derivation)
    - [2.6.2 Loss Landscape Basin Geometry](#262-loss-landscape-basin-geometry)
  - [2.7 Sub-2-Bit and Extreme Low-Bit Paradigms](#27-sub-2-bit-and-extreme-low-bit-paradigms)
    - [2.7.1 BitNet 1-Bit (BitLinear)](#271-bitnet-1-bit-bitlinear)
    - [2.7.2 BitNet 1.58b: Ternary Quantization & Addition-Only Inference](#272-bitnet-158b-ternary-quantization--addition-only-inference)
  - [2.8 Quantization Comparison Matrix](#28-quantization-comparison-matrix)
- [3. Module II: Sequence Modeling Paradigms — Transformers, RNNs & State Space Models (Mamba)](#3-module-ii-sequence-modeling-paradigms--transformers-rnns--state-space-models-mamba)
  - [3.1 The Sequence Modeling Trilemma](#31-the-sequence-modeling-trilemma)
  - [3.2 Continuous-Time State Space Models (SSMs)](#32-continuous-time-state-space-models-ssms)
  - [3.3 Exact Discretization via Zero-Order Hold (ZOH)](#33-exact-discretization-via-zero-order-hold-zoh)
  - [3.4 The Dual Representation of Linear Time-Invariant (LTI) SSMs](#34-the-dual-representation-of-linear-time-invariant-lti-ssms)
    - [3.4.1 Recurrent Representation (Linear Inference Complexity)](#341-recurrent-representation-linear-inference-complexity)
    - [3.4.2 Convolutional Representation (Parallel Training via FFT)](#342-convolutional-representation-parallel-training-via-fft)
  - [3.5 Long-Range Memory & The HiPPO Matrix](#35-long-range-memory--the-hippo-matrix)
  - [3.6 The Fundamental Failure Mode of LTI SSMs: Content Invariance](#36-the-fundamental-failure-mode-of-lti-ssms-content-invariance)
  - [3.7 Mamba: Selective State Spaces (S6)](#37-mamba-selective-state-spaces-s6)
    - [3.7.1 Mathematical Formulation of Selective Parameterization](#371-mathematical-formulation-of-selective-parameterization)
    - [3.7.2 Physical Intuition of Step-Size Delta](#372-physical-intuition-of-step-size-delta)
  - [3.8 Systems Innovations for Selective SSMs](#38-systems-innovations-for-selective-ssms)
    - [3.8.1 The Parallel Associative Scan (Blelloch Algorithm)](#381-the-parallel-associative-scan-blelloch-algorithm)
    - [3.8.2 Hardware-Aware Memory Hierarchy & Kernel Fusion](#382-hardware-aware-memory-hierarchy--kernel-fusion)
    - [3.8.3 Activation Recomputation in the Backward Pass](#383-activation-recomputation-in-the-backward-pass)
  - [3.9 The Mamba Block Architecture vs. Transformer Decoder](#39-the-mamba-block-architecture-vs-transformer-decoder)
- [4. Module III: Distributed Training, Memory Layouts & Systems Parallelism](#4-module-iii-distributed-training-memory-layouts--systems-parallelism)
  - [4.1 Anatomy of GPU Memory in Large Language Models](#41-anatomy-of-gpu-memory-in-large-language-models)
    - [4.1.1 The 16 Bytes per Parameter Rule in Mixed-Precision Adam](#411-the-16-bytes-per-parameter-rule-in-mixed-precision-adam)
    - [4.1.2 Mathematical Proof: Why FP32 Master Weights & Optimizer States Are Required](#412-mathematical-proof-why-fp32-master-weights--optimizer-states-are-required)
  - [4.2 Data Parallelism (DDP) & Ring All-Reduce Proofs](#42-data-parallelism-ddp--ring-all-reduce-proofs)
  - [4.3 Tensor Parallelism (Megatron-LM Intra-Layer Slicing)](#43-tensor-parallelism-megatron-lm-intra-layer-slicing)
    - [4.3.1 Slicing the Multi-Layer Perceptron (MLP) Block](#431-slicing-the-multi-layer-perceptron-mlp-block)
    - [4.3.2 Slicing Multi-Head Self-Attention (MHSA)](#432-slicing-multi-head-self-attention-mhsa)
  - [4.4 Pipeline Parallelism (PP) & Inter-Layer Scheduling](#44-pipeline-parallelism-pp--inter-layer-scheduling)
    - [4.4.1 The Pipeline Bubble: Exact Derivation](#441-the-pipeline-bubble-exact-derivation)
    - [4.4.2 Mathematical Equivalence of Micro-Batch Gradient Accumulation](#442-mathematical-equivalence-of-micro-batch-gradient-accumulation)
    - [4.4.3 GPipe vs. 1F1B Schedule Activation Footprints](#443-gpipe-vs-1f1b-schedule-activation-footprints)
  - [4.5 Context Parallelism (CP) & Ring Attention](#45-context-parallelism-cp--ring-attention)
    - [4.5.1 The Ring Communication Topology](#451-the-ring-communication-topology)
    - [4.5.2 Online Numerically Stable Softmax Accumulation Proof](#452-online-numerically-stable-softmax-accumulation-proof)
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
    - [5.5.2 Inflection Point: The Knee Batch Size Formula](#552-inflection-point-the-knee-batch-size-formula)
    - [5.5.3 Throughput vs. Per-User Latency (TPOT) Trade-Off](#553-throughput-vs-per-user-latency-tpot-trade-off)
    - [5.5.4 Model Bandwidth Utilization (MBU) vs. Model FLOPs Utilization (MFU)](#554-model-bandwidth-utilization-mbu-vs-model-flops-utilization-mfu)
  - [5.6 PagedAttention: OS-Inspired Virtual Memory for KV Cache (vLLM)](#56-pagedattention-os-inspired-virtual-memory-for-kv-cache-vllm)
    - [5.6.1 The Memory Fragmentation Crisis](#561-the-memory-fragmentation-crisis)
    - [5.6.2 Block Tables & Virtual Paging Architecture](#562-block-tables--virtual-paging-architecture)
    - [5.6.3 Copy-on-Write (CoW) Forking & Parallel Sampling](#563-copy-on-write-cow-forking--parallel-sampling)
    - [5.6.4 Prefix Caching (Prompt Caching)](#564-prefix-caching-prompt-caching)
  - [5.7 Continuous (Iteration-Level) Batching vs. Static Batching](#57-continuous-iteration-level-batching-vs-static-batching)
- [6. Module V: Positional Encodings & Rotary Position Embedding (RoPE)](#6-module-v-positional-encodings--rotary-position-embedding-rope)
  - [6.1 The Fundamental Architectural Principle: Vectors vs. Weights](#61-the-fundamental-architectural-principle-vectors-vs-weights)
  - [6.2 The Block-Diagonal 2D Rotation Matrix Formulation](#62-the-block-diagonal-2d-rotation-matrix-formulation)
  - [6.3 Step-by-Step Numerical Example (d = 6 at Position m = 1)](#63-step-by-step-numerical-example-d--6-at-position-m--1)
  - [6.4 Complex Representation & Mathematical Derivation via Euler's Formula](#64-complex-representation--mathematical-derivation-via-eulers-formula)
  - [6.5 Mathematical Proof: Why RoPE Must Be Applied to Both Q and K](#65-mathematical-proof-why-rope-must-be-applied-to-both-q-and-k)
  - [6.6 Hardware-Efficient Implementation via Element-Wise Vector Operations](#66-hardware-efficient-implementation-via-element-wise-vector-operations)
- [7. Module VI: Modern Frontier Topologies — The Gemma 4 Architecture Suite](#7-module-vi-modern-frontier-topologies--the-gemma-4-architecture-suite)
  - [7.1 The Gemma 4 Family Taxonomy (E2B, E4B, 31B Dense, 26B A4B MoE)](#71-the-gemma-4-family-taxonomy-e2b-e4b-31b-dense-26b-a4b-moe)
  - [7.2 Interleaved Local Sliding Window & Global Attention (K=V Fusion)](#72-interleaved-local-sliding-window--global-attention-kv-fusion)
  - [7.3 Partitioned / Low-Frequency-Pruned RoPE (p-RoPE)](#73-partitioned--low-frequency-pruned-rope-p-rope)
  - [7.4 Mixture of Experts (MoE) Architecture in Gemma 4 26B A4B](#74-mixture-of-experts-moe-architecture-in-gemma-4-26b-a4b)
  - [7.5 Per-Layer Embeddings (PLE) in Gemma 4 E2B & E4B](#75-per-layer-embeddings-ple-in-gemma-4-e2b--e4b)
    - [7.5.1 Motivation: Decoupling Capacity from FLOPs](#751-motivation-decoupling-capacity-from-flops)
    - [7.5.2 Mathematical Formulation & Forward Pass](#752-mathematical-formulation--forward-pass)
    - [7.5.3 Why the Hadamard Product Is Structurally Mandatory](#753-why-the-hadamard-product-is-structurally-mandatory)
    - [7.5.4 Information Retention & Semantic Disambiguation Dynamics](#754-information-retention--semantic-disambiguation-dynamics)
  - [7.6 Multi-Token Prediction (MTP) Speculative Drafters](#76-multi-token-prediction-mtp-speculative-drafters)
- [8. Module VII: Cross-Disciplinary Synthesis & Engineering Playbook](#8-module-vii-cross-disciplinary-synthesis--engineering-playbook)
  - [8.1 Full Model Lifecycle Workflow](#81-full-model-lifecycle-workflow)
  - [8.2 Systems Engineering Decision Flowchart](#82-systems-engineering-decision-flowchart)
  - [8.3 Comprehensive Glossary of Symbols & Notation](#83-comprehensive-glossary-of-symbols--notation)
  - [8.4 References & Primary Sources](#84-references--primary-sources)

---

## 1. Foundational Systems Perspective: The Compute & Memory Wall

Modern deep learning performance is strictly governed by computer hardware architecture. To understand why models execute at specific speeds, we analyze algorithms through both asymptotic computational complexity ($O(N)$) and physical data movement constraints across the GPU memory hierarchy.

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

Modern GPU accelerators (e.g., NVIDIA A100, H100, H200, B200) contain two primary storage domains:

1. **On-Chip Registers & SRAM (L1 Cache):**
   - Located directly inside each Streaming Multiprocessor (SM).
   - Massive aggregate bandwidth ($>15\text{ TB/s} - 30\text{ TB/s}$).
   - Extremely limited storage capacity ($<256\text{ KB}$ per SM, $\approx 30 - 50\text{ MB}$ total across the entire chip).
2. **Off-Chip High Bandwidth Memory (HBM3 / Main VRAM):**
   - High storage capacity ($80\text{ GB} - 141\text{ GB}$).
   - Bandwidth is an order of magnitude slower than SRAM ($2.0 - 4.8\text{ TB/s}$).
3. **Interconnects (NVLink vs. PCIe & Network Fabrics):**
   - Intra-node GPU-to-GPU: NVLink provides $900\text{ GB/s}$ bidirectional bandwidth per GPU.
   - Inter-node cluster network: InfiniBand NDR provides $50\text{ GB/s}$ ($400\text{ Gbps}$) per link.

---

### 1.2 The Roofline Model: Mathematical Formulation of Arithmetic Intensity

The attainable execution throughput $P$ (in TFLOPs/second) of any computational kernel is strictly bounded by the **Roofline Equation**:

$$
I = \frac{\text{Floating Point Operations (FLOPs)}}{\text{Memory Access (Bytes transferred from HBM)}}
$$

$$
P = \min\left(P_{\text{peak}}, \; I \times \text{Bandwidth}_{\text{HBM}}\right)
$$

```text
Attainable Performance P (TFLOPs)
   ^
   |                      /---------------- Peak Compute Limit (P_peak)
   |                     /
   |                    /
   |                   /  Compute-Bound Regime (e.g., Prefill Phase, Large GEMMs)
   |                  /
   |                 /
   |                / Memory-Bound Regime (e.g., Single-token Decoding, Softmax, RMSNorm)
   |               /
   |              /
   |             /
   |------------/-------------------------------------------------------->
   0           I_knee                               Arithmetic Intensity I (FLOPs/Byte)
```

The threshold between memory-bound and compute-bound regimes is the **Knee Arithmetic Intensity** ($I_{\text{knee}}$):

$$
I_{\text{knee}} = \frac{P_{\text{peak}}}{\text{Bandwidth}_{\text{HBM}}}
$$

For an NVIDIA H100 SXM GPU with peak compute of 989 TFLOPs (BF16 Tensor Core) and HBM bandwidth of 3.35 TB/s:

$$
I_{\text{knee}} = \frac{989 \times 10^{12}\text{ FLOPs/s}}{3.35 \times 10^{12}\text{ Bytes/s}} \approx \mathbf{295.2\text{ FLOPs/Byte}}
$$

- **Memory-Bound Regime (I < 295.2 FLOPs/Byte):** Any kernel that performs fewer than 295.2 calculations per byte loaded from memory (e.g., LayerNorm, Softmax, single-token autoregressive decoding) leaves Tensor Cores idle while waiting on memory buses.
- **Compute-Bound Regime (I > 295.2 FLOPs/Byte):** Kernels with high data reuse (e.g., large GEMMs during prompt prefill) fully saturate Tensor Core compute units.

---

## 2. Module I: Numerical Precision, Arithmetic & Model Quantization

Model quantization compresses high-precision tensors (FP32, FP16, BF16) into low-bit integers (INT8, INT4) or discrete ternary states {-1, 0, +1}. This reduces memory consumption and memory bandwidth requirements by 2x to 8x.

### 2.1 Data Types & IEEE-754 Binary Encodings

Under the IEEE-754 standard and modern deep learning formats, continuous floating-point numbers are decomposed into three bitfields: Sign (s), Exponent (e), and Mantissa/Fraction (m):

$$
\text{Value} = (-1)^s \times 2^{e - \text{bias}} \times \left(1 + \sum_{i=1}^{M} m_i 2^{-i}\right)
$$

Where the exponent bias is defined as $\text{bias} = 2^{E-1} - 1$ for an $E$-bit exponent.

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

| Format | Total Bits | Exponent Bits | Mantissa Bits | Dynamic Range | Machine Epsilon ($\epsilon$) | Primary Systems Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** | 32 | 8 | 23 | $\approx 10^{\pm 38}$ | $2^{-23} \approx 1.19 \times 10^{-7}$ | Master weights, optimizer accumulators |
| **FP16** | 16 | 5 | 10 | $\approx [-6.5 \times 10^4, 6.5 \times 10^4]$ | $2^{-10} \approx 9.77 \times 10^{-4}$ | Legacy mixed-precision training |
| **BF16** | 16 | 8 | 7 | $\approx 10^{\pm 38}$ | $2^{-7} \approx 7.81 \times 10^{-3}$ | Modern LLM standard training |
| **FP8 (E4M3)** | 8 | 4 | 3 | $\approx [-448, 448]$ | $2^{-3} = 0.125$ | Forward pass activations & weights |
| **FP8 (E5M2)** | 8 | 5 | 2 | $\approx [-5.7 \times 10^4, 5.7 \times 10^4]$ | $2^{-2} = 0.25$ | Backward pass gradient propagation |
| **INT8** | 8 | 0 | 7 | $[-128, 127]$ | Constant step $\Delta = 1$ | High-throughput integer inference |
| **INT4** | 4 | 0 | 3 | $[-8, 7]$ | Constant step $\Delta = 1$ | Weight-only PTQ (GPTQ, AWQ) |
| **Ternary** | 1.58 | 0 | 0 | $\{-1, 0, +1\}$ | Discrete alphabet | BitNet addition-only inference |

---

### 2.2 Linear Quantization Mechanics: Symmetric vs. Asymmetric

Linear quantization maps a continuous floating-point interval $[\alpha, \beta] \subset \mathbb{R}$ to a discrete grid of integers $[q_{\min}, q_{\max}] \subset \mathbb{Z}$.

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

In **symmetric quantization**, the floating-point origin 0.0 maps directly to the integer 0. The zero-point is fixed at z = 0. The dynamic range is symmetric around zero, spanning from negative alpha to positive alpha, where alpha equals the maximum absolute value of the input tensor.

1. **Scale Factor:**

   The scale factor s maps the floating-point range to the integer grid:

   $$
   s = \frac{q_{\max}}{\alpha} = \frac{2^{b-1} - 1}{\max(|x|)}
   $$

2. **Quantization Mapping:**

   Each floating-point value x is scaled, rounded, and clamped to the integer range:

   $$
   x_q = \text{clamp}\left(\text{round}(s \cdot x), \; -q_{\max}, \; q_{\max}\right)
   $$

3. **Dequantization Reconstruction:**

   The approximate floating-point value is recovered by dividing the quantized integer by the scale:

   $$
   \hat{x} = \frac{x_q}{s}
   $$

4. **Maximum Quantization Error:**

   The worst-case reconstruction error is bounded by half the quantization step size:

   $$
   \epsilon_{\max} = |x - \hat{x}| \le \frac{1}{2s}
   $$

#### 2.2.2 Asymmetric Quantization (Zero-Point)

When data distributions are skewed (e.g., post-ReLU or post-GeLU activations where all values are non-negative), symmetric quantization wastes half the integer representation domain. **Asymmetric quantization** introduces an integer **Zero-Point** z that shifts the mapping to cover only the actual data range.

1. **Scale Factor:**

   The scale factor maps the full data range to the full integer range:

   $$
   s = \frac{q_{\max} - q_{\min}}{\max(x) - \min(x)} = \frac{2^b - 1}{\alpha - \beta}
   $$

2. **Zero-Point Calculation:**

   The zero-point z is an integer offset that aligns the floating-point minimum to the integer minimum:

   $$
   z = \text{clamp}\left(\text{round}(-\min(x) \cdot s) + q_{\min}, \; q_{\min}, \; q_{\max}\right)
   $$

3. **Quantization Mapping:**

   Each value is scaled and shifted by the zero-point before clamping:

   $$
   x_q = \text{clamp}\left(\text{round}(s \cdot x) + z, \; q_{\min}, \; q_{\max}\right)
   $$

4. **Dequantization Reconstruction:**

   The approximate floating-point value is recovered by subtracting the zero-point and dividing by the scale:

   $$
   \hat{x} = \frac{x_q - z}{s}
   $$

```text
Step-by-Step Numerical Example (FP32 -> INT8):
Input Array: x = [-6.0, -2.0, 0.0, 1.5, 4.0]

[Symmetric Absmax]:
  alpha = max(|x|) = 6.0
  s = 127 / 6.0 = 21.1667
  x_q = round(x * s) = [-127, -42, 0, 32, 85]
  x_hat = x_q / s = [-6.000, -1.984, 0.000, 1.512, 4.016]

[Asymmetric Zero-Point]:
  min = -6.0, max = 4.0, range = 10.0
  s = 255 / 10.0 = 25.5
  z = round(-(-6.0) * 25.5) - 128 = 153 - 128 = 25
  x_q = round(x * s) + z = [-153+25, -51+25, 0+25, 38+25, 102+25] = [-128, -26, 25, 63, 127]
  x_hat = (x_q - z) / s = [-6.000, -2.000, 0.000, 1.490, 4.000]
```

---

### 2.3 Outliers, Dynamic Range Clipping & Loss Optimization

In transformer models at scale (above 6.7B parameters), activation distributions exhibit **emergent systematic channel outliers** with magnitudes 20x to 100x larger than normal values. If the scale factor s is computed strictly from the maximum absolute value, all normal values collapse into a few integer bins.

#### 2.3.1 Mean Squared Error (MSE) Optimization

MSE calibration finds the optimal clipping threshold that minimizes the reconstruction error. Rather than using the raw maximum, the threshold alpha is tuned to balance clipping loss against quantization granularity:

$$
\alpha^{\ast} = \arg\min_{\alpha} \frac{1}{N} \sum_{i=1}^{N} \left( x_i - \frac{\text{clamp}\left(\text{round}\left(\frac{q_{\max}}{\alpha} x_i\right), -q_{\max}, q_{\max}\right)}{\frac{q_{\max}}{\alpha}} \right)^2
$$

#### 2.3.2 Kullback-Leibler (KL) Divergence Calibration

KL divergence treats the original continuous tensor histogram P and the reconstructed quantized histogram Q as probability distributions, minimizing information loss:

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
│   Input X ──► [Compute scale on-the-fly] ──► [Quantize X] ──► [MatMul]      │
│   - Scale computed at runtime per token/layer.                              │
│   - Higher accuracy, but adds minor reduction overhead.                     │
│                                                                             │
│  [Static Quantization]                                                      │
│   Calibration Data ──► [Offline Profiling] ──► [Precomputed scale in Graph] │
│   Input X ──► [Quantize with Precomputed scale] ──► [MatMul]                │
│   - Scale is frozen; zero runtime calibration latency.                      │
│   - Sensitive to out-of-distribution input shifts.                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 2.5 Advanced 4-Bit Post-Training Quantization (PTQ)

#### 2.5.1 GPTQ: Second-Order Taylor Expansion & Inverse Hessian Compensation

GPTQ quantizes weight matrices column-by-column while continuously compensating unquantized weights for the introduced quantization error.

1. **Objective Function:**

   GPTQ seeks quantized weights that minimize the output reconstruction error:

   $$
   \min_{\hat{W}} \| W X - \hat{W} X \|_2^2
   $$

2. **Second-Order Taylor Series Expansion:**

   Let the quantization perturbation be the difference between quantized and original weights. Expanding the loss function around pre-trained weights W using a second-order Taylor series:

   $$
   \Delta \mathcal{L} \approx (\nabla_W \mathcal{L})^T \Delta W + \frac{1}{2} \Delta W^T H \Delta W
   $$

   Since the pre-trained weights are already at a local minimum, the first-order gradient vanishes. The Hessian matrix H with respect to the weights is:

   $$
   H = 2 X X^T
   $$

3. **Optimal Weight Compensation via Inverse Hessian:**

   When column q is quantized, the quantization error is the difference between the original and quantized column. To minimize total error, all remaining unquantized columns j > q are updated by:

   $$
   W_j \leftarrow W_j - \frac{E_q \cdot [H^{-1}]_{q, j}}{[H^{-1}]_{q, q}}
   $$

4. **Cholesky Decomposition:**

   Computing full matrix inverses repeatedly is computationally unstable. GPTQ factorizes the inverse Hessian using Cholesky decomposition:

   $$
   H^{-1} = L L^T
   $$

   A small diagonal damping term is added to H to guarantee positive-definiteness and numerical stability. The damping coefficient is typically set to 1% of the mean diagonal entry of H.

#### 2.5.2 GGUF & K-Quants: Hierarchical Block-Wise Scaling

GGUF divides weight tensors into hierarchical **super-blocks** (256 weights) and **sub-blocks** (16 or 32 weights):

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

- Sub-block scales $s_i$ are quantized relative to the super-scale $S$, providing low quantization error while compressing metadata overhead.

---

### 2.6 Quantization-Aware Training (QAT) & Loss Landscapes

#### 2.6.1 Straight-Through Estimator (STE) Derivation

The derivative of the rounding operation $\text{round}(x)$ is zero almost everywhere and undefined at integers:

$$
\frac{d}{dx} \text{round}(x) = 0 \quad \forall x \notin \mathbb{Z} + 0.5
$$

To permit gradient descent, the **Straight-Through Estimator (STE)** replaces the true rounding gradient with the identity operator during the backward pass:

$$
\text{Forward: } x_q = \text{round}(x), \qquad \text{Backward: } \frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial x_q} \cdot \mathbf{1}_{|x| \le \text{clip}}
$$

#### 2.6.2 Loss Landscape Basin Geometry

```text
Loss L
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

- **PTQ:** Frequently converges to sharp local basins in FP32 space. Discrete rounding displaces the parameters outside the basin, triggering loss spikes.
- **QAT:** Guides parameter trajectories into broad, flat minima where discrete perturbation preserves near-optimal task performance.

---

### 2.7 Sub-2-Bit and Extreme Low-Bit Paradigms

#### 2.7.1 BitNet 1-Bit (BitLinear)

BitNet replaces standard `nn.Linear` layers with `BitLinear`, mapping weights to 1-bit binary values $\{-1, +1\}$ and activations to INT8:

$$
W_{\text{bin}} = \text{Sign}(W - \bar{W}), \qquad \text{where } \text{Sign}(x) = \begin{cases} +1, & x \ge 0 \\ -1, & x < 0 \end{cases}
$$

$$
\beta = \frac{1}{nm} \|W\|_1 = \frac{1}{nm}\sum_{i=1}^n \sum_{j=1}^m |W_{i,j}|
$$

$$
X_{\text{quant}} = \text{clamp}\left(\text{round}\left( \frac{127}{\alpha} X \right), -128, 127\right), \qquad \alpha = \|X\|_{\infty}
$$

$$
Y = \left( X_{\text{quant}} \times W_{\text{bin}} \right) \times \frac{\alpha \cdot \beta}{127}
$$

#### 2.7.2 BitNet 1.58b: Ternary Quantization & Addition-Only Inference

BitNet 1.58b introduces zero ($0$) into the weight representation, creating a ternary alphabet $\{-1, 0, +1\}$:

$$
\text{Information Capacity per Parameter} = \log_2(3) \approx \mathbf{1.585\text{ bits}}
$$

1. **Weight Quantization (`absmean`):**
   $$
   W_{\text{ternary}} = \text{clamp}\left( \text{round}\left( \frac{W}{\gamma + \epsilon} \right), -1, +1 \right), \qquad \gamma = \frac{1}{nm} \sum_{i,j} |W_{i,j}|
   $$

2. **Activation Quantization:**
   $$
   X_{\text{quant}} = \text{clamp}\left( \text{round}\left( \frac{X \cdot 127}{\|X\|_{\infty}} \right), -128, 127 \right)
   $$

3. **Silicon Execution Elimination:**
   $$
   y_i = \sum_{j=1}^{d} W_{i,j} x_j = \sum_{j: W_{i,j}=+1} x_j - \sum_{j: W_{i,j}=-1} x_j
   $$
   This eliminates floating-point multiplication units (MACs) entirely from the silicon datapath, replacing them with integer addition and subtraction.

---

### 2.8 Quantization Comparison Matrix

| Method | Weight Bits | Activation Bits | Multiply-Accumulate Operations | Accuracy Retention ($>7\text{B}$) | Primary Bottleneck | Key Systems Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP16 / BF16** | 16 | 16 | FP16/BF16 Tensor Core | 100% (Baseline) | Memory capacity & bandwidth | Pre-training, full precision |
| **FP8 (E4M3)** | 8 | 8 | FP8 Tensor Core | $>99.5\%$ | Dynamic range calibration | High-throughput H100 pre-training |
| **INT8 (PTQ)** | 8 | 8 | INT8 Tensor Core | $>99.0\%$ | Activation outlier handling | Enterprise server inference |
| **GPTQ (PTQ)** | 4 | 16 | INT4 Dequant + FP16 GEMM | $>98.0\%$ | Calibration distribution shift | GPU VRAM-constrained serving |
| **GGUF (K-Quants)** | 2-6 | 16 | Block-Dequantized GEMV | $>95.0-99\%$ | CPU memory bandwidth | Consumer hardware, CPU/GPU offload |
| **BitNet 1.0** | 1 | 8 | INT8 Add/Subtract | Medium | High-entropy representational limits | Ultra-low power edge devices |
| **BitNet 1.58b** | 1.58 | 8 | INT8 Addition Only | Parity with FP16 at scale | Specialized ternary ASIC hardware | Next-gen energy-efficient inference |

---

## 3. Module II: Sequence Modeling Paradigms — Transformers, RNNs & State Space Models (Mamba)

### 3.1 The Sequence Modeling Trilemma

Sequence models navigate fundamental tradeoffs between three competing objectives:

1. **Constant Step-Wise Inference ($O(1)$ Complexity):** Generating token $t+1$ requires constant compute and memory, invariant to context length $L$.
2. **Parallelizable Training:** Processing training sequences of length $L$ scales concurrently as $O(1)$ depth across GPU cores.
3. **Uncompressed Contextual Recall:** The architecture models long-range dependencies without information loss.

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

| Architecture | Training Parallelism | Step-Wise Inference Compute | KV Cache Context Memory Footprint | Long-Context Modeling Fidelity |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Transformer** | O(1) Depth (Parallel Attention) | O(L) Compute per step | O(L) KV Cache (Grows linearly) | Exceptional (Raw history access) |
| **Classic RNN / LSTM** | O(L) Sequential (No parallelization) | O(1) Compute per step | O(1) Hidden vector h_t | Poor (Contextual forgetting) |
| **LTI SSMs (S4 / LSSL)** | O(L log L) (Parallel FFT Conv) | O(1) Compute per step | O(1) Hidden vector h_t | Moderate (Fails content filtering) |
| **Selective SSM (Mamba)** | O(L) (Parallel Associative Scan) | O(1) Compute per step | O(1) Hidden vector h_t | Exceptional (Matches Transformers) |

---

### 3.2 Continuous-Time State Space Models (SSMs)

Continuous SSMs map a 1D continuous input signal x(t) to an output signal y(t) through an N-dimensional latent state variable h(t):

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

The analytical solution to this continuous system for an initial condition $h(0)$ is:

$$
h(t) = \exp(\mathbf{A} t) h(0) + \int_{0}^{t} \exp(\mathbf{A}(t - \tau)) \mathbf{B} x(\tau) d\tau
$$

---

### 3.3 Exact Discretization via Zero-Order Hold (ZOH)

Digital systems operate on discrete token sequences $[x_0, x_1, \dots, x_L]$. Under the **Zero-Order Hold (ZOH)** assumption, the continuous input $x(t)$ remains constant across the sample interval $[k\Delta, (k+1)\Delta)$:

$$
x(t) = x_k \quad \forall t \in [k\Delta, (k+1)\Delta)
$$

Integrating the state differential equation over the discrete interval $\Delta$:

$$
h_k = h(k\Delta) = \exp(\Delta \mathbf{A}) h_{k-1} + \left( \int_{0}^{\Delta} \exp(\tau \mathbf{A}) d\tau \cdot \mathbf{B} \right) x_k
$$

Let $\mathbf{\bar{A}}$ and $\mathbf{\bar{B}}$ denote the discretized state matrices:

$$
\mathbf{\bar{A}} = \exp(\Delta \mathbf{A})
$$

$$
\mathbf{\bar{B}} = (\Delta \mathbf{A})^{-1} \left( \exp(\Delta \mathbf{A}) - \mathbf{I} \right) \cdot (\Delta \mathbf{B})
$$

Using a first-order Taylor series approximation for small $\Delta$:

$$
\mathbf{\bar{B}} \approx \Delta \mathbf{B}
$$

$$
\text{Discrete SSM Recurrence: } \begin{cases} h_k = \mathbf{\bar{A}} h_{k-1} + \mathbf{\bar{B}} x_k \\ y_k = \mathbf{C} h_k \end{cases}
$$

---

### 3.4 The Dual Representation of Linear Time-Invariant (LTI) SSMs

When $\mathbf{A}, \mathbf{B}, \mathbf{C}, \Delta$ are constant over time (Linear Time-Invariant), the system exhibits mathematical duality:

#### 3.4.1 Recurrent Representation (Linear Inference Complexity)

$$
h_k = \mathbf{\bar{A}} h_{k-1} + \mathbf{\bar{B}} x_k, \qquad y_k = \mathbf{C} h_k
$$

Generating token $k+1$ requires only loading the hidden state $h_k \in \mathbb{R}^N$, providing strict $O(1)$ compute and memory complexity per step.

#### 3.4.2 Convolutional Representation (Parallel Training via FFT)

Expanding the discrete recurrence explicitly from timestep $0$ to $L$:

$$
y_k = \sum_{j=0}^{k} \mathbf{C} \mathbf{\bar{A}}^{k-j} \mathbf{\bar{B}} x_j
$$

This is a discrete 1D convolution with the SSM kernel $\mathbf{\bar{K}} \in \mathbb{R}^L$:

$$
\mathbf{\bar{K}} = \left( \mathbf{C}\mathbf{\bar{B}}, \mathbf{C}\mathbf{\bar{A}}\mathbf{\bar{B}}, \mathbf{C}\mathbf{\bar{A}}^2\mathbf{\bar{B}}, \dots, \mathbf{C}\mathbf{\bar{A}}^{L-1}\mathbf{\bar{B}} \right)
$$

$$
y = x * \mathbf{\bar{K}}
$$

By the **Convolution Theorem**, computing $y = x * \mathbf{\bar{K}}$ via the Fast Fourier Transform (FFT) requires only $O(L \log L)$ time and is executed in parallel across GPU threads during pre-training:

$$
y = \text{IFFT}\left( \text{FFT}(x) \cdot \text{FFT}(\mathbf{\bar{K}}) \right)
$$

---

### 3.5 Long-Range Memory & The HiPPO Matrix

If matrix $\mathbf{A}$ is initialized randomly, the recurrent power $\mathbf{\bar{A}}^k$ suffers from vanishing or exploding dynamics.

**HiPPO (High-order Polynomial Projection Operators)** initializes $\mathbf{A}$ such that the hidden state $h(t)$ maintains an optimal continuous projection of history onto shifted Legendre polynomials:

$$
\mathbf{A}_{\text{HiPPO}}(n, k) = \begin{cases} (2n + 1)^{1/2} (2k + 1)^{1/2}, & \text{if } n > k \\ n + 1, & \text{if } n = k \\ -(2n + 1)^{1/2} (2k + 1)^{1/2}, & \text{if } n < k \end{cases}
$$

$$
\mathbf{B}_{\text{HiPPO}}(n) = (2n + 1)^{1/2}
$$

---

### 3.6 The Fundamental Failure Mode of LTI SSMs: Content Invariance

Because LTI matrices $\mathbf{A}, \mathbf{B}, \mathbf{C}, \Delta$ are static and independent of input content, LTI SSMs cannot dynamically adjust memory flow based on token identity.

```text
Task 1: Selective Copying
Input:   [BOS]  dog  [PAD]  [PAD]  cat  [PAD]  [EOS]
LTI SSM: Weights every token EQUALLY via static A, B. Cannot ignore [PAD]!

Task 2: Induction Heads
Input:   Harry ... Potter ... Harry ──► Target: [Potter]
LTI SSM: Cannot trigger dynamic memory recall conditioned on seeing "Harry" again.
```

---

### 3.7 Mamba: Selective State Spaces (S6)

Mamba solves content invariance by parameterizing $\mathbf{B}, \mathbf{C}$, and the discretization step size $\Delta$ as **functions of the current input token $x$**:

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

For an input tensor $x \in \mathbb{R}^{B \times L \times D}$:

$$
\mathbf{B}_k = W_B x_k \in \mathbb{R}^{B \times L \times N}
$$

$$
\mathbf{C}_k = W_C x_k \in \mathbb{R}^{B \times L \times N}
$$

$$
\Delta_k = \text{Softplus}\left( \text{Parameter}_{\Delta} + W_{\Delta} x_k \right) \in \mathbb{R}^{B \times L \times D}
$$

$$
\mathbf{\bar{A}}_k = \exp\left( \Delta_k \mathbf{A} \right) \in \mathbb{R}^{B \times L \times D \times N}
$$

$$
\mathbf{\bar{B}}_k = \Delta_k \mathbf{B}_k \in \mathbb{R}^{B \times L \times D \times N}
$$

$$
\text{Selective Recurrence: } h_k = \mathbf{\bar{A}}_k h_{k-1} + \mathbf{\bar{B}}_k x_k, \qquad y_k = \mathbf{C}_k h_k
$$

#### 3.7.2 Physical Intuition of Step-Size Delta

- **Large Delta (approaches infinity):** The discretized matrix A_bar approaches zero, while B_bar becomes large. The model **resets** old history and writes current token x_k into memory.
- **Small Delta (approaches zero):** The discretized matrix A_bar approaches the identity matrix I, while B_bar approaches zero. The model **ignores** current token x_k and preserves existing memory unchanged.

---

### 3.8 Systems Innovations for Selective SSMs

#### 3.8.1 The Parallel Associative Scan (Blelloch Algorithm)

Time-varying $\mathbf{\bar{A}}_k, \mathbf{\bar{B}}_k$ prevents FFT convolution. Mamba formulates the recurrence as an **associative binary operator** $\circ$:

$$
(a_j, b_j) \circ (a_i, b_i) = (a_j \cdot a_i, a_j \cdot b_i + b_j)
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

The sequence of hidden states is computed in parallel across GPU threads in **$O(\log L)$ span/depth** with **$O(L)$ total work**.

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

#### 3.8.3 Activation Recomputation in the Backward Pass

Instead of saving expanded intermediate states $(B, L, D, N)$ in HBM during the forward pass, Mamba discards them and recomputes them dynamically in fast SRAM during backpropagation, bounding activation memory to **$O(L \cdot D)$** instead of $O(L \cdot D \cdot N)$.

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

#### 4.1.1 The 16 Bytes per Parameter Rule in Mixed-Precision Adam

In mixed-precision training (BF16 forward/backward with FP32 Adam optimizer), static memory consumption equals **16 bytes per parameter**:

$$
\text{Memory}_{\text{static}} = \underbrace{2\Phi}_{\text{Weights (BF16)}} + \underbrace{2\Phi}_{\text{Gradients (BF16)}} + \underbrace{4\Phi}_{\text{Master Weights (FP32)}} + \underbrace{4\Phi}_{\text{Momentum } m \text{ (FP32)}} + \underbrace{4\Phi}_{\text{Variance } v \text{ (FP32)}} = \mathbf{16\Phi\text{ Bytes}}
$$

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

#### 4.1.2 Mathematical Proof: Why FP32 Master Weights & Optimizer States Are Required

1. **Underflow of Gradient Updates:**
   In FP16, machine epsilon is $\epsilon_{\text{FP16}} \approx 9.77 \times 10^{-4}$. A standard learning rate update:
   $$
   \Delta w = \eta \cdot g_t = 10^{-4} \times 10^{-3} = 10^{-7}
   $$
   Adding $10^{-7}$ directly to an FP16 weight ($w \approx 1.0$) causes underflow ($\Delta w \to 0$), freezing model updates.

2. **Compound Rounding Errors in Adam Moving Averages:**
   $$
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   $$
   Accumulating these recursive sums over $10^5$ steps in 16-bit precision leads to variance collapse and loss divergence.

---

### 4.2 Data Parallelism (DDP) & Ring All-Reduce Proofs

In Distributed Data Parallelism, $N$ GPUs compute independent mini-batch gradients $\nabla \mathcal{L}_i$. Ring All-Reduce averages gradients in two phases:

1. **Reduce-Scatter:** Each GPU receives an aggregated $1/N$ gradient shard after communicating $\frac{N-1}{N} \times |W|$ bytes.
2. **All-Gather:** Each GPU broadcasts its updated $1/N$ shard after communicating $\frac{N-1}{N} \times |W|$ bytes.

$$
\text{Total All-Reduce Communication Volume} = 2 \times \left(\frac{N-1}{N}\right) \times |W| \text{ bytes}
$$

---

### 4.3 Tensor Parallelism (Megatron-LM Intra-Layer Slicing)

Tensor Parallelism shards individual weight matrices across GPUs within the same server over NVLink.

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

$$
\text{MLP}(X) = \text{GeLU}(X W_1) W_2
$$

1. **Column-Parallel Linear 1:** Slices W_1 column-wise into N shards. Because GeLU is element-wise:
   $$
   Z_i = \text{GeLU}(X W_{1,i}) \quad (\text{Zero inter-GPU communication required})
   $$

2. **Row-Parallel Linear 2:** Slices $W_2$ row-wise:
   $$
   Y = \sum_{i=0}^{N-1} Z_i W_{2,i}
   $$
   A single **All-Reduce (Sum)** operation across ranks yields mathematically exact output $Y$.

#### 4.3.2 Slicing Multi-Head Self-Attention (MHSA)

- $W_Q, W_K, W_V$ projections are sliced column-wise across attention heads.
- $W_O$ output projection is sliced row-wise and summed via a single All-Reduce.
- **Communication Volume:** Exactly $2 \times \text{All-Reduce}$ per layer in the forward pass, and $2 \times \text{All-Reduce}$ in the backward pass.

---

### 4.4 Pipeline Parallelism (PP) & Inter-Layer Scheduling

#### 4.4.1 The Pipeline Bubble: Exact Derivation

When partitioning L layers across p pipeline stages with m micro-batches (where m is much greater than p):

$$
t_{\text{idle}} = (p - 1) \cdot (t_F + t_B), \qquad t_{\text{total}} = (m + p - 1) \cdot (t_F + t_B)
$$

$$
F_{\text{bubble}} = \frac{t_{\text{idle}}}{t_{\text{total}}} = \frac{p - 1}{p - 1 + m}
$$

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

$$
\nabla_W \mathcal{L}_{\text{full}}(W) = \frac{1}{B} \sum_{i=1}^B \nabla_W \ell(x_i; W) = \frac{1}{m} \sum_{k=1}^m \left( \frac{1}{b} \sum_{j=1}^b \nabla_W \ell(x_{k,j}; W) \right) = \frac{1}{m} \sum_{k=1}^m \nabla_W \mathcal{L}_k(W)
$$

#### 4.4.3 GPipe vs. 1F1B Schedule Activation Footprints

```text
1F1B (One-Forward-One-Backward) Schedule on GPU 0:
Time ──►  t1   t2   t3   t4   t5   t6   t7   t8
GPU 0:   [F1] [F2] [F3] [F4] [B1] [F5] [B2] [F6] ...
Act Mem:  +1   +2   +3   +4   -1   +4   -1   +4   (Bounded at p = 4!)
```

- **GPipe:** Peak activation memory scales with micro-batches $O(m)$.
- **1F1B (Megatron-LM):** Peak activation memory is strictly bounded by pipeline stages $O(p)$, independent of $m$.

---

### 4.5 Context Parallelism (CP) & Ring Attention

#### 4.5.1 The Ring Communication Topology

For long sequences (128k to 1M tokens), sequence length L is partitioned into chunks of size L/N across N GPUs organized in a ring.

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

#### 4.5.2 Online Numerically Stable Softmax Accumulation Proof

Let block 1 have local maximum m_1, normalizer d_1, and unnormalized accumulator O_1. When block 2 arrives with stats (m_2, d_2, O_2), the online merge proceeds as:

$$
m_{\text{new}} = \max(m_1, m_2)
$$

$$
d_{\text{new}} = d_1 \cdot \exp(m_1 - m_{\text{new}}) + d_2 \cdot \exp(m_2 - m_{\text{new}})
$$

$$
O_{\text{new}} = \frac{O_1 \cdot d_1 \cdot \exp(m_1 - m_{\text{new}}) + O_2 \cdot d_2 \cdot \exp(m_2 - m_{\text{new}})}{d_{\text{new}}}
$$

This guarantees bit-level mathematical equivalence to monolithic softmax without materializing the global attention score matrix in memory.

---

### 4.6 ZeRO (Zero Redundancy Optimizer) & FSDP Mechanics

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
- Partitions FP32 Adam states ($12\Phi$) across $N$ ranks. Memory per GPU:
  $$
  M_{\text{ZeRO-1}} = 2\Phi + 2\Phi + \frac{12\Phi}{N}
  $$

#### 4.6.2 ZeRO-Stage 2: Gradient Sharding
- Retains gradients only for assigned parameter shards via Reduce-Scatter. Memory per GPU:
  $$
  M_{\text{ZeRO-2}} = 2\Phi + \frac{2\Phi}{N} + \frac{12\Phi}{N}
  $$

#### 4.6.3 ZeRO-Stage 3: Parameter Sharding (Fully Sharded Data Parallel)
- Shards parameters themselves across ranks. Layers are gathered on-the-fly and discarded immediately after forward/backward execution. Memory per GPU:
  $$
  M_{\text{ZeRO-3}} = \frac{2\Phi}{N} + \frac{2\Phi}{N} + \frac{12\Phi}{N} = \frac{16\Phi}{N}
  $$

#### 4.6.4 Communication vs. Memory Trade-Off Across Stages

| Configuration | Parameters / GPU | Gradients / GPU | Optimizer / GPU | Total Static Memory (7B on 4 GPUs) | Total Extra Communication Volume |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DDP (ZeRO-0)** | 2P (14 GB) | 2P (14 GB) | 12P (84 GB) | 112.0 GB | Baseline (2P All-Reduce) |
| **ZeRO-1** | 2P (14 GB) | 2P (14 GB) | 12P/N (21 GB) | 49.0 GB | Baseline + P All-Gather |
| **ZeRO-2** | 2P (14 GB) | 2P/N (3.5 GB) | 12P/N (21 GB) | 38.5 GB | Baseline (Reduce-Scatter replaces All-Reduce) |
| **ZeRO-3 / FSDP** | 2P/N (3.5 GB) | 2P/N (3.5 GB) | 12P/N (21 GB) | 28.0 GB | Baseline + 1.5x Forward/Backward All-Gather |

---

### 4.7 Operator & Kernel Fusion (FlashAttention Deep Dive)

Standard attention incurs $O(L^2)$ HBM memory read/writes:

$$
S = Q K^T \in \mathbb{R}^{L \times L}, \qquad P = \text{softmax}(S) \in \mathbb{R}^{L \times L}, \qquad O = P V \in \mathbb{R}^{L \times d}
$$

FlashAttention partitions Q, K, V into tiles that fit inside on-chip SRAM, computing online softmax rescaling entirely within SRAM and reducing HBM transfers from O(L^2) to O(L).

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

In production inference systems, model serving profitability is determined by unit token cost:

$$
\text{Cost per Token} = \frac{\text{Hourly GPU Server Cost}}{\text{Tokens Generated per Hour}} = \frac{\text{Hourly Cost}}{\text{Throughput (Tokens/s)} \times 3600}
$$

---

### 5.2 The Two Execution Phases: Prefill vs. Decoding

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PREFILL VS. DECODING PHASES                           │
│                                                                             │
│  [Phase 1: Prefill / Prompt Processing]                                     │
│   Input: Full prompt of S tokens in parallel ──► Matrix-Matrix GEMM         │
│   - Compute-Bound (Very High Arithmetic Intensity: I >> I_knee)             │
│   - Sets the Time To First Token (TTFT).                                    │
│                                                                             │
│  [Phase 2: Autoregressive Decoding]                                         │
│   Input: 1 token at a time ──► Matrix-Vector GEMV                           │
│   - Memory-Bound (Very Low Arithmetic Intensity: I << I_knee)               │
│   - Must read all model weights (2 * Phi bytes) from HBM for EVERY token!   │
│   - Sets the Time Per Output Token (TPOT) and generation speed.             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 5.3 First-Principles FLOP & Parameter Counting (Case Study: LLaMA 3.3 70B)

Consider a transformer with hidden dimension d, L layers, vocabulary V, query heads n_q, key-value heads n_kv (with ratio r = n_kv / n_q), and SwiGLU intermediate size d_ffn.

#### 5.3.1 Layer-by-Layer FLOP Derivations

Multiplying an (m x k) matrix by a (k x n) matrix requires 2mkn FLOPs.

For sequence length $S$:

1. **RMSNorm Operations (4Sd FLOPs per normalization):**
   $$
   \text{FLOPs}_{\text{RMSNorm}} = S \cdot d \text{ (square)} + S \cdot d \text{ (sum)} + S \cdot d \text{ (div)} + S \cdot d \text{ (scale)} = 4Sd
   $$

2. **Query Projection (W_Q is d x d):**
   $$
   \text{FLOPs}_Q = 2 S d^2
   $$

3. **Key & Value Projections (W_K, W_V are d x rd):**
   $$
   \text{FLOPs}_{KV} = 2 \times \left( 2 S d \cdot (r d) \right) = 4 r S d^2 \quad \left(\text{For } r = \frac{1}{8}, \text{ FLOPs}_{KV} = 0.5 S d^2\right)
   $$

4. **Attention Computations (QK^T and AV):**
   $$
   \text{FLOPs}_{QK^T} = 2 S^2 d, \qquad \text{FLOPs}_{AV} = 2 S^2 d
   $$

5. **Output Projection (W_O is d x d):**
   $$
   \text{FLOPs}_O = 2 S d^2
   $$

6. **SwiGLU Feed-Forward Network (W_gate, W_up, W_down are d x d_ffn):**
   $$
   \text{FLOPs}_{\text{FFN}} = 3 \times \left( 2 S d d_{\text{ffn}} \right) = 6 S d d_{\text{ffn}} \quad (\text{For } d_{\text{ffn}} = 3.5d, \text{ FLOPs}_{\text{FFN}} = 21 S d^2)
   $$

$$
\text{Total Layer FLOPs} = \left( 4 + 4r + \frac{6 d_{\text{ffn}}}{d} \right) S d^2 + 4 S^2 d
$$

During single-token decoding (S = 1), attention quadratic terms are negligible:

$$
\text{Forward Compute per Token} \approx 2 \times \Phi \text{ FLOPs}
$$

#### 5.3.2 Total Parameter Calculation

$$
\Phi = 2 V d + L \left[ 2 d^2 (1 + r) + 3 d d_{\text{ffn}} + 2 d \right] + d
$$

For LLaMA 3.3 70B ($d = 8192, L = 80, V = 128256, r = 1/8, d_{\text{ffn}} = 28672$):

$$
\Phi = 2(128256)(8192) + 80\left[ 2(8192^2)(1.125) + 3(8192)(28672) + 2(8192) \right] + 8192 = \mathbf{70,553,706,496\text{ parameters}}
$$

At BF16 precision ($2\text{ bytes/param}$):

$$
\text{Model Weight Memory} = 70.5537 \times 10^9 \times 2 = \mathbf{141.1\text{ GB}}
$$

---

### 5.4 The KV Cache: Eliminating Redundant Attention Computations

Without caching, generating token $t$ requires recomputing attention over all $t-1$ historical tokens, resulting in $O(L^2)$ total operations:

$$
\text{Operations}_{\text{NoCache}} = \sum_{t=1}^L (2\Phi + 4 t d) = 2 L \Phi + 2 L^2 d = O(L^2)
$$

Caching Key and Value vectors reduces generation to strict **$O(L)$ total operations** ($O(1)$ compute per step).

#### 5.4.1 Mathematical Memory Footprint Formulation

$$
\text{KV Cache Size (Bytes)} = 2 \times \underbrace{2}_{\text{K and V}} \times n_{\text{layers}} \times n_{kv} \times d_{\text{head}} \times S \times B \times b_{\text{elem}}
$$

Where $b_{\text{elem}} = 2\text{ bytes}$ in 16-bit precision.

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

During single-request generation ($B = 1$), loading the entire model from HBM yields an arithmetic intensity of only $1\text{ FLOP/Byte}$.

With batch size $B$, weights are loaded once from HBM and reused across all $B$ requests:

$$
I(B) = \frac{2 B \Phi}{2 \Phi + \text{KV}_{\text{Cache}}(B)} = \frac{2 B \Phi}{2 \Phi + 4 L n_{kv} d_{\text{head}} S B} \approx B \text{ FLOPs/Byte}
$$

#### 5.5.2 Inflection Point: The Knee Batch Size Formula

The transition from memory-bound to compute-bound generation occurs at $B_{\text{knee}}$:

$$
B_{\text{knee}} = \frac{P_{\text{peak}}}{\text{Bandwidth}_{\text{HBM}}} = I_{\text{knee}}
$$

On an 4x H100 system ($P_{\text{peak}} = 3956\text{ TFLOPs}, \text{Bandwidth} = 13.4\text{ TB/s}$), $B_{\text{knee}} \approx 295$.

#### 5.5.3 Throughput vs. Per-User Latency (TPOT) Trade-Off

$$
\text{TPOT}(B) = \frac{2 \Phi + \text{KV}(B)}{\min\left( \text{Bandwidth}_{\text{HBM}}, \; \frac{P_{\text{peak}}}{I(B)} \right)}
$$

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

#### 5.5.4 Model Bandwidth Utilization (MBU) vs. Model FLOPs Utilization (MFU)

$$
\text{MBU} = \frac{\text{Throughput (tok/s)} \times \left( \text{Memory}_{\text{Weights}} + \text{Memory}_{\text{KV}} \right)}{\text{Peak GPU Memory Bandwidth}}
$$

$$
\text{MFU} = \frac{\text{Throughput (tok/s)} \times 2 \Phi}{\text{Peak Hardware TFLOPs}}
$$

---

### 5.6 PagedAttention: OS-Inspired Virtual Memory for KV Cache (vLLM)

#### 5.6.1 The Memory Fragmentation Crisis

Pre-allocating contiguous memory for `max_seq_len` (e.g., 4096 tokens) wastes $60\% - 80\%$ of VRAM due to internal and external fragmentation.

#### 5.6.2 Block Tables & Virtual Paging Architecture

PagedAttention partitions KV tensors into fixed-size **Pages** (e.g., $B_{\text{block}} = 16$ tokens) and maps them via a **Block Table**:

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

The memory waste fraction is strictly bounded by the last block:

$$
\text{Memory Waste Fraction} \le \frac{B_{\text{block}} - 1}{2 S} < \mathbf{4\%}
$$

#### 5.6.3 Copy-on-Write (CoW) Forking & Parallel Sampling

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

#### 5.6.4 Prefix Caching (Prompt Caching)

Hashes prompt token prefixes $\mathcal{H}(t_0, \dots, t_k)$ to index precomputed physical KV pages in an LRU cache, reducing Time To First Token (TTFT) by $80\% - 95\%$.

---

### 5.7 Continuous (Iteration-Level) Batching vs. Static Batching

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

---

## 6. Module V: Positional Encodings & Rotary Position Embedding (RoPE)

In transformer self-attention, the dot-product $\text{Attn}(Q, K) = Q K^T$ is permutation-equivariant: shuffling the input token sequence produces the exact same attention weights. To encode word order and relative syntactic distances, modern architectures inject **Rotary Position Embedding (RoPE)** (Su et al., 2021).

### 6.1 The Fundamental Architectural Principle: Vectors vs. Weights

> [!IMPORTANT]
> **Core Architectural Rule:** RoPE is applied **strictly to the dynamic Query (Q) and Key (K) activation vectors** after linear projection. It is **never applied to the static projection weight matrices (W_Q, W_K)**.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RoPE IN THE FORWARD PASS                           │
│                                                                             │
│   Input Token X_m at Position m                                             │
│         │                                                                   │
│         ├──► [ Linear W_Q (Static) ] ──► Raw Vector Q_m ──► [ RoPE R_m ] ──► Rotated Vector Q̃_m ──┐
│         │                                                                                          ├──► Dot Product Score(m, n)
│         └──► [ Linear W_K (Static) ] ──► Raw Vector K_m ──► [ RoPE R_m ] ──► Rotated Vector K̃_m ──┘
│                                                                                                    (Encodes relative lag n - m!)
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Why RoPE Cannot Be Applied to Weights:
1. **Dynamic Position Dependence:** Weight matrices W_Q and W_K (both in R^{d x d}) are static, learned parameters shared universally across all tokens. In contrast, the token position m changes dynamically for every token in a sentence. Rotation must be token-specific.
2. **Relative Distance Encoding:** Applying rotation matrices R_m and R_n to vectors ensures that their product encodes strictly the relative displacement (n - m) directly into the attention inner product:

   $$
   R_{\Theta, m}^T \; R_{\Theta, n} = R_{\Theta, \; n-m}
   $$

---

### 6.2 The Block-Diagonal 2D Rotation Matrix Formulation

For an attention head of dimension d (where d is an even integer), RoPE constructs a d x d orthogonal block-diagonal rotation matrix composed of d/2 independent 2D rotation sub-matrices:

$$
R_{\Theta, m} = \begin{pmatrix}
\cos(m\theta_1) & -\sin(m\theta_1) & 0 & 0 & \cdots & 0 & 0 \\
\sin(m\theta_1) & \cos(m\theta_1) & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos(m\theta_2) & -\sin(m\theta_2) & \cdots & 0 & 0 \\
0 & 0 & \sin(m\theta_2) & \cos(m\theta_2) & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos(m\theta_{d/2}) & -\sin(m\theta_{d/2}) \\
0 & 0 & 0 & 0 & \cdots & \sin(m\theta_{d/2}) & \cos(m\theta_{d/2})
\end{pmatrix}
$$

1. **Token Position Index (m):** The integer index of the token in the sequence (m = 0, 1, 2, ...).
2. **Base Channel Frequencies:** Dimension-dependent base frequencies defined geometrically:
   $$
   \theta_i = 10000^{-2(i-1)/d} \quad \text{for } i \in \{1, 2, \dots, d/2\}
   $$
3. **2D Coordinate Subspaces:** The matrix groups vector elements into 2D pairs:
   $$
   (q_1, q_2), \; (q_3, q_4), \; \dots, \; (q_{d-1}, q_d)
   $$
   and rotates each pair in its own 2D plane by angle $\alpha_i = m \theta_i$.

---

### 6.3 Step-by-Step Numerical Example (d = 6 at Position m = 1)

Consider a query vector $Q_1 = [1, 0, 2, 1, 0, 3]^T$ at sequence position $m = 1$ with head dimension $d = 6$.

#### Step 1: Compute the $d/2 = 3$ Base Frequencies
$$
\theta_1 = 10000^{-2(1-1)/6} = 10000^0 = \mathbf{1.0}
$$

$$
\theta_2 = 10000^{-2(2-1)/6} = 10000^{-1/3} \approx \mathbf{0.046416}
$$

$$
\theta_3 = 10000^{-2(3-1)/6} = 10000^{-2/3} \approx \mathbf{0.002154}
$$

#### Step 2: Compute Rotation Angles ($m \theta_i$ for $m = 1$)
- Plane 1: $\alpha_1 = 1 \times 1.0 = 1.0\text{ rad}$
- Plane 2: $\alpha_2 = 1 \times 0.046416 = 0.046416\text{ rad}$
- Plane 3: $\alpha_3 = 1 \times 0.002154 = 0.002154\text{ rad}$

#### Step 3: Apply 2D Rotation to Each Coordinate Pair

1. **Pair 1 $(q_1, q_2) = (1, 0)$:**
   $$
   \begin{pmatrix} \tilde{q}_1 \\ \tilde{q}_2 \end{pmatrix} = \begin{pmatrix} \cos(1.0) & -\sin(1.0) \\ \sin(1.0) & \cos(1.0) \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 0.540302 \\ 0.841471 \end{pmatrix}
   $$

2. **Pair 2 $(q_3, q_4) = (2, 1)$:**
   $$
   \begin{pmatrix} \tilde{q}_3 \\ \tilde{q}_4 \end{pmatrix} = \begin{pmatrix} \cos(0.0464) & -\sin(0.0464) \\ \sin(0.0464) & \cos(0.0464) \end{pmatrix} \begin{pmatrix} 2 \\ 1 \end{pmatrix} \approx \begin{pmatrix} 2(0.9989) - 1(0.0464) \\ 2(0.0464) + 1(0.9989) \end{pmatrix} = \begin{pmatrix} 1.9514 \\ 1.0917 \end{pmatrix}
   $$

3. **Pair 3 $(q_5, q_6) = (0, 3)$:**
   $$
   \begin{pmatrix} \tilde{q}_5 \\ \tilde{q}_6 \end{pmatrix} = \begin{pmatrix} \cos(0.00215) & -\sin(0.00215) \\ \sin(0.00215) & \cos(0.00215) \end{pmatrix} \begin{pmatrix} 0 \\ 3 \end{pmatrix} \approx \begin{pmatrix} -3(0.00215) \\ 3(0.99999) \end{pmatrix} = \begin{pmatrix} -0.00646 \\ 2.99999 \end{pmatrix}
   $$

#### Step 4: Final Rotated Vector
$$
\tilde{Q}_1 \approx \begin{bmatrix} 0.5403, & 0.8415, & 1.9514, & 1.0917, & -0.0065, & 3.0000 \end{bmatrix}^T
$$

---

### 6.4 Complex Representation & Mathematical Derivation via Euler's Formula

RoPE was derived by seeking a function f(x, m) such that the inner product between query at position m and key at position n depends purely on the relative distance (m - n):

$$
\langle f(Q, m), \; f(K, n) \rangle = g(Q, K, m - n)
$$

1. **Complex Number Embedding:** A 2D real coordinate pair $q = (q_1, q_2)^T$ is mapped to a single complex number:
   $$
   q_{\mathbb{C}} = q_1 + i q_2
   $$
   The standard Euclidean inner product equals the real part of the complex product with the complex conjugate:
   $$
   \langle q, k \rangle = \text{Re}\left( q_{\mathbb{C}} \cdot k_{\mathbb{C}}^* \right)
   $$

2. **Phase Rotation:** We encode position $m$ by multiplying by the unitary complex phase factor $e^{i m \theta}$:
   $$
   f(q, m) = q_{\mathbb{C}} \cdot e^{i m \theta}
   $$

3. **Invariance Proof:**
   $$
   \langle f(q, m), \; f(k, n) \rangle = \text{Re}\left( (q_{\mathbb{C}} e^{i m \theta}) \cdot (k_{\mathbb{C}} e^{i n \theta})^* \right) = \text{Re}\left( q_{\mathbb{C}} k_{\mathbb{C}}^* \cdot e^{i(m - n)\theta} \right)
   $$
   Absolute positions $m$ and $n$ drop out, leaving strictly the relative displacement $(m - n)$.

4. **Matrix Realization via Euler's Identity:**

   Using Euler's formula, the complex exponential expands as:
   $$
   f(q, m) = (q_1 + i q_2)(\cos(m\theta) + i \sin(m\theta)) = (q_1 \cos(m\theta) - q_2 \sin(m\theta)) + i (q_1 \sin(m\theta) + q_2 \cos(m\theta))
   $$
   Converting back to matrix notation yields the 2D rotation block $R_{\Theta, m}$.

---

### 6.5 Mathematical Proof: Why RoPE Must Be Applied to Both Q and K

Why not save compute by applying rotation only to Query $Q$?

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                 WHY ROTATION MUST BE APPLIED TO BOTH Q AND K                │
│                                                                             │
│  Scenario 1: Rotate BOTH Q and K (Correct - Translation Invariant)          │
│   Score(m, n) = (R_m Q_m)^T (R_n K_n) = Q_m^T (R_m^T R_n) K_n              │
│               = Q_m^T (R_{-m} R_n) K_n = Q_m^T R_{n-m} K_n                  │
│   ──► Inner product depends purely on RELATIVE distance (n - m)!            │
│                                                                             │
│  Scenario 2: Rotate ONLY Q (Incorrect - Fails Relative Invariance)          │
│   Score(m, n) = (R_m Q_m)^T (K_n) = Q_m^T R_m^T K_n = Q_m^T R_{-m} K_n     │
│   ──► Inner product depends on ABSOLUTE position m.                         │
│   ──► Key position n is absent! Model cannot measure distance!              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### The Clock Hands Analogy:
- **Both Rotated:** Token position m rotates Query hand by m degrees; position n rotates Key hand by n degrees. The dot product measures the **relative angle between the two clock hands** ((n - m) degrees). Tokens separated by 2 positions at indices (1, 3) or (101, 103) produce the exact same angle (2 degrees).
- **Only Q Rotated:** Query hand rotates by m degrees, but Key hand stays pinned at 12 o'clock (0 degrees). The measured angle is always m degrees, destroying distance context.

---

### 6.6 Hardware-Efficient Implementation via Element-Wise Vector Operations

Materializing the sparse $d \times d$ matrix $R_{\Theta, m}$ in GPU memory is slow. Production engines (PyTorch, FlashAttention, vLLM) compute RoPE using fast element-wise vector operations:

$$
\tilde{Q}_m = Q_m \odot \cos(m\Theta) + \tilde{Q}_{\text{swap}} \odot \sin(m\Theta)
$$

Where:
$$
\Theta = [\theta_1, \theta_1, \theta_2, \theta_2, \dots, \theta_{d/2}, \theta_{d/2}]^T \in \mathbb{R}^d
$$

$$
\tilde{Q}_{\text{swap}} = [-q_2, \; q_1, \; -q_4, \; q_3, \; \dots, \; -q_d, \; q_{d-1}]^T
$$

---

## 7. Module VI: Modern Frontier Topologies — The Gemma 4 Architecture Suite

The **Gemma 4** open-model family introduces architectural innovations designed to maximize parameter capacity and generation throughput while remaining hardware-efficient.

### 7.1 The Gemma 4 Family Taxonomy (E2B, E4B, 31B Dense, 26B A4B MoE)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GEMMA 4 MODEL ARCHITECTURES                       │
│                                                                             │
│  [Gemma 4 - E2B & E4B (Dense with PLE)]                                     │
│   - Target: Mobile & Edge Devices (Under 4 GB VRAM)                         │
│   - Innovation: Per-Layer Embeddings (PLE) scale effective parameter count  │
│     to 2B / 4B while executing at the FLOP footprint of a sub-1B trunk.     │
│                                                                             │
│  [Gemma 4 - 31B (Flagship Dense)]                                           │
│   - Target: High-Capacity Reasoning & Coding                                │
│   - Fully dense 31B parameter model with interleaved local/global attention.│
│                                                                             │
│  [Gemma 4 - 26B A4B (Mixture of Experts - MoE)]                             │
│   - Target: High-Throughput Server Serving                                  │
│   - Total Parameters: 26 Billion | Active per Token: 4 Billion              │
│   - Dynamic top-k expert routing with load balancing.                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 7.2 Interleaved Local Sliding Window & Global Attention (K=V Fusion)

Standard global attention incurs $O(L^2)$ computation for context length $L$. Gemma 4 interleaves **Local Sliding Window Attention** with **Global Full-Context Attention**:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GEMMA 4 ATTENTION PATTERN INTERLEAVING                   │
│                                                                             │
│  Layer 0: Local Sliding Window Attention (e.g., Window = 512 or 1024 tokens)│
│  Layer 1: Local Sliding Window Attention                                    │
│  Layer 2: Local Sliding Window Attention                                    │
│  Layer 3: Global Full Attention (K=V Weight Fusion + Full Context Access)   │
│  ...                                                                        │
│  Layer L-1 (Final Layer): ALWAYS Global Full Attention Layer                │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Sliding Window Size:** Small models (E2B, E4B) use a local window of 512 tokens; large models (26B A4B, 31B) use 1024 tokens. Compute complexity drops from O(L^2) to O(L * W).
2. **K=V Global Attention Fusion:** In global attention layers, Key projections are set equal to Value projections (K = V), halving KV cache memory footprint for full-context layers.
3. **Guaranteed Global Aggregation:** The final layer is always a global attention layer, ensuring representations across the entire context window are synthesized before generating predictions.

---

### 7.3 Partitioned / Low-Frequency-Pruned RoPE (p-RoPE)

In standard RoPE, high channel dimensions have near-zero frequency ($\theta \approx 0$), resulting in negligible rotation over moderate context lengths.

**p-RoPE (Partitioned RoPE)** applies RoPE only to the high-frequency subspace while pruning low-frequency channels or reallocating them for lexical feature embeddings, improving long-context stability.

---

### 7.4 Mixture of Experts (MoE) Architecture in Gemma 4 26B A4B

In Gemma 4 26B A4B, the dense FFN block is replaced by a sparse Mixture-of-Experts layer with a top-$k$ Softmax Router:

```text
Input Activation x ──┬──► [ Router Gate: G(x) = Softmax(TopK(x W_gate)) ]
                     │                   │
                     ├──► [ Expert 1 ] ──┤
                     ├──► [ Expert 2 ] ──┼──► Weighted Sum Y = sum_{i=1}^k G(x)_i Expert_i(x)
                     │    ...            │
                     └──► [ Expert E ] ──┘
```

- Total parameters: $26\text{ Billion}$.
- Active parameters per token: $4\text{ Billion}$.
- **Throughput Advantage:** Delivers the reasoning capacity of a $26\text{B}$ model at the serving latency and FLOP cost of a $4\text{B}$ dense model.

---

### 7.5 Per-Layer Embeddings (PLE) in Gemma 4 E2B & E4B

#### 7.5.1 Motivation: Decoupling Capacity from FLOPs

In standard Transformers, input tokens are mapped to continuous vectors *only once* at layer $0$:

$$
h_L = h_0 + \sum_{l=0}^{L-1} f_l(h_l)
$$

As depth increases ($L > 32$), residual accumulation dilutes original lexical identity. **Per-Layer Embeddings (PLE)** decouple parameter capacity from per-token compute by providing dedicated embedding tables at each layer.

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

#### 7.5.2 Mathematical Formulation & Forward Pass

1. **Pre-Layer Representation:**
   $$
   p_l = \frac{1}{\sqrt{2}} \left( e_{\text{lookup}}^{(l)} + W_{\text{proj}}^{(l)} e_0 \right) \in \mathbb{R}^{d_{\text{ple}}}
   $$

2. **State-Dependent Gating & Hadamard Integration:**
   $$
   g_l = \sigma(W_g h_l'') \in (0, 1)^{d_{\text{ple}}}
   $$
   $$
   v_l = g_l \odot p_l \in \mathbb{R}^{d_{\text{ple}}}
   $$
   $$
   h_{l+1} = h_l'' + \text{RMSNorm}(W_{\text{out}} v_l)
   $$

#### 7.5.3 Why the Hadamard Product Is Structurally Mandatory

- **Independent Dimension-Wise Selection:** Allows the model to independently open or close individual feature axes (e.g., maintaining semantic entity category while filtering grammatical tense):
  $$
  \frac{\partial v_{l,i}}{\partial p_{l,i}} = g_{l,i} \in (0, 1)
  $$
- **Preserves Coordinate Disentanglement:** Element-wise multiplication does not rotate or blend features across orthogonal axes.
- **Zero Heavy Compute:** Requires only **$d_{\text{ple}}$ FLOPs** instead of dense matrix multiplication.

#### 7.5.4 Information Retention & Semantic Disambiguation Dynamics

```text
Case Study: "The river bank overflowed"
- Early Layers (Syntax): Gate g_2 -> 1.0 (OPENS) to inject raw syntax embeddings.
- Deep Layers (Semantics): Context has resolved "river bank" = [Shoreline].
  Gate g_24 -> 0.0 (CLOSES) to prevent static financial word embeddings from corrupting the contextual state!
```

---

### 7.6 Multi-Token Prediction (MTP) Speculative Drafters

Gemma 4 models integrate native **Multi-Token Prediction (MTP)** drafter modules:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-TOKEN PREDICTION (MTP) SPECULATIVE FLOW            │
│                                                                             │
│  Step 1: Target Model Forward Pass (Token t)                                │
│          Outputs: Token t+1 + Intermediate Hidden State h_t                 │
│                                                                             │
│  Step 2: MTP Drafter Head (Lightweight Module)                              │
│          Takes h_t ──► Autoregressively generates Draft Tokens [t+2, t+3]   │
│                                                                             │
│  Step 3: Parallel Verification in Next Forward Pass                         │
│          Target Model validates [t+1, t+2, t+3] CONCURRENTLY in 1 forward pass!│
│          ──► Accepted tokens committed instantly (2x - 3x speedup!)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **Speculative Verification:** Rather than requiring a separate draft model (e.g. speculative decoding with a 1B helper model), Gemma 4 uses internal MTP heads to propose candidates that the trunk model verifies in a single forward pass, achieving **$2\times - 3\times$ lower decode latency**.

---

## 8. Module VII: Cross-Disciplinary Synthesis & Engineering Playbook

### 8.1 Full Model Lifecycle Workflow

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                    END-TO-END LLM SYSTEMS LIFECYCLE                         │
│                                                                             │
│  [Phase 1: Architecture Selection]                                          │
│   - Attention (Transformer) vs. Selective SSM (Mamba) vs. Gemma 4 Hybrid    │
│   - RoPE Vector Rotations + Interleaved Local/Global Attention              │
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
│   - Native Speculative Decoding via Gemma 4 Multi-Token Prediction (MTP)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 8.2 Systems Engineering Decision Flowchart

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
│   ├── RAM < 4 GB: Gemma 4 E2B / E4B with Per-Layer Embeddings (PLE)
│   ├── RAM < 8 GB: GGUF / K-Quants (4-bit or 2-bit offloading to CPU/Metal)
│   ├── Single GPU 24 GB: 4-Bit GPTQ or AWQ
│   └── Next-Gen Custom Silicon: BitNet 1.58b Ternary Kernels
│
└── Long-Context Streaming / Real-Time Generation
    ├── Unbounded Sequence Processing: Mamba (Selective SSM S6 Block)
    └── High-Speed Decoding: Gemma 4 with Multi-Token Prediction (MTP)
```

---

### 8.3 Comprehensive Glossary of Symbols & Notation

| Symbol | Mathematical Definition | Domain / Systems Role |
| :--- | :--- | :--- |
| $\mathbf{A}$ | Continuous state transition matrix ($\mathbb{R}^{N \times N}$) | Defines continuous-time internal memory dynamics in SSMs |
| $\mathbf{\bar{A}}, \mathbf{\bar{B}}$ | Discretized state matrices | Discretized via Zero-Order Hold (ZOH) with step $\Delta$ |
| $\Delta$ | Step size parameter ($\mathbb{R}^+$) | Timescale resolution; acts as input-dependent gate in Mamba |
| $\mathbf{\bar{K}}$ | SSM convolution kernel ($\mathbb{R}^L$) | Enables FFT-based parallel sequence training in LTI SSMs |
| R(Theta, m) | Block-diagonal rotation matrix (d x d) | Rotates Q, K vectors by position m in RoPE |
| $\theta_i$ | Geometric base frequency ($10000^{-2(i-1)/d}$) | Frequency scale for $i$-th 2D coordinate plane in RoPE |
| $s$ | Quantization scale factor ($\mathbb{R}^+$) | Maps continuous interval to discrete integer grid |
| $z$ | Integer zero-point ($\mathbb{Z}$) | Offsets asymmetric quantized values |
| $H^{-1}$ | Inverse Hessian matrix ($2 X X^T + \lambda I)^{-1}$ | Second-order sensitivity matrix in GPTQ error redistribution |
| $\Phi$ | Total model parameter count | Base unit for distributed memory and compute calculations |
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

---

### 8.4 References & Primary Sources

This technical compendium synthesizes theoretical principles, systems implementations, and architectural benchmarks from the following primary publications and engineering references:

1. **Rotary Position Embeddings & Attention:**
   - Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, Yunfeng Liu. *"RoFormer: Enhanced Transformer with Rotary Position Embedding."* [arXiv:2104.09864](https://arxiv.org/abs/2104.09864).
   - Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. *"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness."* [NeurIPS 2022 / arXiv:2205.14135](https://arxiv.org/abs/2205.14135).

2. **The Gemma 4 Frontier Architecture Suite:**
   - Maarten Grootendorst. *"A Visual Guide to Gemma 4."* [Exploring Language Models / Newsletter](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4).
   - Google DeepMind Gemma Team. *"Gemma 4: Open Model Architecture Suite with Per-Layer Embeddings, MoE, and Multi-Token Prediction."* Technical Report.
   - Internal Repository: [per_layer_embeddings_gemma4.md](file:///home/backup-279/ML_Basics/per_layer_embeddings_gemma4.md).

3. **Model Quantization & Precision:**
   - Maarten Grootendorst. *"A Visual Guide to Quantization."* [Towards Data Science](https://medium.com/data-science/a-visual-guide-to-quantization-930ebcd9be94).
   - Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. *"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers."* [arXiv:2210.17323](https://arxiv.org/abs/2210.17323).
   - Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Saksham Bhatia, Jiayu Ding, Jilong Xue, Furu Wei. *"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits."* [arXiv:2402.17764](https://arxiv.org/abs/2402.17764).

4. **State Space Models & Mamba Architectures:**
   - Maarten Grootendorst. *"A Visual Guide to Mamba and State Space Models."* [Towards Data Science](https://medium.com/data-science/a-visual-guide-to-mamba-and-state-space-models-8d0d3f7d3ea6).
   - Albert Gu, Tri Dao. *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces."* [arXiv:2312.00752](https://arxiv.org/abs/2312.00752).
   - Albert Gu, Karan Goel, Christopher Ré. *"Efficiently Modeling Long Sequences with Structured State Spaces (S4)."* [ICLR 2022 / arXiv:2111.00396](https://arxiv.org/abs/2111.00396).

5. **Distributed Systems Parallelism & Hardware Tiling:**
   - Internal Repository: [parallelism_examples.md](file:///home/backup-279/ML_Basics/parallelism_examples.md) (*Data Parallelism, Megatron-LM Tensor Parallelism, Pipeline Bubbles, Ring Attention Online Softmax, ZeRO-1/2/3, FlashAttention*).
   - Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro. *"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism."* [arXiv:1909.08053](https://arxiv.org/abs/1909.08053).
   - Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. *"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models."* [SC20 / arXiv:1910.02054](https://arxiv.org/abs/1910.02054).

6. **Inference Economics, KV Caching & PagedAttention:**
   - Tensor Economics. *"LLM Inference Economics from First Principles."* [Tensor Economics Newsletter](https://www.tensoreconomics.com/p/llm-inference-economics-from-first).
   - Musings with LLMs. *"Understanding KV Cache and PagedAttention in LLMs: A Deep Dive into Efficient Inference."* [Medium](https://medium.com/my-musings-with-llms/understanding-kv-cache-and-paged-attention-in-llms-a-deep-dive-into-efficient-inference-62fa372432ce).
   - Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Haotong Zhang, Ion Stoica. *"Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)."* [SOSP 2023 / arXiv:2309.06180](https://arxiv.org/abs/2309.06180).

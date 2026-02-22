# Neural Network Data Types in SystemVerilog

Synthesisable SystemVerilog implementations of the numerical formats used in modern neural-network training and inference hardware, with testbenches and a comparative analysis report.

Covers **nine formats** spanning the full precision stack from FP32 down to 4-bit:

| Format | Bits | Layout | Key Use |
|--------|------|--------|---------|
| **FP32** | 32 | 1/8/23 | Master weights, loss accumulation |
| **FP16** | 16 | 1/5/10 | Mixed-precision forward pass |
| **BF16** | 16 | 1/8/7 | Training (TPU, A100+) |
| **TF32** | 19 | 1/8/10 | Tensor core matmul (A100+) |
| **FP8 E4M3** | 8 | 1/4/3 | Inference weights (H100, MI300) |
| **FP8 E5M2** | 8 | 1/5/2 | Training gradients (H100, MI300) |
| **FP4 E2M1** | 4 | 1/2/1 | MXFP4 inference (Blackwell, CDNA4) |
| **NF4** | 4 | LUT(16) | QLoRA weight storage |
| **INT8** | 8 | signed | Quantised inference |

---

## Repository Structure

```
nn_datatypes/
├── sv/                          # Synthesisable RTL
│   ├── nn_dtypes_pkg.sv         # Package: packed structs, constants, conversion functions
│   ├── fp32_arith.sv            # FP32 add / multiply with normalisation
│   ├── bf16_arith.sv            # BF16 mixed-precision multiply-accumulate
│   ├── fp8_arith.sv             # FP8 E4M3×E5M2 multiply and dot product
│   ├── fp4_arith.sv             # FP4 E2M1 (MXFP4) and NF4 (QLoRA) arithmetic
│   ├── int8_arith.sv            # INT8 quantised MAC, ReLU, requantisation
│   └── type_convert.sv          # Broadcast FP32→all conversion, normalisation, INT8 quant/dequant
│
├── tb/                          # Testbenches
│   ├── tb_fp32_arith.sv         # FP32 add/mul edge cases, overflow/underflow
│   ├── tb_bf16_arith.sv         # BF16 precision loss analysis, FMA
│   ├── tb_fp8_arith.sv          # FP8 multiply, 4-element dot product
│   ├── tb_fp4_arith.sv          # FP4 E2M1/NF4 enumeration, error comparison
│   ├── tb_int8_arith.sv         # INT8 MAC, ReLU, saturation, requantisation
│   └── tb_type_convert.sv       # Cross-format conversion, round-trip error
│
└── nn_datatypes_report.docx     # Comparative analysis report
```

## Building and Running

The design targets any SystemVerilog-2012 compliant simulator. All modules depend on `nn_dtypes_pkg.sv`, which must be compiled first.

### Icarus Verilog (iverilog)

> **Note:** Icarus has limited SystemVerilog support. The testbenches use `real`, `$shortrealtobits`, and packed-struct features that may require commercial simulators.

### Verilator

```bash
verilator --sv --binary --timing \
  sv/nn_dtypes_pkg.sv sv/fp32_arith.sv tb/tb_fp32_arith.sv \
  -o tb_fp32_arith
./obj_dir/tb_fp32_arith
```

### Synopsys VCS

```bash
vcs -sverilog -full64 \
  sv/nn_dtypes_pkg.sv sv/fp32_arith.sv tb/tb_fp32_arith.sv \
  -o tb_fp32_arith
./tb_fp32_arith
```

### Cadence Xcelium

```bash
xrun -sv \
  sv/nn_dtypes_pkg.sv sv/fp32_arith.sv tb/tb_fp32_arith.sv
```

### Mentor Questa / ModelSim

```bash
vlog -sv sv/nn_dtypes_pkg.sv sv/fp32_arith.sv tb/tb_fp32_arith.sv
vsim -run -all tb_fp32_arith
```

Replace the filenames to run any other testbench. The package must always be compiled first. For example, to run all FP4 tests:

```bash
vcs -sverilog -full64 \
  sv/nn_dtypes_pkg.sv sv/fp4_arith.sv tb/tb_fp4_arith.sv \
  -o tb_fp4_arith && ./tb_fp4_arith
```

## Module Descriptions

### `nn_dtypes_pkg`

Central package defining:

- **Packed structs** for all nine formats (`fp32_t`, `fp16_t`, `bf16_t`, `tf32_t`, `fp8_e4m3_t`, `fp8_e5m2_t`, `fp4_e2m1_t`, `nf4_t`, `int8_t`)
- **Exponent biases** and special-value constants (zero, inf, NaN)
- **Conversion functions**: `fp32_to_bf16`, `fp32_to_fp16`, `fp32_to_fp8_e4m3`, `fp32_to_fp4_e2m1`, `fp32_to_nf4`, and their inverses
- **NF4 lookup table**: the 16 quantile values from N(0,1) in both fixed-point Q1.15 (for synthesis) and `real` (for testbench)
- **Display helpers**: `fp32_to_str`, `bf16_to_str`, `fp4_e2m1_to_str`, `nf4_to_str` for readable simulation logs

### `fp32_arith`

IEEE 754 single-precision add and multiply with an explicit four-stage FSM: unpack → compute → normalise → pack. Demonstrates exponent alignment for addition, mantissa multiplication, leading-zero normalisation, and overflow/underflow detection.

### `bf16_arith`

The mixed-precision pattern at the heart of modern ML training: BF16 inputs → upcast to FP32 → 24×24-bit mantissa multiply → FP32 accumulation → truncate back to BF16. Shows fused multiply-accumulate (FMA) with an external FP32 accumulator, mirroring the tensor-core dataflow on TPUs and A100+ GPUs.

### `fp8_arith`

FP8 E4M3 (weights) × E5M2 (activations) multiply and 4-element dot product — the fundamental tensor-core operation on H100 FP8 mode. Includes upcast to FP32, computation, and downcast to both E4M3 and E5M2 for storage.

### `fp4_arith`

Demonstrates both FP4 variants used in production:

- **E2M1 (MXFP4)**: Multiply and dot product with MX block scaling via the E8M0 shared exponent. Each element's value = `fp4_value × 2^(block_exp − 127)`.
- **NF4 (QLoRA)**: Dequantise via 16-entry quantile LUT × absmax, compute in FP32, requantise. Shows the complete storage→compute→storage round-trip.

### `int8_arith`

INT8 quantised inference pipeline: 4-wide vector multiply-accumulate with INT32 accumulation, ReLU activation, and the requantisation step (`output = round(acc × scale) + zero_point`) that converts INT32 accumulators back to INT8 between layers. Includes saturation detection.

### `type_convert`

Format conversion and normalisation hub:
- **Broadcast convert**: FP32 → all nine formats simultaneously
- **Normalise**: `y = x × scale + bias` (simulating LayerNorm's affine step)
- **INT8 quantise/dequantise**: full round-trip with configurable scale and zero-point
- **Min/Max tracking**: running calibration for quantisation range finding

## What the Testbenches Demonstrate

| Testbench | Key Tests |
|-----------|-----------|
| `tb_fp32_arith` | Simple and edge-case add/mul, overflow, underflow, special values (±inf) |
| `tb_bf16_arith` | Precision loss from BF16 truncation, FMA accumulation, near-unity values |
| `tb_fp8_arith` | E4M3×E5M2 products, 4-element dot product, range/precision summary |
| `tb_fp4_arith` | Full enumeration of all 16 E2M1 and NF4 encodings, MX block scaling with varied exponents, side-by-side quantisation error table |
| `tb_int8_arith` | Vector MAC, ReLU, saturation, requantisation with scale/zero-point |
| `tb_type_convert` | Cross-format conversion of representative values (π, 0.1, 65504, etc.), normalisation, INT8 round-trip error |

## Design Notes

**Educational, not production.** These modules prioritise clarity over area/timing. A production FPU would differ in several ways:

- **Rounding modes**: only truncation (round-toward-zero) is implemented; IEEE 754 requires round-to-nearest-even by default.
- **Denormals**: minimal support. Full denormal handling adds significant logic.
- **Pipelining**: each module uses a multi-cycle FSM. A real tensor core would be deeply pipelined with single-cycle throughput.
- **NF4 LUT**: the `real`-based quantisation search loop is not synthesisable as-is. A hardware NF4 dequantiser would use a 16×16-bit ROM and a single multiply.
- **MX block scaling**: demonstrated per-element; a real MXFP4 unit would process entire 32-element blocks in parallel with shared exponent logic.

## The Report

`nn_datatypes_report.docx` contains:

1. Detailed description of each format's encoding and design rationale
2. Comparison table covering bits, layout, range, precision, use case, and hardware support
3. Hardware throughput comparison (A100, H100, B200, TPU v5e)
4. Precision and quantisation error analysis (machine epsilon, dynamic range, FP4 block scaling, INT8 calibration)
5. Format selection guidelines for training, inference, and research
6. SystemVerilog implementation notes

## References

- IEEE 754-2019: *Floating-Point Arithmetic*
- OCP Microscaling Formats (MX) Specification v1.0 — defines MXFP4 (E2M1), MXFP8, and E8M0 block scales
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (NeurIPS 2023) — introduces NF4
- Micikevicius et al., *FP8 Formats for Deep Learning* (arXiv 2209.05433)
- NVIDIA, *TensorFloat-32 in the A100 GPU Accelerates AI Training* (2020)
- NVIDIA Blackwell Architecture Whitepaper (2024) — NVFP4 and FP4 tensor core support

## Licence

MIT

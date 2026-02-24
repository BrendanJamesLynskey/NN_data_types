// =============================================================================
// nn_dtypes_pkg.sv - Neural Network Data Types Package
// =============================================================================
// Defines packed structs and helper functions for the major floating-point and
// integer formats used in modern neural-network accelerators.
//
// Formats covered:
//   FP32    (IEEE 754 binary32)       - 1/8/23   sign/exp/mantissa
//   FP16    (IEEE 754 binary16)       - 1/5/10
//   BF16    (Brain Floating Point)    - 1/8/7
//   TF32    (TensorFloat-32, NVIDIA)  - 1/8/10   (19 bits total)
//   FP8_E4M3 (FP8, OCP/NVIDIA)       - 1/4/3
//   FP8_E5M2 (FP8, OCP/NVIDIA)       - 1/5/2
//   FP4_E2M1 (MXFP4, OCP/Blackwell)  - 1/2/1   (4 bits total)
//   NF4      (NormalFloat4, QLoRA)    - 4-bit LUT-based (16 quantiles)
//   INT8    (Signed 8-bit integer)
// =============================================================================

package nn_dtypes_pkg;

  // -------------------------------------------------------------------------
  // Packed type definitions
  // -------------------------------------------------------------------------

  // FP32 - IEEE 754 single precision
  typedef struct packed {
    logic        sign;
    logic [7:0]  exponent;   // bias = 127
    logic [22:0] mantissa;
  } fp32_t;

  // FP16 - IEEE 754 half precision
  typedef struct packed {
    logic        sign;
    logic [4:0]  exponent;   // bias = 15
    logic [9:0]  mantissa;
  } fp16_t;

  // BF16 - Brain Floating Point 16
  typedef struct packed {
    logic        sign;
    logic [7:0]  exponent;   // bias = 127 (same range as FP32)
    logic [6:0]  mantissa;
  } bf16_t;

  // TF32 - TensorFloat-32 (19-bit)
  typedef struct packed {
    logic        sign;
    logic [7:0]  exponent;   // bias = 127
    logic [9:0]  mantissa;
  } tf32_t;

  // FP8 E4M3 - 4-bit exponent, 3-bit mantissa
  typedef struct packed {
    logic       sign;
    logic [3:0] exponent;    // bias = 7
    logic [2:0] mantissa;
  } fp8_e4m3_t;

  // FP8 E5M2 - 5-bit exponent, 2-bit mantissa
  typedef struct packed {
    logic       sign;
    logic [4:0] exponent;    // bias = 15
    logic [1:0] mantissa;
  } fp8_e5m2_t;

  // FP4 E2M1 - OCP Microscaling format (MXFP4, Blackwell/CDNA4)
  // 1-bit sign, 2-bit exponent, 1-bit mantissa
  // No Inf, no NaN. Range: ±6.0. Only 16 total encodings.
  typedef struct packed {
    logic       sign;
    logic [1:0] exponent;    // bias = 1
    logic       mantissa;
  } fp4_e2m1_t;

  // NF4 - NormalFloat4 (QLoRA / bitsandbytes)
  // NOT a conventional float — uses a 4-bit index into a 16-entry
  // lookup table of quantiles from the standard normal distribution.
  // Stored as a 4-bit unsigned index; dequantised via LUT × block scale.
  typedef logic [3:0] nf4_t;

  // INT8 - Signed 8-bit integer (two's complement)
  typedef logic signed [7:0] int8_t;

  // -------------------------------------------------------------------------
  // Constants
  // -------------------------------------------------------------------------

  // Exponent biases
  localparam int FP32_BIAS     = 127;
  localparam int FP16_BIAS     = 15;
  localparam int BF16_BIAS     = 127;
  localparam int TF32_BIAS     = 127;
  localparam int FP8_E4M3_BIAS = 7;
  localparam int FP8_E5M2_BIAS = 15;
  localparam int FP4_E2M1_BIAS = 1;

  // NF4 lookup table: 16 quantiles of N(0,1), from QLoRA (Dettmers et al.)
  // Index 0..15 maps to these normalised values in [-1.0, +1.0].
  // Actual value = nf4_lut[index] × block_absmax.
  // We store as fixed-point Q1.15 (signed, 16-bit) for synthesisable logic.
  // Real values:
  //   { -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
  //      0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230, 1.0 }
  localparam logic signed [15:0] NF4_LUT [0:15] = '{
    -16'sd32768,  // -1.0000  (index 0)
    -16'sd22817,  // -0.6962
    -16'sd17208,  // -0.5251
    -16'sd12941,  // -0.3949
    -16'sd9320,   // -0.2844
    -16'sd6055,   // -0.1848
    -16'sd2985,   // -0.0911
     16'sd0,      //  0.0000  (index 7)
     16'sd2608,   //  0.0796
     16'sd5273,   //  0.1609
     16'sd8066,   //  0.2461
     16'sd11073,  //  0.3379
     16'sd14440,  //  0.4407
     16'sd18432,  //  0.5626
     16'sd23691,  //  0.7230
     16'sd32767   //  1.0000  (index 15)
  };

  // NF4 real-valued LUT (for testbench use with $realtobits)
  localparam real NF4_REAL_LUT [0:15] = '{
    -1.0,       -0.6961928, -0.5250731, -0.3949175,
    -0.2844414, -0.1847734, -0.0910500,  0.0,
     0.0795803,  0.1609302,  0.2461123,  0.3379350,
     0.4407233,  0.5626170,  0.7229568,  1.0
  };

  // Special values - FP32
  localparam fp32_t FP32_POS_ZERO = '{sign: 1'b0, exponent: 8'h00, mantissa: 23'h000000};
  localparam fp32_t FP32_NEG_ZERO = '{sign: 1'b1, exponent: 8'h00, mantissa: 23'h000000};
  localparam fp32_t FP32_POS_INF  = '{sign: 1'b0, exponent: 8'hFF, mantissa: 23'h000000};
  localparam fp32_t FP32_NEG_INF  = '{sign: 1'b1, exponent: 8'hFF, mantissa: 23'h000000};
  localparam fp32_t FP32_NAN      = '{sign: 1'b0, exponent: 8'hFF, mantissa: 23'h400000};

  // Special values - FP16
  localparam fp16_t FP16_POS_INF  = '{sign: 1'b0, exponent: 5'h1F, mantissa: 10'h000};
  localparam fp16_t FP16_NAN      = '{sign: 1'b0, exponent: 5'h1F, mantissa: 10'h200};

  // Special values - BF16
  localparam bf16_t BF16_POS_INF  = '{sign: 1'b0, exponent: 8'hFF, mantissa: 7'h00};
  localparam bf16_t BF16_NAN      = '{sign: 1'b0, exponent: 8'hFF, mantissa: 7'h40};

  // -------------------------------------------------------------------------
  // Helper Functions: Real ↔ FP32 conversion (for testbench use)
  // -------------------------------------------------------------------------

  function automatic fp32_t real_to_fp32(input real val);
    fp32_t result;
    logic [31:0] bits;
    bits = $shortrealtobits(shortreal'(val));
    result.sign     = bits[31];
    result.exponent = bits[30:23];
    result.mantissa = bits[22:0];
    return result;
  endfunction

  function automatic real fp32_to_real(input fp32_t val);
    logic [31:0] bits;
    bits = {val.sign, val.exponent, val.mantissa};
    return real'($bitstoshortreal(bits));
  endfunction

  // -------------------------------------------------------------------------
  // Helper Functions: Type conversions (truncation-based downcast)
  // -------------------------------------------------------------------------

  // FP32 → BF16 (truncate lower 16 mantissa bits — the "cheap cast")
  function automatic bf16_t fp32_to_bf16(input fp32_t val);
    bf16_t result;
    result.sign     = val.sign;
    result.exponent = val.exponent;
    result.mantissa = val.mantissa[22:16];
    return result;
  endfunction

  // BF16 → FP32 (zero-extend mantissa)
  function automatic fp32_t bf16_to_fp32(input bf16_t val);
    fp32_t result;
    result.sign     = val.sign;
    result.exponent = val.exponent;
    result.mantissa = {val.mantissa, 16'h0000};
    return result;
  endfunction

  // FP32 → FP16 (simplified truncation, no rounding/overflow handling)
  function automatic fp16_t fp32_to_fp16(input fp32_t val);
    fp16_t result;
    int    new_exp;
    result.sign = val.sign;
    new_exp = int'(val.exponent) - FP32_BIAS + FP16_BIAS;
    if (new_exp >= 31) begin
      result.exponent = 5'h1F;
      result.mantissa = 10'h000; // infinity
    end else if (new_exp <= 0) begin
      result.exponent = 5'h00;
      result.mantissa = 10'h000; // flush to zero
    end else begin
      result.exponent = new_exp[4:0];
      result.mantissa = val.mantissa[22:13];
    end
    return result;
  endfunction

  // FP32 → TF32 (truncate lower 13 mantissa bits)
  function automatic tf32_t fp32_to_tf32(input fp32_t val);
    tf32_t result;
    result.sign     = val.sign;
    result.exponent = val.exponent;
    result.mantissa = val.mantissa[22:13];
    return result;
  endfunction

  // FP32 → FP8 E4M3
  function automatic fp8_e4m3_t fp32_to_fp8_e4m3(input fp32_t val);
    fp8_e4m3_t result;
    int new_exp;
    result.sign = val.sign;
    new_exp = int'(val.exponent) - FP32_BIAS + FP8_E4M3_BIAS;
    if (new_exp >= 15) begin
      // Saturate to max normal (E4M3 has no inf; max = 1|1110|111 = 448)
      result.exponent = 4'hE;
      result.mantissa = 3'h7;
    end else if (new_exp <= 0) begin
      result.exponent = 4'h0;
      result.mantissa = 3'h0;
    end else begin
      result.exponent = new_exp[3:0];
      result.mantissa = val.mantissa[22:20];
    end
    return result;
  endfunction

  // FP32 → FP8 E5M2
  function automatic fp8_e5m2_t fp32_to_fp8_e5m2(input fp32_t val);
    fp8_e5m2_t result;
    int new_exp;
    result.sign = val.sign;
    new_exp = int'(val.exponent) - FP32_BIAS + FP8_E5M2_BIAS;
    if (new_exp >= 31) begin
      result.exponent = 5'h1F;
      result.mantissa = 2'h0; // infinity
    end else if (new_exp <= 0) begin
      result.exponent = 5'h0;
      result.mantissa = 2'h0;
    end else begin
      result.exponent = new_exp[4:0];
      result.mantissa = val.mantissa[22:21];
    end
    return result;
  endfunction

  // FP32 → FP4 E2M1 (OCP MXFP4)
  // E2M1 encoding table (positive values; negate for sign=1):
  //   S|EE|M  → value
  //   0|00|0  → +0.0        0|00|1 → +0.5 (subnormal)
  //   0|01|0  → +1.0        0|01|1 → +1.5
  //   0|10|0  → +2.0        0|10|1 → +3.0
  //   0|11|0  → +4.0        0|11|1 → +6.0 (max)
  function automatic fp4_e2m1_t fp32_to_fp4_e2m1(input fp32_t val);
    fp4_e2m1_t result;
    int new_exp;
    result.sign = val.sign;
    if (val.exponent == 0) begin
      // Input is zero or denormal → FP4 zero
      result.exponent = 2'b00;
      result.mantissa = 1'b0;
    end else begin
      new_exp = int'(val.exponent) - FP32_BIAS + FP4_E2M1_BIAS;
      if (new_exp >= 3) begin
        // Saturate to max normal ±6.0 (no inf in E2M1)
        result.exponent = 2'b11;
        result.mantissa = 1'b1;
      end else if (new_exp <= 0) begin
        // Subnormal region: only ±0.5 is representable
        if (new_exp == 0 && val.mantissa != 0) begin
          result.exponent = 2'b00;
          result.mantissa = 1'b1; // 0.5
        end else begin
          result.exponent = 2'b00;
          result.mantissa = 1'b0; // flush to zero
        end
      end else begin
        result.exponent = new_exp[1:0];
        result.mantissa = val.mantissa[22]; // single mantissa bit
      end
    end
    return result;
  endfunction

  // FP4 E2M1 → FP32
  function automatic fp32_t fp4_e2m1_to_fp32(input fp4_e2m1_t val);
    fp32_t r;
    r.sign = val.sign;
    if (val.exponent == 0 && val.mantissa == 0) begin
      r.exponent = 8'h00;
      r.mantissa = 23'h0;
    end else if (val.exponent == 0) begin
      // Subnormal: 0.5 = 2^(1-1) × 0.5 = 2^(-1)
      r.exponent = FP32_BIAS - 1;  // 126
      r.mantissa = 23'h0;          // 1.0 × 2^(-1) = 0.5
    end else begin
      automatic int biased = int'(val.exponent) - FP4_E2M1_BIAS + FP32_BIAS;
      r.exponent = biased[7:0];
      r.mantissa = {val.mantissa, 22'h0};
    end
    return r;
  endfunction

  // FP32 → NF4 (find nearest LUT entry after normalisation by absmax)
  // In practice, absmax is the block scale provided externally.
  function automatic nf4_t fp32_to_nf4(input fp32_t val, input fp32_t absmax);
    real x, normalised;
    real best_dist;
    int  best_idx;
    x = fp32_to_real(val);
    automatic real am = fp32_to_real(absmax);
    if (am == 0.0)
      return 4'd7; // map to zero entry
    normalised = x / am;
    // Clamp to [-1, 1]
    if (normalised > 1.0) normalised = 1.0;
    if (normalised < -1.0) normalised = -1.0;
    // Find nearest LUT entry
    best_dist = 1.0e30;
    best_idx  = 0;
    for (int i = 0; i < 16; i++) begin
      automatic real dist = (normalised - NF4_REAL_LUT[i]) >= 0 ?
                            (normalised - NF4_REAL_LUT[i]) :
                            -(normalised - NF4_REAL_LUT[i]);
      if (dist < best_dist) begin
        best_dist = dist;
        best_idx  = i;
      end
    end
    return nf4_t'(best_idx);
  endfunction

  // NF4 → FP32 (dequantise: lookup × absmax)
  function automatic fp32_t nf4_to_fp32(input nf4_t idx, input fp32_t absmax);
    real dequant;
    dequant = NF4_REAL_LUT[int'(idx)] * fp32_to_real(absmax);
    return real_to_fp32(dequant);
  endfunction

  // -------------------------------------------------------------------------
  // -------------------------------------------------------------------------

  function automatic string fp32_to_str(input fp32_t val);
    return $sformatf("FP32[s=%0b e=%0d(%02h) m=%06h] = %0g",
                     val.sign, int'(val.exponent) - FP32_BIAS,
                     val.exponent, val.mantissa,
                     fp32_to_real(val));
  endfunction

  function automatic string bf16_to_str(input bf16_t val);
    return $sformatf("BF16[s=%0b e=%0d(%02h) m=%02h]",
                     val.sign, int'(val.exponent) - BF16_BIAS,
                     val.exponent, val.mantissa);
  endfunction

  function automatic string fp4_e2m1_to_str(input fp4_e2m1_t val);
    return $sformatf("FP4[s=%0b e=%0b m=%0b] = %0g",
                     val.sign, val.exponent, val.mantissa,
                     fp32_to_real(fp4_e2m1_to_fp32(val)));
  endfunction

  function automatic string nf4_to_str(input nf4_t idx, input fp32_t absmax);
    return $sformatf("NF4[idx=%0d lut=%0g] × %0g = %0g",
                     int'(idx), NF4_REAL_LUT[int'(idx)],
                     fp32_to_real(absmax),
                     fp32_to_real(nf4_to_fp32(idx, absmax)));
  endfunction

endpackage

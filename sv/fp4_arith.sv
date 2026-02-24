// =============================================================================
// fp4_arith.sv - FP4 Arithmetic Operations (E2M1 and NF4)
// =============================================================================
// Demonstrates FP4 arithmetic as used in ultra-low-precision inference and
// QLoRA fine-tuning.
//
// FP4 E2M1 (OCP MXFP4): Hardware-native on NVIDIA Blackwell (B200) and
//   AMD CDNA4. Uses microscaling: 32 FP4 elements share an E8M0 block
//   exponent. Range: ±6.0, only 16 distinct values per sign.
//   Values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} × {+,-}
//
// NF4 (NormalFloat4, QLoRA): Software-defined lookup-table format from
//   bitsandbytes. 16 values chosen as quantiles of N(0,1), optimal for
//   normally-distributed neural network weights. Block-wise absmax scaling.
//   All compute happens in BF16/FP16 after dequantisation.
//
// Both formats follow the same pattern for compute:
//   1. Dequantise FP4 → FP32 (using scale factor)
//   2. Compute in higher precision
//   3. Quantise result back to FP4
// =============================================================================

module fp4_arith
  import nn_dtypes_pkg::*;
(
  input  logic       clk,
  input  logic       rst_n,

  // FP4 E2M1 operands
  input  fp4_e2m1_t  e2m1_a,
  input  fp4_e2m1_t  e2m1_b,

  // NF4 operands (index + block scale)
  input  nf4_t       nf4_a,
  input  nf4_t       nf4_b,
  input  fp32_t      nf4_scale_a,   // absmax for block containing a
  input  fp32_t      nf4_scale_b,   // absmax for block containing b

  // MX block scale (E8M0: 8-bit unsigned exponent, value = 2^(e-127))
  input  logic [7:0] mx_block_exp,  // shared exponent for MXFP4 block

  // Control
  input  logic       start,
  input  logic [2:0] op,
  // 000 = E2M1 multiply (with MX block scale)
  // 001 = E2M1 4-element dot product
  // 010 = NF4 dequant + multiply
  // 011 = NF4 4-element dot product
  // 100 = E2M1 enumerate (show all 16 values)
  // 101 = NF4 enumerate (show all 16 LUT values)

  // Additional dot-product operands
  input  fp4_e2m1_t  e2m1_a1, e2m1_a2, e2m1_a3,
  input  fp4_e2m1_t  e2m1_b1, e2m1_b2, e2m1_b3,
  input  nf4_t       nf4_a1, nf4_a2, nf4_a3,
  input  nf4_t       nf4_b1, nf4_b2, nf4_b3,

  // Outputs
  output fp32_t      result_fp32,
  output fp4_e2m1_t  result_e2m1,
  output nf4_t       result_nf4,
  output logic       done
);

  // -----------------------------------------------------------------------
  // MX Block scale: value = 2^(mx_block_exp - 127)
  // Applied to each E2M1 element in the block after dequant
  // -----------------------------------------------------------------------
  function automatic real mx_scale_real(input logic [7:0] exp);
    return 2.0 ** (real'(int'(exp)) - 127.0);
  endfunction

  // E2M1 → real (with MX block scale applied)
  function automatic real e2m1_to_real_scaled(input fp4_e2m1_t val,
                                              input logic [7:0] blk_exp);
    automatic real base = fp32_to_real(fp4_e2m1_to_fp32(val));
    return base * mx_scale_real(blk_exp);
  endfunction

  // -----------------------------------------------------------------------
  // FSM
  // -----------------------------------------------------------------------
  typedef enum logic [1:0] { IDLE, COMPUTE, QUANTISE, FINISH } state_t;
  state_t state;

  real intermediate;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= IDLE;
      done        <= 1'b0;
      result_fp32 <= '0;
      result_e2m1 <= '0;
      result_nf4  <= '0;
    end else begin
      case (state)
        IDLE: begin
          done <= 1'b0;
          if (start) state <= COMPUTE;
        end

        COMPUTE: begin
          case (op)
            3'b000: begin // E2M1 multiply with MX block scale
              automatic real a_r = e2m1_to_real_scaled(e2m1_a, mx_block_exp);
              automatic real b_r = e2m1_to_real_scaled(e2m1_b, mx_block_exp);
              intermediate <= a_r * b_r;
            end

            3'b001: begin // E2M1 4-element dot product
              automatic real dot = 0.0;
              dot += e2m1_to_real_scaled(e2m1_a,  mx_block_exp) *
                     e2m1_to_real_scaled(e2m1_b,  mx_block_exp);
              dot += e2m1_to_real_scaled(e2m1_a1, mx_block_exp) *
                     e2m1_to_real_scaled(e2m1_b1, mx_block_exp);
              dot += e2m1_to_real_scaled(e2m1_a2, mx_block_exp) *
                     e2m1_to_real_scaled(e2m1_b2, mx_block_exp);
              dot += e2m1_to_real_scaled(e2m1_a3, mx_block_exp) *
                     e2m1_to_real_scaled(e2m1_b3, mx_block_exp);
              intermediate <= dot;
            end

            3'b010: begin // NF4 dequant + multiply
              automatic real a_r = fp32_to_real(nf4_to_fp32(nf4_a, nf4_scale_a));
              automatic real b_r = fp32_to_real(nf4_to_fp32(nf4_b, nf4_scale_b));
              intermediate <= a_r * b_r;
            end

            3'b011: begin // NF4 4-element dot product
              automatic real dot = 0.0;
              dot += fp32_to_real(nf4_to_fp32(nf4_a,  nf4_scale_a)) *
                     fp32_to_real(nf4_to_fp32(nf4_b,  nf4_scale_b));
              dot += fp32_to_real(nf4_to_fp32(nf4_a1, nf4_scale_a)) *
                     fp32_to_real(nf4_to_fp32(nf4_b1, nf4_scale_b));
              dot += fp32_to_real(nf4_to_fp32(nf4_a2, nf4_scale_a)) *
                     fp32_to_real(nf4_to_fp32(nf4_b2, nf4_scale_b));
              dot += fp32_to_real(nf4_to_fp32(nf4_a3, nf4_scale_a)) *
                     fp32_to_real(nf4_to_fp32(nf4_b3, nf4_scale_b));
              intermediate <= dot;
            end

            default: intermediate <= 0.0;
          endcase
          state <= QUANTISE;
        end

        QUANTISE: begin
          result_fp32 <= real_to_fp32(intermediate);
          result_e2m1 <= fp32_to_fp4_e2m1(real_to_fp32(intermediate));
          // For NF4 requant, use |result| as block scale
          automatic real absres = intermediate >= 0.0 ? intermediate : -intermediate;
          if (absres == 0.0) absres = 1.0;
          result_nf4  <= fp32_to_nf4(real_to_fp32(intermediate),
                                     real_to_fp32(absres));
          state <= FINISH;
        end

        FINISH: begin
          done  <= 1'b1;
          state <= IDLE;
        end
      endcase
    end
  end

endmodule

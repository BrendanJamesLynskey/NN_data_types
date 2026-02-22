// =============================================================================
// bf16_arith.sv - BF16 Arithmetic with FP32 Accumulation
// =============================================================================
// Demonstrates the most common mixed-precision pattern in modern ML training:
//   - Inputs are BF16
//   - Multiply produces an FP32 intermediate
//   - Accumulate in FP32
//   - Truncate result back to BF16
//
// This mirrors what happens inside a TPU or GPU tensor core.
// =============================================================================

module bf16_arith
  import nn_dtypes_pkg::*;
(
  input  logic  clk,
  input  logic  rst_n,

  // BF16 operands
  input  bf16_t a,
  input  bf16_t b,

  // Control
  input  logic  start,
  input  logic  op,        // 0 = multiply-only, 1 = fused multiply-accumulate (a*b + acc)

  // FP32 accumulator (externally managed for chaining)
  input  fp32_t acc_in,
  output fp32_t acc_out,   // FP32 result of operation

  // BF16 truncated output (for storage / next-layer input)
  output bf16_t result_bf16,

  output logic  done
);

  // -----------------------------------------------------------------------
  // Stage 1: Upcast BF16 → FP32 and multiply
  // -----------------------------------------------------------------------
  fp32_t a_fp32, b_fp32;
  fp32_t product;
  logic  prod_sign;
  int    prod_exp;
  logic [47:0] prod_mant;

  // -----------------------------------------------------------------------
  // FSM
  // -----------------------------------------------------------------------
  typedef enum logic [1:0] { IDLE, MULTIPLY, ACCUMULATE, OUTPUT } state_t;
  state_t state;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= IDLE;
      done        <= 1'b0;
      acc_out     <= '0;
      result_bf16 <= '0;
    end else begin
      case (state)
        IDLE: begin
          done <= 1'b0;
          if (start) begin
            // Upcast BF16 operands to FP32
            a_fp32 <= bf16_to_fp32(a);
            b_fp32 <= bf16_to_fp32(b);
            state  <= MULTIPLY;
          end
        end

        MULTIPLY: begin
          // Compute product in FP32 precision
          prod_sign <= a_fp32.sign ^ b_fp32.sign;

          // Exponents (unbiased)
          automatic int ea = int'(a_fp32.exponent) - FP32_BIAS;
          automatic int eb = int'(b_fp32.exponent) - FP32_BIAS;
          prod_exp <= ea + eb;

          // Mantissa multiply (24×24 → 48 bits)
          automatic logic [23:0] ma = (a_fp32.exponent != 0) ?
                                      {1'b1, a_fp32.mantissa} : {1'b0, a_fp32.mantissa};
          automatic logic [23:0] mb = (b_fp32.exponent != 0) ?
                                      {1'b1, b_fp32.mantissa} : {1'b0, b_fp32.mantissa};
          prod_mant <= ma * mb;

          state <= ACCUMULATE;
        end

        ACCUMULATE: begin
          // Normalise product
          automatic logic [47:0] nm = prod_mant;
          automatic int          ne = prod_exp;

          // Product of two 1.xxx numbers is in range [1.0, 4.0)
          if (nm[47]) begin
            nm = nm >> 1;
            ne = ne + 1;
          end
          // find leading 1
          while (nm != 0 && !nm[46]) begin
            nm = nm << 1;
            ne = ne - 1;
          end

          // Pack into FP32
          automatic int biased = ne + FP32_BIAS;
          if (nm == 0) begin
            product <= FP32_POS_ZERO;
          end else if (biased >= 255) begin
            product <= (prod_sign) ? FP32_NEG_INF : FP32_POS_INF;
          end else if (biased <= 0) begin
            product <= FP32_POS_ZERO;
            product.sign <= prod_sign;
          end else begin
            product.sign     <= prod_sign;
            product.exponent <= biased[7:0];
            product.mantissa <= nm[45:23];
          end

          state <= OUTPUT;
        end

        OUTPUT: begin
          if (op) begin
            // FMA: product + accumulator
            // Simplified: use real arithmetic for accumulation demo
            automatic real p = fp32_to_real(product);
            automatic real a_val = fp32_to_real(acc_in);
            automatic real sum = p + a_val;
            acc_out <= real_to_fp32(sum);
          end else begin
            acc_out <= product;
          end

          // Truncate FP32 → BF16 for output
          result_bf16 <= fp32_to_bf16(op ? real_to_fp32(fp32_to_real(product) + fp32_to_real(acc_in)) : product);

          done  <= 1'b1;
          state <= IDLE;
        end
      endcase
    end
  end

endmodule

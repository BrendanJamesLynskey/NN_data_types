// =============================================================================
// fp32_arith.sv - FP32 Arithmetic Operations
// =============================================================================
// Implements basic IEEE 754 single-precision operations for educational
// purposes. Demonstrates the full pipeline: unpack → compute → normalise →
// pack. Not a production FPU (no rounding modes, denormal support is minimal).
// =============================================================================

module fp32_arith
  import nn_dtypes_pkg::*;
(
  input  logic  clk,
  input  logic  rst_n,
  input  fp32_t a,
  input  fp32_t b,
  input  logic  start,
  input  logic  op,       // 0 = add, 1 = multiply
  output fp32_t result,
  output logic  done,
  output logic  overflow,
  output logic  underflow
);

  // -----------------------------------------------------------------------
  // Internal signals
  // -----------------------------------------------------------------------
  typedef enum logic [1:0] { IDLE, COMPUTE, NORMALISE, DONE } state_t;
  state_t state;

  // Extended mantissa with implicit 1 and guard bits
  logic [47:0] mant_a, mant_b;
  logic [47:0] mant_sum;
  logic [47:0] mant_prod;
  int          exp_a, exp_b, exp_r;
  logic        sign_r;

  // Normalisation shift
  logic [47:0] mant_r;
  int          shift;

  // -----------------------------------------------------------------------
  // Unpack helper: extract implicit leading 1
  // -----------------------------------------------------------------------
  function automatic logic [47:0] unpack_mantissa(input fp32_t val);
    if (val.exponent == 8'h00)
      return {1'b0, val.mantissa, 24'b0};  // denormal
    else
      return {1'b1, val.mantissa, 24'b0};  // normal: implicit 1
  endfunction

  // -----------------------------------------------------------------------
  // FSM
  // -----------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state     <= IDLE;
      result    <= '0;
      done      <= 1'b0;
      overflow  <= 1'b0;
      underflow <= 1'b0;
    end else begin
      case (state)
        // -----------------------------------------------------------------
        IDLE: begin
          done <= 1'b0;
          if (start) begin
            // Unpack operands
            mant_a <= unpack_mantissa(a);
            mant_b <= unpack_mantissa(b);
            exp_a  <= int'(a.exponent) - FP32_BIAS;
            exp_b  <= int'(b.exponent) - FP32_BIAS;
            state  <= COMPUTE;
          end
        end

        // -----------------------------------------------------------------
        COMPUTE: begin
          if (!op) begin
            // === ADDITION ===
            // Align exponents: shift smaller mantissa right
            automatic int exp_diff;
            automatic logic [47:0] aligned_a, aligned_b;
            if (exp_a >= exp_b) begin
              exp_r     = exp_a;
              aligned_a = mant_a;
              exp_diff  = exp_a - exp_b;
              aligned_b = (exp_diff < 48) ? (mant_b >> exp_diff) : 48'b0;
            end else begin
              exp_r     = exp_b;
              aligned_b = mant_b;
              exp_diff  = exp_b - exp_a;
              aligned_a = (exp_diff < 48) ? (mant_a >> exp_diff) : 48'b0;
            end

            if (a.sign == b.sign) begin
              sign_r  <= a.sign;
              mant_r  <= aligned_a + aligned_b;
            end else begin
              if (aligned_a >= aligned_b) begin
                sign_r <= a.sign;
                mant_r <= aligned_a - aligned_b;
              end else begin
                sign_r <= b.sign;
                mant_r <= aligned_b - aligned_a;
              end
            end
          end else begin
            // === MULTIPLICATION ===
            sign_r <= a.sign ^ b.sign;
            exp_r  <= exp_a + exp_b;
            // Multiply upper 24 bits (1.mantissa × 1.mantissa)
            mant_r <= (mant_a[47:24] * mant_b[47:24]);
          end
          state <= NORMALISE;
        end

        // -----------------------------------------------------------------
        NORMALISE: begin
          overflow  <= 1'b0;
          underflow <= 1'b0;

          if (mant_r == 0) begin
            result <= FP32_POS_ZERO;
            result.sign <= sign_r;
          end else begin
            // Shift left until MSB is in bit 47 (for add) or bit 47 (for mul)
            automatic logic [47:0] norm_mant = mant_r;
            automatic int          norm_exp  = exp_r;

            // Handle carry out (mantissa >= 2.0)
            if (norm_mant[47]) begin
              norm_mant = norm_mant >> 1;
              norm_exp  = norm_exp + 1;
            end

            // Leading-zero normalisation
            while (norm_mant != 0 && !norm_mant[46]) begin
              norm_mant = norm_mant << 1;
              norm_exp  = norm_exp - 1;
            end

            // Biased exponent
            automatic int biased_exp = norm_exp + FP32_BIAS;

            if (biased_exp >= 255) begin
              result   <= (sign_r) ? FP32_NEG_INF : FP32_POS_INF;
              overflow <= 1'b1;
            end else if (biased_exp <= 0) begin
              result    <= FP32_POS_ZERO;
              result.sign <= sign_r;
              underflow <= 1'b1;
            end else begin
              result.sign     <= sign_r;
              result.exponent <= biased_exp[7:0];
              result.mantissa <= norm_mant[45:23]; // strip implicit 1
            end
          end
          state <= DONE;
        end

        // -----------------------------------------------------------------
        DONE: begin
          done  <= 1'b1;
          state <= IDLE;
        end
      endcase
    end
  end

endmodule

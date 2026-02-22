// =============================================================================
// fp8_arith.sv - FP8 Arithmetic Operations (E4M3 and E5M2)
// =============================================================================
// Demonstrates FP8 arithmetic as used in modern inference accelerators
// (NVIDIA H100, AMD MI300). FP8 operations are typically performed by:
//   1. Upcasting FP8 → FP16/FP32
//   2. Computing in higher precision
//   3. Downcasting result back to FP8
//
// E4M3: Higher precision (3 mantissa bits), smaller range — used for weights
// E5M2: Lower precision (2 mantissa bits), larger range  — used for gradients
// =============================================================================

module fp8_arith
  import nn_dtypes_pkg::*;
(
  input  logic       clk,
  input  logic       rst_n,

  // E4M3 operand (typically weights)
  input  fp8_e4m3_t  weight,
  // E5M2 operand (typically activations/gradients)
  input  fp8_e5m2_t  activation,

  input  logic       start,
  input  logic [1:0] op,      // 00=mul, 01=add_e4m3, 10=dot4 (4-element dot product)

  // Additional operands for dot product
  input  fp8_e4m3_t  w1, w2, w3,
  input  fp8_e5m2_t  a1, a2, a3,

  // Outputs
  output fp32_t      result_fp32,   // Full-precision result
  output fp8_e4m3_t  result_e4m3,   // Quantised back to E4M3
  output fp8_e5m2_t  result_e5m2,   // Quantised back to E5M2
  output logic       done
);

  // -----------------------------------------------------------------------
  // FP8 → FP32 upcast functions
  // -----------------------------------------------------------------------
  function automatic fp32_t e4m3_to_fp32(input fp8_e4m3_t val);
    fp32_t r;
    r.sign = val.sign;
    if (val.exponent == 0 && val.mantissa == 0) begin
      r.exponent = 8'h00;
      r.mantissa = 23'h0;
    end else if (val.exponent == 0) begin
      // Denormal: value = (-1)^s × 0.mantissa × 2^(1-bias)
      r.exponent = 8'h00;
      r.mantissa = {val.mantissa, 20'h0};
    end else begin
      // Normal: rebias exponent
      automatic int new_exp = int'(val.exponent) - FP8_E4M3_BIAS + FP32_BIAS;
      r.exponent = new_exp[7:0];
      r.mantissa = {val.mantissa, 20'h0};
    end
    return r;
  endfunction

  function automatic fp32_t e5m2_to_fp32(input fp8_e5m2_t val);
    fp32_t r;
    r.sign = val.sign;
    if (val.exponent == 0 && val.mantissa == 0) begin
      r.exponent = 8'h00;
      r.mantissa = 23'h0;
    end else if (val.exponent == 5'h1F) begin
      // Inf/NaN
      r.exponent = 8'hFF;
      r.mantissa = (val.mantissa != 0) ? 23'h400000 : 23'h0;
    end else if (val.exponent == 0) begin
      r.exponent = 8'h00;
      r.mantissa = {val.mantissa, 21'h0};
    end else begin
      automatic int new_exp = int'(val.exponent) - FP8_E5M2_BIAS + FP32_BIAS;
      r.exponent = new_exp[7:0];
      r.mantissa = {val.mantissa, 21'h0};
    end
    return r;
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
      result_e4m3 <= '0;
      result_e5m2 <= '0;
    end else begin
      case (state)
        IDLE: begin
          done <= 1'b0;
          if (start) state <= COMPUTE;
        end

        COMPUTE: begin
          // Upcast to FP32 then use real arithmetic for computation
          automatic real w_real = fp32_to_real(e4m3_to_fp32(weight));
          automatic real a_real = fp32_to_real(e5m2_to_fp32(activation));

          case (op)
            2'b00: begin // Multiply
              intermediate <= w_real * a_real;
            end
            2'b01: begin // Add (both E4M3)
              intermediate <= w_real + a_real;
            end
            2'b10: begin // 4-element dot product (the core tensor-core op)
              automatic real w1r = fp32_to_real(e4m3_to_fp32(w1));
              automatic real w2r = fp32_to_real(e4m3_to_fp32(w2));
              automatic real w3r = fp32_to_real(e4m3_to_fp32(w3));
              automatic real a1r = fp32_to_real(e5m2_to_fp32(a1));
              automatic real a2r = fp32_to_real(e5m2_to_fp32(a2));
              automatic real a3r = fp32_to_real(e5m2_to_fp32(a3));
              intermediate <= (w_real * a_real) + (w1r * a1r) +
                              (w2r * a2r) + (w3r * a3r);
            end
            default: intermediate <= 0.0;
          endcase
          state <= QUANTISE;
        end

        QUANTISE: begin
          // Store full-precision result
          result_fp32 <= real_to_fp32(intermediate);

          // Quantise back to FP8 formats
          result_e4m3 <= fp32_to_fp8_e4m3(real_to_fp32(intermediate));
          result_e5m2 <= fp32_to_fp8_e5m2(real_to_fp32(intermediate));

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

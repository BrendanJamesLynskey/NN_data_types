// =============================================================================
// type_convert.sv - Format Conversion & Normalisation
// =============================================================================
// Demonstrates conversions between all NN data types and common normalisation
// operations (layer-norm style scaling).
//
// Conversion matrix (implemented):
//   FP32 → { FP16, BF16, TF32, FP8_E4M3, FP8_E5M2, INT8 }
//   BF16 → FP32
//   FP16 → FP32
//
// Also implements:
//   - Min/Max tracking (for calibration / quantisation range finding)
//   - Scale-factor computation for INT8 quantisation
//   - Simplified LayerNorm-style normalisation
// =============================================================================

module type_convert
  import nn_dtypes_pkg::*;
(
  input  logic       clk,
  input  logic       rst_n,

  // Input value (FP32)
  input  fp32_t      in_fp32,

  // Quantisation parameters
  input  fp32_t      scale_factor,   // For normalisation: x * scale + bias
  input  fp32_t      bias,
  input  fp32_t      qscale,         // For INT8: scale = (max-min) / 255
  input  int8_t      qzero,          // For INT8: zero_point

  // Control
  input  logic       start,
  input  logic [2:0] op,
  // 000 = FP32→all (broadcast convert)
  // 001 = Normalise (x * scale + bias)
  // 010 = INT8 quantise (FP32 → INT8)
  // 011 = INT8 dequantise (INT8 → FP32)
  // 100 = Min/Max update

  input  int8_t      in_int8,        // For dequantise

  // Converted outputs
  output fp16_t      out_fp16,
  output bf16_t      out_bf16,
  output tf32_t      out_tf32,
  output fp8_e4m3_t  out_e4m3,
  output fp8_e5m2_t  out_e5m2,
  output int8_t      out_int8,
  output fp32_t      out_fp32,

  // FP4 outputs
  output fp4_e2m1_t  out_fp4,
  output nf4_t       out_nf4,

  // Min/Max tracking
  output fp32_t      running_min,
  output fp32_t      running_max,

  output logic       done
);

  // -----------------------------------------------------------------------
  // Min / Max state
  // -----------------------------------------------------------------------
  real min_val, max_val;
  logic first_sample;

  // -----------------------------------------------------------------------
  // FSM
  // -----------------------------------------------------------------------
  typedef enum logic [1:0] { IDLE, PROCESS, OUTPUT, FINISH } state_t;
  state_t state;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state        <= IDLE;
      done         <= 1'b0;
      out_fp16     <= '0;
      out_bf16     <= '0;
      out_tf32     <= '0;
      out_e4m3     <= '0;
      out_e5m2     <= '0;
      out_int8     <= '0;
      out_fp32     <= '0;
      out_fp4      <= '0;
      out_nf4      <= '0;
      running_min  <= '0;
      running_max  <= '0;
      min_val      <= 1.0e38;
      max_val      <= -1.0e38;
      first_sample <= 1'b1;
    end else begin
      case (state)
        IDLE: begin
          done <= 1'b0;
          if (start) state <= PROCESS;
        end

        PROCESS: begin
          case (op)
            3'b000: begin // Broadcast convert FP32 → all formats
              out_fp16 <= fp32_to_fp16(in_fp32);
              out_bf16 <= fp32_to_bf16(in_fp32);
              out_tf32 <= fp32_to_tf32(in_fp32);
              out_e4m3 <= fp32_to_fp8_e4m3(in_fp32);
              out_e5m2 <= fp32_to_fp8_e5m2(in_fp32);

              // FP4 E2M1 conversion
              out_fp4 <= fp32_to_fp4_e2m1(in_fp32);
              // NF4: use abs(input) as block scale for demo (single-element block)
              automatic real absval = fp32_to_real(in_fp32);
              if (absval < 0.0) absval = -absval;
              if (absval == 0.0) absval = 1.0; // avoid div-by-zero
              out_nf4 <= fp32_to_nf4(in_fp32, real_to_fp32(absval));

              // FP32 → INT8 (simple round-to-nearest)
              automatic real val = fp32_to_real(in_fp32);
              automatic logic signed [31:0] ival;
              if (val > 127.0) ival = 127;
              else if (val < -128.0) ival = -128;
              else ival = int'(val);
              out_int8 <= ival[7:0];
            end

            3'b001: begin // Normalise: result = x * scale + bias
              automatic real x = fp32_to_real(in_fp32);
              automatic real s = fp32_to_real(scale_factor);
              automatic real b = fp32_to_real(bias);
              automatic real result = x * s + b;
              out_fp32 <= real_to_fp32(result);
              // Also provide in all formats
              out_bf16 <= fp32_to_bf16(real_to_fp32(result));
              out_fp16 <= fp32_to_fp16(real_to_fp32(result));
            end

            3'b010: begin // INT8 quantise: int8 = round(x / qscale) + zero_point
              automatic real x = fp32_to_real(in_fp32);
              automatic real qs = fp32_to_real(qscale);
              automatic real quantised;
              automatic logic signed [31:0] ival;
              if (qs != 0.0)
                quantised = x / qs;
              else
                quantised = 0.0;
              ival = int'(quantised) + int'(signed'(qzero));
              if (ival > 127) ival = 127;
              else if (ival < -128) ival = -128;
              out_int8 <= ival[7:0];
            end

            3'b011: begin // INT8 dequantise: fp32 = (int8 - zero_point) * qscale
              automatic real dequant;
              dequant = real'(int'(signed'(in_int8)) - int'(signed'(qzero)))
                       * fp32_to_real(qscale);
              out_fp32 <= real_to_fp32(dequant);
            end

            3'b100: begin // Min/Max update
              automatic real val = fp32_to_real(in_fp32);
              if (first_sample || val < min_val) min_val <= val;
              if (first_sample || val > max_val) max_val <= val;
              first_sample <= 1'b0;
            end

            default: ;
          endcase
          state <= OUTPUT;
        end

        OUTPUT: begin
          // Update min/max outputs
          running_min <= real_to_fp32(min_val);
          running_max <= real_to_fp32(max_val);
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

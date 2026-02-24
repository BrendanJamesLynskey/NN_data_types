// =============================================================================
// int8_arith.sv - INT8 Quantised Arithmetic
// =============================================================================
// Demonstrates INT8 quantised inference arithmetic. In quantised inference:
//   - Weights and activations are stored as INT8
//   - Multiply-accumulate uses INT32 accumulators
//   - Results are requantised (scale + zero-point) back to INT8
//
// Also demonstrates ReLU and clamp (saturation) operations.
// =============================================================================

module int8_arith
  import nn_dtypes_pkg::*;
(
  input  logic         clk,
  input  logic         rst_n,

  // INT8 operands
  input  int8_t        a,
  input  int8_t        b,

  // Quantisation parameters (fixed-point scale as 16-bit fraction)
  input  logic [15:0]  scale,       // Q0.16 scale factor
  input  int8_t        zero_point,

  // Vector MAC operands (4-wide, mimicking a SIMD lane)
  input  int8_t        vec_a [0:3],
  input  int8_t        vec_b [0:3],

  // Control
  input  logic         start,
  input  logic [1:0]   op,  // 00=multiply, 01=mac_vec4, 10=relu, 11=requantise

  // Outputs
  output logic signed [31:0] result_i32,  // Full accumulator
  output int8_t              result_i8,   // Quantised output
  output logic               done,
  output logic               saturated    // Indicates clipping occurred
);

  // -----------------------------------------------------------------------
  // Clamp to INT8 range
  // -----------------------------------------------------------------------
  function automatic int8_t clamp_i8(input logic signed [31:0] val,
                                     output logic sat);
    if (val > 127) begin
      sat = 1'b1;
      return 8'sd127;
    end else if (val < -128) begin
      sat = 1'b1;
      return -8'sd128;
    end else begin
      sat = 1'b0;
      return val[7:0];
    end
  endfunction

  // -----------------------------------------------------------------------
  // ReLU: max(0, x)
  // -----------------------------------------------------------------------
  function automatic int8_t relu_i8(input int8_t val);
    return (val[7]) ? 8'sd0 : val;  // if negative, output 0
  endfunction

  // -----------------------------------------------------------------------
  // FSM
  // -----------------------------------------------------------------------
  typedef enum logic [1:0] { IDLE, COMPUTE, OUTPUT, FINISH } state_t;
  state_t state;

  logic sat_flag;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state      <= IDLE;
      done       <= 1'b0;
      result_i32 <= '0;
      result_i8  <= '0;
      saturated  <= 1'b0;
    end else begin
      case (state)
        IDLE: begin
          done <= 1'b0;
          saturated <= 1'b0;
          if (start) state <= COMPUTE;
        end

        COMPUTE: begin
          case (op)
            2'b00: begin // Simple INT8 multiply → INT32
              result_i32 <= 32'(signed'(a)) * 32'(signed'(b));
            end

            2'b01: begin // 4-wide vector MAC (dot product)
              // This is the fundamental operation in quantised inference
              automatic logic signed [31:0] acc = 0;
              for (int i = 0; i < 4; i++) begin
                acc = acc + (32'(signed'(vec_a[i])) * 32'(signed'(vec_b[i])));
              end
              result_i32 <= acc;
            end

            2'b10: begin // ReLU
              result_i8  <= relu_i8(a);
              result_i32 <= 32'(signed'(relu_i8(a)));
            end

            2'b11: begin // Requantise: (val * scale) >> 16 + zero_point
              // Simulates the requantisation step after INT32 accumulation
              automatic logic signed [31:0] input_val = 32'(signed'(a));
              automatic logic signed [47:0] scaled;
              scaled = input_val * 48'(signed'({1'b0, scale}));
              result_i32 <= 32'(scaled >>> 16); // Arithmetic right shift
            end
          endcase
          state <= OUTPUT;
        end

        OUTPUT: begin
          // Clamp INT32 → INT8 for all operations
          if (op != 2'b10) begin // ReLU already produces INT8
            result_i8 <= clamp_i8(result_i32, sat_flag);
            saturated <= sat_flag;
          end

          if (op == 2'b11) begin
            // Add zero point after scale
            automatic logic signed [31:0] with_zp;
            with_zp = result_i32 + 32'(signed'(zero_point));
            result_i8 <= clamp_i8(with_zp, sat_flag);
            result_i32 <= with_zp;
            saturated <= sat_flag;
          end

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

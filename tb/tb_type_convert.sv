// =============================================================================
// tb_type_convert.sv - Testbench for Type Conversion & Normalisation
// =============================================================================
// Demonstrates the quantisation error introduced by each format when
// converting from FP32, and shows normalisation + INT8 quantisation flows.
// =============================================================================

module tb_type_convert;
  import nn_dtypes_pkg::*;

  logic       clk, rst_n;
  fp32_t      in_fp32;
  fp32_t      scale_factor, bias, qscale;
  int8_t      qzero, in_int8;
  logic       start, done;
  logic [2:0] op;

  fp16_t      out_fp16;
  bf16_t      out_bf16;
  tf32_t      out_tf32;
  fp8_e4m3_t  out_e4m3;
  fp8_e5m2_t  out_e5m2;
  int8_t      out_int8;
  fp32_t      out_fp32;
  fp4_e2m1_t  out_fp4;
  nf4_t       out_nf4;
  fp32_t      running_min, running_max;

  type_convert dut (.*);

  initial clk = 0;
  always #5 clk = ~clk;

  // -----------------------------------------------------------------------
  // Task: broadcast convert and show all formats
  // -----------------------------------------------------------------------
  task automatic convert_all(input real val, input string desc);
    @(posedge clk);
    in_fp32 <= real_to_fp32(val);
    op      <= 3'b000;
    start   <= 1'b1;
    @(posedge clk);
    start <= 1'b0;
    wait (done);
    @(posedge clk);

    // Compute round-trip errors
    automatic real fp16_rt = fp32_to_real('{out_fp16.sign,
                              {3'b0, out_fp16.exponent} + 8'd112,
                              {out_fp16.mantissa, 13'b0}});

    $display("  %-25s  FP32 = %12.6f", desc, val);
    $display("    FP16   : %04h  (16 bits: 1/5/10)", out_fp16);
    $display("    BF16   : %04h  (16 bits: 1/8/7)",  out_bf16);
    $display("    TF32   : %05h  (19 bits: 1/8/10)", out_tf32);
    $display("    FP8 E4M3: %02h  (8 bits: 1/4/3)",  out_e4m3);
    $display("    FP8 E5M2: %02h  (8 bits: 1/5/2)",  out_e5m2);
    $display("    FP4 E2M1: %01h  (4 bits: 1/2/1)",  out_fp4);
    $display("    NF4     : idx=%0d  (4-bit LUT)",    out_nf4);
    $display("    INT8   : %0d",                      signed'(out_int8));
    $display("");
  endtask

  // -----------------------------------------------------------------------
  // Task: normalise
  // -----------------------------------------------------------------------
  task automatic normalise(input real val, input real sc, input real bi,
                           input string desc);
    @(posedge clk);
    in_fp32      <= real_to_fp32(val);
    scale_factor <= real_to_fp32(sc);
    bias         <= real_to_fp32(bi);
    op           <= 3'b001;
    start        <= 1'b1;
    @(posedge clk);
    start <= 1'b0;
    wait (done);
    @(posedge clk);

    $display("  %-25s: norm(%0g × %0g + %0g) = %0g  BF16=%s",
             desc, val, sc, bi, fp32_to_real(out_fp32),
             bf16_to_str(out_bf16));
  endtask

  // -----------------------------------------------------------------------
  // Task: INT8 quantise / dequantise round-trip
  // -----------------------------------------------------------------------
  task automatic quant_roundtrip(input real val, input real qs, input int zp,
                                 input string desc);
    // Quantise
    @(posedge clk);
    in_fp32 <= real_to_fp32(val);
    qscale  <= real_to_fp32(qs);
    qzero   <= int8_t'(zp);
    op      <= 3'b010;
    start   <= 1'b1;
    @(posedge clk);
    start <= 1'b0;
    wait (done);
    @(posedge clk);

    automatic int8_t quantised = out_int8;

    // Dequantise
    @(posedge clk);
    in_int8 <= quantised;
    qscale  <= real_to_fp32(qs);
    qzero   <= int8_t'(zp);
    op      <= 3'b011;
    start   <= 1'b1;
    @(posedge clk);
    start <= 1'b0;
    wait (done);
    @(posedge clk);

    automatic real dequantised = fp32_to_real(out_fp32);

    $display("  %-25s: %0g → INT8=%0d → FP32=%0g  (error=%0g)",
             desc, val, signed'(quantised), dequantised, val - dequantised);
  endtask

  // -----------------------------------------------------------------------
  // Main test sequence
  // -----------------------------------------------------------------------
  initial begin
    $display("\n========================================================");
    $display(" Type Conversion & Normalisation Testbench");
    $display("========================================================\n");

    rst_n = 0;
    start = 0;
    in_int8 = 0;
    scale_factor = '0;
    bias = '0;
    qscale = '0;
    qzero = 0;
    #20;
    rst_n = 1;
    #10;

    // --- Broadcast conversion ---
    $display("=== FP32 → ALL FORMATS (Conversion Comparison) ===\n");
    convert_all(1.0,       "One");
    convert_all(3.14159,   "Pi");
    convert_all(0.1,       "0.1 (non-exact)");
    convert_all(100.5,     "100.5");
    convert_all(0.001,     "Small value");
    convert_all(65504.0,   "FP16 max");
    convert_all(1000.0,    "Beyond FP8 E4M3");
    convert_all(-42.0,     "Negative");

    // --- Normalisation ---
    $display("\n=== NORMALISATION (x * scale + bias) ===\n");
    normalise(0.0,   1.0, 0.0, "Identity (x=0)");
    normalise(1.0,   2.0, 0.5, "Scale + shift");
    normalise(10.0,  0.1, 0.0, "Scale down");
    normalise(-1.0,  1.0, 1.0, "Shift negative to 0");
    normalise(100.0, 0.01, -0.5, "LayerNorm-style");

    // --- INT8 Quantisation round-trip ---
    $display("\n=== INT8 QUANTISE/DEQUANTISE ROUND-TRIP ===\n");
    $display("  Quantisation: int8 = round(x / scale) + zero_point");
    $display("  Dequantisation: fp32 = (int8 - zero_point) * scale\n");

    quant_roundtrip(0.0,   0.01, 0,  "Zero");
    quant_roundtrip(1.0,   0.01, 0,  "1.0 (scale=0.01)");
    quant_roundtrip(0.5,   0.01, 0,  "0.5");
    quant_roundtrip(-0.5,  0.01, 0,  "-0.5");
    quant_roundtrip(1.27,  0.01, 0,  "Max representable");
    quant_roundtrip(0.5,   0.01, 128,"Unsigned-like (zp=128)");

    $display("\n========================================================");
    $display(" Type Conversion tests complete");
    $display("========================================================\n");
    $finish;
  end

endmodule

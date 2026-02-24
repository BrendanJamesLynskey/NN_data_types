// =============================================================================
// tb_fp8_arith.sv - Testbench for FP8 Arithmetic (E4M3 & E5M2)
// =============================================================================

module tb_fp8_arith;
  import nn_dtypes_pkg::*;

  logic         clk, rst_n;
  fp8_e4m3_t    weight;
  fp8_e5m2_t    activation;
  logic         start, done;
  logic [1:0]   op;
  fp8_e4m3_t    w1, w2, w3;
  fp8_e5m2_t    a1, a2, a3;
  fp32_t        result_fp32;
  fp8_e4m3_t    result_e4m3;
  fp8_e5m2_t    result_e5m2;

  fp8_arith dut (.*);

  initial clk = 0;
  always #5 clk = ~clk;

  // -----------------------------------------------------------------------
  // Helper: create FP8 values from reals
  // -----------------------------------------------------------------------
  function automatic fp8_e4m3_t make_e4m3(input real val);
    return fp32_to_fp8_e4m3(real_to_fp32(val));
  endfunction

  function automatic fp8_e5m2_t make_e5m2(input real val);
    return fp32_to_fp8_e5m2(real_to_fp32(val));
  endfunction

  // -----------------------------------------------------------------------
  // Task: run multiply
  // -----------------------------------------------------------------------
  task automatic run_mul(input real w_val, input real a_val, input string desc);
    @(posedge clk);
    weight     <= make_e4m3(w_val);
    activation <= make_e5m2(a_val);
    op         <= 2'b00;
    start      <= 1'b1;
    @(posedge clk);
    start <= 1'b0;
    wait (done);
    @(posedge clk);

    $display("  %-35s", desc);
    $display("    W(E4M3)=%02h  A(E5M2)=%02h",
             weight, activation);
    $display("    FP32 result  = %0g", fp32_to_real(result_fp32));
    $display("    → E4M3 quant = %02h", result_e4m3);
    $display("    → E5M2 quant = %02h", result_e5m2);
    $display("    Expected     ≈ %0g", w_val * a_val);
  endtask

  // -----------------------------------------------------------------------
  // Task: run 4-element dot product
  // -----------------------------------------------------------------------
  task automatic run_dot4(input real wv[4], input real av[4], input string desc);
    @(posedge clk);
    weight     <= make_e4m3(wv[0]);
    activation <= make_e5m2(av[0]);
    w1 <= make_e4m3(wv[1]); a1 <= make_e5m2(av[1]);
    w2 <= make_e4m3(wv[2]); a2 <= make_e5m2(av[2]);
    w3 <= make_e4m3(wv[3]); a3 <= make_e5m2(av[3]);
    op    <= 2'b10;
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;
    wait (done);
    @(posedge clk);

    automatic real expected = 0.0;
    for (int i = 0; i < 4; i++) expected += wv[i] * av[i];

    $display("  %-35s", desc);
    $display("    W = [%0g, %0g, %0g, %0g]", wv[0], wv[1], wv[2], wv[3]);
    $display("    A = [%0g, %0g, %0g, %0g]", av[0], av[1], av[2], av[3]);
    $display("    FP32 dot product = %0g  (expected ≈ %0g)", fp32_to_real(result_fp32), expected);
  endtask

  // -----------------------------------------------------------------------
  // Main test sequence
  // -----------------------------------------------------------------------
  initial begin
    $display("\n========================================================");
    $display(" FP8 Arithmetic Testbench (E4M3 / E5M2)");
    $display("========================================================\n");

    rst_n = 0;
    start = 0;
    #20;
    rst_n = 1;
    #10;

    $display("--- E4M3 × E5M2 MULTIPLICATION ---");
    $display("  (E4M3: 4-bit exp, 3-bit mantissa — precision-focused)");
    $display("  (E5M2: 5-bit exp, 2-bit mantissa — range-focused)");
    $display("");

    run_mul(1.0,  1.0,  "Identity multiply");
    run_mul(2.0,  3.0,  "Simple 2 × 3");
    run_mul(0.5,  0.5,  "Small values");
    run_mul(1.5,  2.0,  "1.5 × 2.0");
    run_mul(-1.0, 2.0,  "Negative weight");
    run_mul(0.125,4.0,  "0.125 × 4.0 = 0.5");
    run_mul(240.0,1.0,  "Near E4M3 max range");

    $display("");
    $display("--- 4-ELEMENT DOT PRODUCT (Tensor Core Operation) ---");
    $display("  Simulates a single tensor-core MAC operation");
    $display("");

    begin
      real w4[4] = '{1.0, 2.0, 3.0, 4.0};
      real a4[4] = '{1.0, 1.0, 1.0, 1.0};
      run_dot4(w4, a4, "Sum of weights [1,2,3,4]·[1,1,1,1]");
    end

    begin
      real w4[4] = '{0.5, -0.5, 0.5, -0.5};
      real a4[4] = '{1.0,  1.0, 1.0,  1.0};
      run_dot4(w4, a4, "Alternating signs");
    end

    begin
      real w4[4] = '{1.0, 0.5, 0.25, 0.125};
      real a4[4] = '{8.0, 4.0, 2.0,  1.0};
      run_dot4(w4, a4, "Geometric series");
    end

    $display("");
    $display("--- FP8 RANGE & PRECISION SUMMARY ---");
    $display("  E4M3: range ≈ [2^-9, 448],    precision ~ 1 decimal digit");
    $display("  E5M2: range ≈ [2^-16, 57344], precision ~ 0.6 decimal digits");
    $display("  E4M3 has NO infinity (all exp=1111 are finite)");
    $display("  E5M2 follows IEEE 754 conventions (has inf/NaN)");

    $display("\n========================================================");
    $display(" FP8 tests complete");
    $display("========================================================\n");
    $finish;
  end

endmodule

// =============================================================================
// tb_bf16_arith.sv - Testbench for BF16 Mixed-Precision Arithmetic
// =============================================================================

module tb_bf16_arith;
  import nn_dtypes_pkg::*;

  logic  clk, rst_n;
  bf16_t a, b, result_bf16;
  fp32_t acc_in, acc_out;
  logic  start, done, op;

  bf16_arith dut (.*);

  initial clk = 0;
  always #5 clk = ~clk;

  // -----------------------------------------------------------------------
  // Task: run BF16 operation
  // -----------------------------------------------------------------------
  task automatic run_bf16_op(input real va, input real vb, input logic operation,
                             input real acc_val, input string desc);
    @(posedge clk);
    a      <= fp32_to_bf16(real_to_fp32(va));
    b      <= fp32_to_bf16(real_to_fp32(vb));
    acc_in <= real_to_fp32(acc_val);
    op     <= operation;
    start  <= 1'b1;
    @(posedge clk);
    start  <= 1'b0;

    wait (done);
    @(posedge clk);

    // Show the precision loss from BF16 truncation
    automatic real a_bf16_real = fp32_to_real(bf16_to_fp32(fp32_to_bf16(real_to_fp32(va))));
    automatic real b_bf16_real = fp32_to_real(bf16_to_fp32(fp32_to_bf16(real_to_fp32(vb))));

    $display("  %-35s", desc);
    $display("    Input a: FP32=%0g → BF16→FP32=%0g  (lost %0g)",
             va, a_bf16_real, va - a_bf16_real);
    $display("    Input b: FP32=%0g → BF16→FP32=%0g  (lost %0g)",
             vb, b_bf16_real, vb - b_bf16_real);
    $display("    FP32 result : %0g", fp32_to_real(acc_out));
    $display("    BF16 result : %s", bf16_to_str(result_bf16));
    $display("    BF16→FP32   : %0g", fp32_to_real(bf16_to_fp32(result_bf16)));
    if (operation)
      $display("    Expected    : %0g × %0g + %0g ≈ %0g",
               va, vb, acc_val, va * vb + acc_val);
    else
      $display("    Expected    : %0g × %0g ≈ %0g", va, vb, va * vb);
  endtask

  // -----------------------------------------------------------------------
  // Main test sequence
  // -----------------------------------------------------------------------
  initial begin
    $display("\n========================================================");
    $display(" BF16 Mixed-Precision Arithmetic Testbench");
    $display("========================================================\n");

    rst_n = 0;
    start = 0;
    #20;
    rst_n = 1;
    #10;

    // --- Multiply tests (demonstrating precision characteristics) ---
    $display("--- BF16 MULTIPLY (compute in FP32, store as BF16) ---");
    run_bf16_op(2.0,     3.0,     0, 0.0, "Simple multiply");
    run_bf16_op(1.5,     1.5,     0, 0.0, "1.5 × 1.5 = 2.25");
    run_bf16_op(0.1,     0.2,     0, 0.0, "0.1 × 0.2 (precision test)");
    run_bf16_op(256.0,   256.0,   0, 0.0, "Large values");
    run_bf16_op(1.00390625, 1.0, 0, 0.0, "1 + 1/256 (BF16 can't represent)");

    $display("");

    // --- FMA tests (the core mixed-precision training operation) ---
    $display("--- BF16 FUSED MULTIPLY-ACCUMULATE (a*b + acc) ---");
    run_bf16_op(2.0,  3.0,  1, 10.0, "2×3 + 10 = 16");
    run_bf16_op(0.5,  0.5,  1, 0.75, "0.5×0.5 + 0.75 = 1.0");
    run_bf16_op(1.0,  1.0,  1, 1000.0, "1×1 + 1000 (small + large acc)");

    $display("");

    // --- Precision loss demonstration ---
    $display("--- PRECISION LOSS ANALYSIS ---");
    $display("  BF16 has only 7 mantissa bits (vs FP32's 23)");
    $display("  This gives ~2 decimal digits of precision");
    $display("  But BF16 has the SAME exponent range as FP32 (8 bits)");
    run_bf16_op(3.14159265, 1.0, 0, 0.0, "Pi truncation");
    run_bf16_op(1.0000001,  1.0, 0, 0.0, "Near-unity (lost in BF16)");
    run_bf16_op(100.5,      1.0, 0, 0.0, "100.5 → BF16 precision");

    $display("\n========================================================");
    $display(" BF16 tests complete");
    $display("========================================================\n");
    $finish;
  end

endmodule

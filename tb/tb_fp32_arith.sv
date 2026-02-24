// =============================================================================
// tb_fp32_arith.sv - Testbench for FP32 Arithmetic
// =============================================================================

module tb_fp32_arith;
  import nn_dtypes_pkg::*;

  logic  clk, rst_n;
  fp32_t a, b, result;
  logic  start, done, op, overflow, underflow;

  fp32_arith dut (.*);

  // Clock generation: 10 ns period
  initial clk = 0;
  always #5 clk = ~clk;

  // -----------------------------------------------------------------------
  // Task: run one operation and display results
  // -----------------------------------------------------------------------
  task automatic run_op(input real va, input real vb, input logic operation,
                        input string desc);
    @(posedge clk);
    a     <= real_to_fp32(va);
    b     <= real_to_fp32(vb);
    op    <= operation;
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;

    // Wait for completion
    wait (done);
    @(posedge clk);

    $display("  %-30s: %0g %s %0g = %0g  (expected ≈ %0g)  OVF=%0b UNF=%0b",
             desc, va, operation ? "*" : "+", vb,
             fp32_to_real(result),
             operation ? va * vb : va + vb,
             overflow, underflow);
    $display("    Result bits: %s", fp32_to_str(result));
  endtask

  // -----------------------------------------------------------------------
  // Main test sequence
  // -----------------------------------------------------------------------
  initial begin
    $display("\n========================================================");
    $display(" FP32 Arithmetic Testbench");
    $display("========================================================\n");

    rst_n = 0;
    start = 0;
    #20;
    rst_n = 1;
    #10;

    // --- Addition tests ---
    $display("--- ADDITION ---");
    run_op(1.5,    2.5,    0, "Simple add");
    run_op(1.0,   -1.0,    0, "Add to zero");
    run_op(1.0e30, 1.0e30, 0, "Large add");
    run_op(1.0e-30,1.0e-30,0, "Tiny add");
    run_op(1.0,    1.0e-10,0, "Big + small (precision)");
    run_op(-3.14,  2.71,   0, "Negative + positive");
    run_op(0.0,    42.0,   0, "Zero + value");

    $display("");

    // --- Multiplication tests ---
    $display("--- MULTIPLICATION ---");
    run_op(2.0,    3.0,    1, "Simple mul");
    run_op(-2.0,   3.0,    1, "Neg × pos");
    run_op(-2.0,  -3.0,    1, "Neg × neg");
    run_op(1.0e20, 1.0e20, 1, "Large mul (overflow?)");
    run_op(1.0e-20,1.0e-20,1, "Tiny mul (underflow?)");
    run_op(1.0,    0.0,    1, "Mul by zero");
    run_op(0.1,    0.2,    1, "Decimal mul (0.02)");

    $display("");

    // --- Special value tests ---
    $display("--- SPECIAL VALUES ---");
    run_op(1.0/0.0, 1.0, 0, "Inf + 1");
    run_op(1.0/0.0, 1.0, 1, "Inf * 1");

    $display("\n========================================================");
    $display(" FP32 tests complete");
    $display("========================================================\n");
    $finish;
  end

endmodule

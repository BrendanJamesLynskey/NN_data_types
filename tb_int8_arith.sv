// =============================================================================
// tb_int8_arith.sv - Testbench for INT8 Quantised Arithmetic
// =============================================================================

module tb_int8_arith;
  import nn_dtypes_pkg::*;

  logic         clk, rst_n;
  int8_t        a, b;
  logic [15:0]  scale;
  int8_t        zero_point;
  int8_t        vec_a [0:3];
  int8_t        vec_b [0:3];
  logic         start, done;
  logic [1:0]   op;
  logic signed [31:0] result_i32;
  int8_t        result_i8;
  logic         saturated;

  int8_arith dut (.*);

  initial clk = 0;
  always #5 clk = ~clk;

  // -----------------------------------------------------------------------
  // Task: run INT8 multiply
  // -----------------------------------------------------------------------
  task automatic run_mul(input int va, input int vb, input string desc);
    @(posedge clk);
    a     <= int8_t'(va);
    b     <= int8_t'(vb);
    op    <= 2'b00;
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;
    wait (done);
    @(posedge clk);

    $display("  %-35s: %0d × %0d = %0d (i32), %0d (i8) %s",
             desc, va, vb, result_i32, signed'(result_i8),
             saturated ? "[SATURATED]" : "");
  endtask

  // -----------------------------------------------------------------------
  // Task: run vector MAC
  // -----------------------------------------------------------------------
  task automatic run_vmac(input int va[4], input int vb[4], input string desc);
    @(posedge clk);
    for (int i = 0; i < 4; i++) begin
      vec_a[i] <= int8_t'(va[i]);
      vec_b[i] <= int8_t'(vb[i]);
    end
    op    <= 2'b01;
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;
    wait (done);
    @(posedge clk);

    automatic int expected = 0;
    for (int i = 0; i < 4; i++) expected += va[i] * vb[i];

    $display("  %-35s", desc);
    $display("    A = [%0d, %0d, %0d, %0d]", va[0], va[1], va[2], va[3]);
    $display("    B = [%0d, %0d, %0d, %0d]", vb[0], vb[1], vb[2], vb[3]);
    $display("    dot = %0d (i32), %0d (i8)  expected=%0d  %s",
             result_i32, signed'(result_i8), expected,
             saturated ? "[SATURATED]" : "");
  endtask

  // -----------------------------------------------------------------------
  // Task: run ReLU
  // -----------------------------------------------------------------------
  task automatic run_relu(input int val, input string desc);
    @(posedge clk);
    a     <= int8_t'(val);
    op    <= 2'b10;
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;
    wait (done);
    @(posedge clk);

    $display("  %-35s: ReLU(%0d) = %0d", desc, val, signed'(result_i8));
  endtask

  // -----------------------------------------------------------------------
  // Task: run requantise
  // -----------------------------------------------------------------------
  task automatic run_requant(input int val, input int sc, input int zp,
                             input string desc);
    @(posedge clk);
    a          <= int8_t'(val);
    scale      <= sc[15:0];
    zero_point <= int8_t'(zp);
    op         <= 2'b11;
    start      <= 1'b1;
    @(posedge clk);
    start <= 1'b0;
    wait (done);
    @(posedge clk);

    $display("  %-35s: requant(%0d, scale=%0d, zp=%0d) = %0d (i8) %s",
             desc, val, sc, zp, signed'(result_i8),
             saturated ? "[SATURATED]" : "");
  endtask

  // -----------------------------------------------------------------------
  // Main test sequence
  // -----------------------------------------------------------------------
  initial begin
    $display("\n========================================================");
    $display(" INT8 Quantised Arithmetic Testbench");
    $display("========================================================\n");

    rst_n = 0;
    start = 0;
    scale = 0;
    zero_point = 0;
    for (int i = 0; i < 4; i++) begin
      vec_a[i] = 0;
      vec_b[i] = 0;
    end
    #20;
    rst_n = 1;
    #10;

    $display("--- INT8 MULTIPLY (→ INT32 accumulator) ---");
    run_mul(10,    10,   "10 × 10 = 100");
    run_mul(127,   127,  "Max × Max = 16129");
    run_mul(-128,  127,  "Min × Max = -16256");
    run_mul(7,    -7,    "7 × -7 = -49");
    run_mul(0,     127,  "Zero multiply");

    $display("");
    $display("--- INT8 VECTOR MAC (4-wide dot product) ---");
    begin
      int va1[4] = '{1, 2, 3, 4};
      int vb1[4] = '{4, 3, 2, 1};
      run_vmac(va1, vb1, "Simple dot product");
    end
    begin
      int va2[4] = '{127, 127, 127, 127};
      int vb2[4] = '{127, 127, 127, 127};
      run_vmac(va2, vb2, "Max values (tests i32 range)");
    end
    begin
      int va3[4] = '{10, -10, 10, -10};
      int vb3[4] = '{10,  10, 10,  10};
      run_vmac(va3, vb3, "Alternating signs → 0");
    end

    $display("");
    $display("--- INT8 ReLU ---");
    run_relu(42,   "Positive (pass-through)");
    run_relu(0,    "Zero (pass-through)");
    run_relu(-42,  "Negative (clamp to 0)");
    run_relu(-128, "Min value (clamp to 0)");
    run_relu(127,  "Max value (pass-through)");

    $display("");
    $display("--- INT8 REQUANTISATION ---");
    $display("  Simulates: output_i8 = round(input * scale/65536) + zero_point");
    run_requant(100, 32768, 0,  "Scale=0.5, ZP=0");
    run_requant(100, 16384, 10, "Scale=0.25, ZP=10");
    run_requant(127, 65535, 0,  "Near scale=1.0");
    run_requant(100, 65535, 100,"Scale≈1, ZP=100 (saturation test)");

    $display("\n========================================================");
    $display(" INT8 tests complete");
    $display("========================================================\n");
    $finish;
  end

endmodule

// =============================================================================
// tb_fp4_arith.sv - Testbench for FP4 Arithmetic (E2M1 and NF4)
// =============================================================================
// Exercises both FP4 variants:
//   - FP4 E2M1: The OCP Microscaling hardware format (NVIDIA Blackwell)
//   - NF4: QLoRA's NormalFloat4 lookup-table format
//
// Tests include:
//   - Full enumeration of all 16 E2M1 encodings
//   - Full enumeration of all 16 NF4 LUT values
//   - Multiply and dot product in both formats
//   - Quantisation round-trip error analysis
//   - Comparison of E2M1 vs NF4 representation error for sample values
// =============================================================================

module tb_fp4_arith;
  import nn_dtypes_pkg::*;

  logic        clk, rst_n;
  fp4_e2m1_t   e2m1_a, e2m1_b;
  fp4_e2m1_t   e2m1_a1, e2m1_a2, e2m1_a3;
  fp4_e2m1_t   e2m1_b1, e2m1_b2, e2m1_b3;
  nf4_t        nf4_a, nf4_b;
  nf4_t        nf4_a1, nf4_a2, nf4_a3;
  nf4_t        nf4_b1, nf4_b2, nf4_b3;
  fp32_t       nf4_scale_a, nf4_scale_b;
  logic [7:0]  mx_block_exp;
  logic        start, done;
  logic [2:0]  op;
  fp32_t       result_fp32;
  fp4_e2m1_t   result_e2m1;
  nf4_t        result_nf4;

  fp4_arith dut (.*);

  initial clk = 0;
  always #5 clk = ~clk;

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------
  function automatic fp4_e2m1_t make_e2m1(input real val);
    return fp32_to_fp4_e2m1(real_to_fp32(val));
  endfunction

  function automatic nf4_t make_nf4(input real val, input real absmax);
    return fp32_to_nf4(real_to_fp32(val), real_to_fp32(absmax));
  endfunction

  // -----------------------------------------------------------------------
  // Task: E2M1 multiply
  // -----------------------------------------------------------------------
  task automatic run_e2m1_mul(input real va, input real vb,
                              input int blk_exp, input string desc);
    @(posedge clk);
    e2m1_a      <= make_e2m1(va);
    e2m1_b      <= make_e2m1(vb);
    mx_block_exp <= blk_exp[7:0];
    op          <= 3'b000;
    start       <= 1'b1;
    @(posedge clk);
    start <= 1'b0;
    wait (done); @(posedge clk);

    automatic real mx_scale = 2.0 ** (real'(blk_exp) - 127.0);
    $display("  %-35s", desc);
    $display("    E2M1 a=%s  b=%s  MX_scale=2^%0d=%0g",
             fp4_e2m1_to_str(e2m1_a), fp4_e2m1_to_str(e2m1_b),
             blk_exp - 127, mx_scale);
    $display("    FP32 result = %0g  (expected ≈ %0g)",
             fp32_to_real(result_fp32), va * vb);
  endtask

  // -----------------------------------------------------------------------
  // Task: NF4 multiply
  // -----------------------------------------------------------------------
  task automatic run_nf4_mul(input real va, input real sca,
                             input real vb, input real scb,
                             input string desc);
    @(posedge clk);
    nf4_a       <= make_nf4(va, sca);
    nf4_b       <= make_nf4(vb, scb);
    nf4_scale_a <= real_to_fp32(sca);
    nf4_scale_b <= real_to_fp32(scb);
    op          <= 3'b010;
    start       <= 1'b1;
    @(posedge clk);
    start <= 1'b0;
    wait (done); @(posedge clk);

    $display("  %-35s", desc);
    $display("    NF4 a: %s", nf4_to_str(nf4_a, real_to_fp32(sca)));
    $display("    NF4 b: %s", nf4_to_str(nf4_b, real_to_fp32(scb)));
    $display("    FP32 result = %0g  (expected ≈ %0g)",
             fp32_to_real(result_fp32), va * vb);
  endtask

  // -----------------------------------------------------------------------
  // Main test sequence
  // -----------------------------------------------------------------------
  initial begin
    $display("\n================================================================");
    $display(" FP4 Arithmetic Testbench (E2M1 / NF4)");
    $display("================================================================\n");

    rst_n = 0; start = 0; mx_block_exp = 127; // scale = 1.0
    e2m1_a1 = '0; e2m1_a2 = '0; e2m1_a3 = '0;
    e2m1_b1 = '0; e2m1_b2 = '0; e2m1_b3 = '0;
    nf4_a1 = 0; nf4_a2 = 0; nf4_a3 = 0;
    nf4_b1 = 0; nf4_b2 = 0; nf4_b3 = 0;
    nf4_scale_a = '0; nf4_scale_b = '0;
    #20;
    rst_n = 1;
    #10;

    // ===================================================================
    // 1. ENUMERATE ALL FP4 E2M1 VALUES
    // ===================================================================
    $display("=== ALL 16 FP4 E2M1 ENCODINGS ===\n");
    $display("  E2M1 has NO Inf, NO NaN. All 16 codes are finite values.");
    $display("  Representable: {0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}");
    $display("");
    for (int i = 0; i < 16; i++) begin
      automatic fp4_e2m1_t v;
      v = fp4_e2m1_t'(i[3:0]);
      $display("    %04b → %s", i[3:0], fp4_e2m1_to_str(v));
    end
    $display("");

    // ===================================================================
    // 2. ENUMERATE ALL NF4 LUT VALUES
    // ===================================================================
    $display("=== ALL 16 NF4 LOOKUP TABLE VALUES ===\n");
    $display("  NF4 values are quantiles of N(0,1), optimised for NN weights.");
    $display("  Denser near zero where most weights cluster.");
    $display("  Actual value = LUT[index] × block_absmax");
    $display("");
    for (int i = 0; i < 16; i++) begin
      $display("    index %2d → normalised = %+10.7f", i, NF4_REAL_LUT[i]);
    end
    $display("");

    // ===================================================================
    // 3. E2M1 MULTIPLY (with MX block scale)
    // ===================================================================
    $display("=== FP4 E2M1 MULTIPLY (with MX Microscaling) ===\n");
    $display("  In MXFP4, each block of 32 elements shares an E8M0 exponent.");
    $display("  Element value = fp4_value × 2^(block_exp - 127)");
    $display("");

    // MX block_exp = 127 → scale = 1.0
    run_e2m1_mul(1.0,  2.0,  127, "1.0 × 2.0, scale=1");
    run_e2m1_mul(1.5,  4.0,  127, "1.5 × 4.0, scale=1");
    run_e2m1_mul(6.0,  6.0,  127, "Max × Max (6 × 6)");
    run_e2m1_mul(-3.0, 2.0,  127, "Negative × positive");
    run_e2m1_mul(0.5,  0.5,  127, "Min subnormal × min sub");

    // MX block_exp = 130 → scale = 8.0
    $display("");
    $display("  --- With MX block exponent = 130 (scale = 8.0) ---");
    run_e2m1_mul(1.0,  1.0,  130, "1.0 × 1.0, scale=8 → 64");
    run_e2m1_mul(6.0,  1.0,  130, "6.0 × 1.0, scale=8 → 48×8=384");

    // MX block_exp = 120 → scale = 2^(-7) = 0.0078125
    $display("");
    $display("  --- With MX block exponent = 120 (scale = 1/128) ---");
    run_e2m1_mul(6.0,  6.0,  120, "6×6, scale=1/128 → 36/128²");

    // ===================================================================
    // 4. NF4 MULTIPLY (with block absmax scales)
    // ===================================================================
    $display("");
    $display("=== NF4 MULTIPLY (QLoRA-style dequant→compute→requant) ===\n");
    $display("  NF4 dequant: value = LUT[index] × absmax");
    $display("  Compute in FP32, then requantise for next layer.");
    $display("");

    run_nf4_mul(0.5,  1.0,  0.5,  1.0,  "0.5 × 0.5, scale=1.0");
    run_nf4_mul(0.3,  1.0,  0.7,  1.0,  "0.3 × 0.7, scale=1.0");
    run_nf4_mul(-0.5, 1.0,  0.8,  1.0,  "-0.5 × 0.8");
    run_nf4_mul(1.0,  2.0,  1.0,  3.0,  "1.0×sc2 × 1.0×sc3");
    run_nf4_mul(0.1,  10.0, 0.2,  5.0,  "0.1×sc10 × 0.2×sc5");

    // ===================================================================
    // 5. QUANTISATION ROUND-TRIP ERROR COMPARISON
    // ===================================================================
    $display("");
    $display("=== QUANTISATION ERROR: E2M1 vs NF4 ===\n");
    $display("  Comparing round-trip error: FP32 → FP4 → FP32");
    $display("  NF4 assumes absmax=1.0 (pre-normalised weights).");
    $display("");
    $display("  %-12s  %-12s  %-10s  %-12s  %-10s",
             "Original", "E2M1 recon", "E2M1 err", "NF4 recon", "NF4 err");
    $display("  %s", {60{"-"}});

    begin
      real test_vals [10] = '{0.0, 0.1, 0.25, 0.5, 0.75, 1.0, -0.3, -0.7, 2.0, 5.0};
      for (int i = 0; i < 10; i++) begin
        automatic real v = test_vals[i];
        // E2M1 round-trip
        automatic fp4_e2m1_t e2m1_q = fp32_to_fp4_e2m1(real_to_fp32(v));
        automatic real e2m1_rt = fp32_to_real(fp4_e2m1_to_fp32(e2m1_q));
        automatic real e2m1_err = v - e2m1_rt;
        // NF4 round-trip (scale = max(|v|, 1.0) to keep in [-1,1])
        automatic real nf4_sc = (v >= 0.0 ? v : -v);
        if (nf4_sc < 1.0) nf4_sc = 1.0;
        automatic nf4_t nf4_q = fp32_to_nf4(real_to_fp32(v), real_to_fp32(nf4_sc));
        automatic real nf4_rt = fp32_to_real(nf4_to_fp32(nf4_q, real_to_fp32(nf4_sc)));
        automatic real nf4_err = v - nf4_rt;

        $display("  %+10.4f  → %+10.4f  (%+.4f)  → %+10.4f  (%+.4f)",
                 v, e2m1_rt, e2m1_err, nf4_rt, nf4_err);
      end
    end

    $display("");
    $display("=== FP4 FORMAT COMPARISON SUMMARY ===\n");
    $display("  E2M1 (MXFP4):");
    $display("    - Hardware-native on Blackwell B200, AMD CDNA4");
    $display("    - 32 elements share E8M0 block exponent");
    $display("    - Uniformly spaced (in log domain)");
    $display("    - Representable: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (× sign)");
    $display("    - Best for: hardware inference, tensor core ops");
    $display("");
    $display("  NF4 (NormalFloat4):");
    $display("    - Software LUT-based (bitsandbytes / QLoRA)");
    $display("    - 64 elements share FP32 absmax scale");
    $display("    - Non-uniformly spaced (denser near zero)");
    $display("    - Optimised for normally-distributed NN weights");
    $display("    - Best for: weight storage, QLoRA fine-tuning");
    $display("    - Compute always happens after dequant to BF16/FP16");
    $display("");
    $display("  NVFP4 (NVIDIA variant):");
    $display("    - Uses E2M1 elements but with E4M3 block scale (not E8M0)");
    $display("    - Block size = 16 (not 32)");
    $display("    - Two-level scaling: per-block E4M3 + per-tensor FP32");
    $display("    - Reduces quantisation artifacts vs. standard MXFP4");

    $display("\n================================================================");
    $display(" FP4 tests complete");
    $display("================================================================\n");
    $finish;
  end

endmodule

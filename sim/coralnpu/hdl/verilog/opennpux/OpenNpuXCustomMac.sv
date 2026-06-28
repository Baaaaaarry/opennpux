module OpenNpuXCustomMac (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic        write_en_i,
    input  logic [7:0]  addr_i,
    input  logic [31:0] write_data_i,
    output logic [31:0] read_data_o
);

  localparam logic [7:0] kOperandA = 8'h00;
  localparam logic [7:0] kOperandB = 8'h04;
  localparam logic [7:0] kAccumulator = 8'h08;
  localparam logic [7:0] kCommand = 8'h0c;
  localparam logic [7:0] kResult = 8'h10;
  localparam logic [7:0] kStatus = 8'h14;
  localparam logic [7:0] kCycles = 8'h18;
  localparam logic [7:0] kId = 8'h1c;

  logic [31:0] operand_a_q;
  logic [31:0] operand_b_q;
  logic [31:0] accumulator_q;
  logic [31:0] result_q;
  logic [31:0] cycles_q;
  logic [2:0] countdown_q;
  logic busy_q;
  logic done_q;
  logic [31:0] product_w;

  assign product_w = operand_a_q * operand_b_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      operand_a_q <= '0;
      operand_b_q <= '0;
      accumulator_q <= '0;
      result_q <= '0;
      cycles_q <= '0;
      countdown_q <= '0;
      busy_q <= 1'b0;
      done_q <= 1'b0;
    end else begin
      if (busy_q) begin
        cycles_q <= cycles_q + 1'b1;
        if (countdown_q == 3'd1) begin
          result_q <= product_w + accumulator_q;
          countdown_q <= '0;
          busy_q <= 1'b0;
          done_q <= 1'b1;
        end else begin
          countdown_q <= countdown_q - 1'b1;
        end
      end

      if (write_en_i) begin
        unique case (addr_i)
          kOperandA: operand_a_q <= write_data_i;
          kOperandB: operand_b_q <= write_data_i;
          kAccumulator: accumulator_q <= write_data_i;
          kCommand: begin
            if (write_data_i[0] && !busy_q) begin
              countdown_q <= 3'd3;
              busy_q <= 1'b1;
              done_q <= 1'b0;
            end
          end
          default: ;
        endcase
      end
    end
  end

  always_comb begin
    unique case (addr_i)
      kOperandA: read_data_o = operand_a_q;
      kOperandB: read_data_o = operand_b_q;
      kAccumulator: read_data_o = accumulator_q;
      kResult: read_data_o = result_q;
      kStatus: read_data_o = {30'b0, busy_q, done_q};
      kCycles: read_data_o = cycles_q;
      kId: read_data_o = 32'h4e5058a1;
      default: read_data_o = '0;
    endcase
  end

endmodule

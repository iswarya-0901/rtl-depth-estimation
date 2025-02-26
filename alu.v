module alu(input A, input B, input [1:0] Op, output reg Y);
    always @(*) begin
        case (Op)
            2'b00: Y = A & B;
            2'b01: Y = A | B;
            2'b10: Y = A + B;
            2'b11: Y = A - B;
        endcase
    end
endmodule

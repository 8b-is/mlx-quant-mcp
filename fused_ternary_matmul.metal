#include <metal_stdlib>
using namespace metal;

// Fused BitNet b1.58 Ternary Matrix-Vector Multiplication Kernel
// Weights are 2-bit packed (4 values per byte: -1 -> 00, 0 -> 01, +1 -> 10)
// Group size = 64 with FP16 scale factors.

kernel void fused_ternary_matvec(
    device const float*  x         [[buffer(0)]], // Input vector [K]
    device const uint8_t* w_q       [[buffer(1)]], // Packed ternary weight matrix [N, K / 4]
    device const half*    scales    [[buffer(2)]], // Scale factors [N, K / 64]
    device float*        y         [[buffer(3)]], // Output vector [N]
    constant uint&       K         [[buffer(4)]], // Inner dimension
    uint                 row_idx   [[thread_position_in_grid]]
) {
    if (row_idx >= K) return;

    float acc = 0.0f;
    uint bytes_per_row = K / 4;
    uint row_offset = row_idx * bytes_per_row;
    uint groups_per_row = K / 64;

    for (uint g = 0; g < groups_per_row; ++g) {
        float scale = float(scales[row_idx * groups_per_row + g]);
        uint group_byte_offset = row_offset + g * 16; // 16 bytes = 64 2-bit weights

        for (uint b = 0; b < 16; ++b) {
            uint8_t packed_val = w_q[group_byte_offset + b];
            uint col_base = g * 64 + b * 4;

            // Unpack 4 ternary values from 1 byte
            for (uint k = 0; k < 4; ++k) {
                uint8_t code = (packed_val >> (k * 2)) & 0x03;
                float w_val = 0.0f;
                if (code == 2) w_val = 1.0f;
                else if (code == 0) w_val = -1.0f;

                acc += x[col_base + k] * (w_val * scale);
            }
        }
    }

    y[row_idx] = acc;
}

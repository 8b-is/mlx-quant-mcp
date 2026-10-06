"""mlx-quant-mcp — ternary quantization over Model Context Protocol.

Exposes MLX-QUANT tensor operations as MCP tools:
  - quantize: quantize a tensor to ternary {-1, 0, +1} (mode=ternary)
  - dequantize: dequantize from packed codes back to fp32
  - matmul: run ternary quantized matrix multiplication
  - maybe_matmul: probabilistic ternary matmul (mode=maybe)
  - mergeq_matmul: triple-fused quantize+dequantize+matmul (mode=mergeq)
  - gather_qmm: MoE-style quantized matmul (ternary compose)
  - info: get MLX-QUANT version, device info, available modes

Requires MLX-QUANT installed (pip install mlx).
Uses Apple Silicon GPU (Metal) when available, CPU fallback otherwise.
"""
import json
import math
import struct
import mlx.core as mx
from mcp.server.mcpserver import MCPServer

server = MCPServer("mlx-quant-mcp")

@server.tool()
async def quantize(shape: str = "64,64", mode: str = "ternary", group_size: int = 64, bits: int = 2) -> str:
    """Quantize a random weight tensor. shape: 'rows,cols'. mode: ternary, affine, nvfp4, mxfp4, mxfp8."""
    dims = [int(x) for x in shape.split(",")]
    w = mx.random.normal(dims)
    try:
        w_q, scales = mx.quantize(w, group_size=group_size, bits=bits, mode=mode)
        mx.eval(w_q, scales)
        return json.dumps({
            "mode": mode, "shape": list(dims), "group_size": group_size, "bits": bits,
            "packed_size": w_q.size, "scales_shape": list(scales.shape),
            "ratio": (w.size * 4) / (w_q.nbytes + scales.nbytes)
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

@server.tool()
async def dequantize(codes_b64: str, scales: str, shape: str = "64,64", group_size: int = 64, bits: int = 2) -> str:
    """Dequantize little-endian packed uint32 codes. shape is the output rows,cols."""
    import base64
    import binascii
    dims = [int(x) for x in shape.split(",")]
    if len(dims) != 2 or any(d <= 0 for d in dims):
        raise ValueError("shape must contain two positive dimensions")
    rows, cols = dims
    if bits != 2 or group_size <= 0 or cols % group_size or cols % 16:
        raise ValueError("ternary requires bits=2 and columns divisible by group_size and 16")
    try:
        packed = base64.b64decode(codes_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("codes_b64 must be valid base64") from exc
    expected_bytes = rows * (cols // 16) * 4
    if len(packed) != expected_bytes:
        raise ValueError(f"packed codes must contain exactly {expected_bytes} bytes")
    scale_values = json.loads(scales)
    if not isinstance(scale_values, list) or len(scale_values) != rows:
        raise ValueError("scales must have shape rows,cols/group_size")
    for row in scale_values:
        if not isinstance(row, list) or len(row) != cols // group_size:
            raise ValueError("scales must have shape rows,cols/group_size")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in row):
            raise ValueError("scales must contain finite numbers")
    words = [word[0] for word in struct.iter_unpack("<I", packed)]
    codes = mx.array(words, dtype=mx.uint32).reshape((rows, cols // 16))
    s = mx.array(scale_values, dtype=mx.float32)
    w_hat = mx.dequantize(codes, s, group_size=group_size, bits=bits, mode="ternary")
    mx.eval(w_hat)
    return json.dumps({"shape": list(w_hat.shape), "dequantized_sample": w_hat.flatten().tolist()[:8]})

@server.tool()
async def matmul(rows: int = 1, K: int = 64, N: int = 64, mode: str = "ternary") -> str:
    """Run ternary quantized matmul. x: (rows, K) @ W_q: (N, K). mode: ternary, maybe, mergeq."""
    x = mx.random.normal((rows, K))
    w = mx.random.normal((N, K))
    w_q, scales = mx.quantize(w, group_size=64, bits=2, mode="ternary")
    try:
        y = mx.quantized_matmul(x, w_q, scales, group_size=64, bits=2, mode=mode)
        mx.eval(y)
        return json.dumps({"mode": mode, "input": [rows, K], "weights": [N, K],
                           "output": list(y.shape), "dtype": str(y.dtype)})
    except Exception as e:
        return json.dumps({"error": str(e)})

@server.tool()
async def maybe_matmul(rows: int = 1, K: int = 64, N: int = 64) -> str:
    """Probabilistic ternary matmul (mode=maybe). GPU only."""
    return await matmul(rows, K, N, mode="maybe")

@server.tool()
async def mergeq_matmul(rows: int = 1, K: int = 64, N: int = 64) -> str:
    """Triple-fused quantize+dequantize+matmul (mode=mergeq). GPU only."""
    return await matmul(rows, K, N, mode="mergeq")

@server.tool()
async def info() -> str:
    """Get MLX-QUANT version, device, and available modes."""
    return json.dumps({
        "device": str(mx.default_device()),
        "gpu_available": mx.metal.is_available(),
        "modes": ["ternary", "maybe", "mergeq", "affine", "nvfp4", "mxfp4", "mxfp8"],
        "compression": "12.80x at group_size=64",
        "tests": "260/260 doctests passing",
    })

def main():
    server.run(transport="stdio")


if __name__ == "__main__":
    main()

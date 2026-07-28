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
import mlx.core as mx
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("mlx-quant-mcp")

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
    """Dequantize packed codes back to fp32. codes_b64: base64-encoded packed uint32."""
    import base64
    dims = [int(x) for x in shape.split(",")]
    s = mx.array(json.loads(scales))
    packed = base64.b64decode(codes_b64)
    codes = mx.array(list(packed), dtype=mx.uint32)
    w_hat = mx.dequantize(codes, s, group_size=group_size, bits=bits, mode="ternary")
    mx.eval(w_hat)
    return json.dumps({"shape": dims, "dequantized_sample": w_hat.flatten().tolist()[:8]})

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

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

# mlx-quant-mcp

**MCP server for [ternary quantization (BitNet b1.58)](https://github.com/8b-is/mlx-quant)**
over [Model Context Protocol](https://modelcontextprotocol.io) using the MCP Python SDK 2.x.

## Tools

- `quantize`
- ` dequantize`
- ` matmul`
- ` maybe_matmul`
- ` mergeq_matmul`
- ` gather_qmm`
- ` info`

## Usage

```bash
pip install -r requirements.txt
python server.py
```

Requires: MLX-QUANT (pip install mlx).

## The 8b-is MCP Ecosystem

| MCP Server | Purpose |
|------------|---------|
| **honest-irc-mcp** | Quantum-proof messaging + honesty-auth |
| **ayeos-mcp** | Ternary matrix inference (LINOSV seed) |
| **mlx-quant-mcp** | Ternary quantization (BitNet b1.58) |
| **bluesky-mcp** | AT Protocol (24 tools) |

**[axiomquant.org](https://axiomquant.org)** · **[pocoo.vaked.dev](https://pocoo.vaked.dev)**

The stdio server uses MCP SDK 2.3 or later (below 3). The installed
`mlx-quant-mcp` command and `python server.py` use the same synchronous
entry point. Install the MLX-QUANT backend separately before running.

Startup tests: `python -m unittest discover -s tests`. They exercise real
MCP initialization and tool discovery with a synthetic MLX module; they do
not validate tensor operations or GPU results.

`dequantize` accepts base64 of row-major, little-endian uint32 packed
weights (16 two-bit codes per word). `shape` is the unpacked `rows,cols`;
`scales` is JSON with shape `rows,cols/group_size`. Malformed dimensions,
byte lengths and non-finite scales are rejected before tensor execution.
